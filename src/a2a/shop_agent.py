import sys
import os
import json
import logging
import re
import argparse
import uvicorn
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv 

# --- LangChain & LangGraph Imports ---
from langchain_openai import ChatOpenAI 
from langchain_core.tools import tool 
from langchain_community.callbacks import get_openai_callback 
from langgraph.prebuilt import create_react_agent 

# --- Path Injection: Must happen BEFORE importing local packages ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.dirname(current_dir)

# Ensure 'src' is the first place Python looks for modules
if src_path not in sys.path:
    sys.path.insert(0, src_path)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(src_path), ".env"))

# --- Local Package Imports: Clean and grouped ---
try:
    from nlweb_mcp.search_engine import SearchEngine
    from nlweb_mcp.woocommerce_client import WooCommerceClient
    print(f"✅ Successfully loaded local modules from: {src_path}")
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Ensure you have these imports at the top ---
from nlweb_mcp.elasticsearch_client import ElasticsearchClient
from nlweb_mcp.embedding_service import EmbeddingService
# Keep your existing imports...

class ShopAgentInstance:
    def __init__(self, shop_id: str, index_name: str):
        """
        Initializes the Shop Agent with necessary search and embedding services.
        """
        self.shop_id = shop_id
        self.index_name = index_name

        # --- 2. Initialize required services for SearchEngine ---
        # These services rely on your .env settings (ES_URL, OPENAI_API_KEY, etc.)
        self.es_client = ElasticsearchClient()
        self.embedding_service = EmbeddingService()

        # --- 3. Correct SearchEngine initialization ---
        # Pass the missing positional arguments as required by the library
        self.search_engine = SearchEngine(
            elasticsearch_client=self.es_client,
            embedding_service=self.embedding_service,
            index_name=index_name
        )
        
        # WooCommerce and LLM setup
        # Note: Ensure woocommerce_client.py matches the filename we discussed
        from nlweb_mcp.woocommerce_client import WooCommerceClient
        shop_url = f"https://{shop_id.replace('_', '-')}.informatik.uni-mannheim.de"
        self.woo_client = WooCommerceClient(base_url=shop_url)
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent_executor = self._setup_agent()

    def _setup_agent(self):
        """Sets up the LangGraph ReAct agent with shop-specific tools."""
        
        @tool
        def search_inventory(query: str) -> str:
            """
            Search the shop's local database for products. 
            Returns a JSON string of found products.
            """
            raw_results = self.search_engine.search(query=query)
            items = raw_results.get("marketplace_inventory", [])
            
            if not items:
                return "No products found for this query. Try broader or different keywords."
            
            return json.dumps([
                {
                    "name": i.get("name"),
                    "price": i.get("price"),
                    "url": i.get("url"),
                    "description": i.get("description")[:100]
                } for i in items
            ])

        system_message = (
            f"You are the Shop Agent for {self.shop_id}. "
            "1. Extract keywords. 2. Search inventory. "
            "3. If no results found, try ONCE with simpler keywords. " # Limit retries
            "4. If still no results, state 'No products found' and stop. " # Define a terminal state
            "5. Return a JSON list of product URLs."
        )

        # Change 'state_modifier' to 'prompt' to fix the TypeError
        return create_react_agent(
            model=self.llm,
            tools=[search_inventory],
            prompt=system_message 
        )

    async def process_message(self, wish: str):
        """Processes the buyer wish using the LangGraph agent."""
        inputs = {"messages": [("user", wish)]}
        
        try:
            # Add a config dictionary to control recursion_limit
            result = await self.agent_executor.ainvoke(
                inputs, 
                config={"recursion_limit": 15} # Set a hard stop at 15 steps
            )
            
            # The last message in the state contains the agent's output
            agent_output = result["messages"][-1].content
            
            # (Rest of your regex logic to extract URLs...)
            return urls, usage
        except Exception as e:
            print(f"Agent failed or hit recursion limit: {e}")
            return [], {"prompt_tokens": 0, "completion_tokens": 0}


app = FastAPI()

# Global variable to hold the agent instance
# This will be initialized in the main block below
shop_instance: ShopAgentInstance = None
@app.post("/messages")
async def handle_a2a_request(request: Request):
    """Standard A2A JSON-RPC interface."""
    if shop_instance is None:
        return {"jsonrpc": "2.0", "error": {"code": -32000, "message": "Agent not initialized"}, "id": None}
        
    body = await request.json()
    wish = body.get("params", {}).get("query", "")
    
    urls, usage = await shop_instance.process_message(wish)
    
    return {
        "jsonrpc": "2.0",
        "result": {
            "agent_name": shop_instance.shop_id,
            "offers": urls,
            "tokens": usage
        },
        "id": body.get("id")
    }

# --- CLI Entry Point ---
# This part allows run_a2a_exp.sh to pass --shop_id and --port
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2A Shop Agent Server")
    parser.add_argument("--shop_id", type=str, required=True, help="ID of the shop (e.g., webmall_1)")
    parser.add_argument("--port", type=int, required=True, help="Port to run the server on")
    
    args = parser.parse_args()

    # Initialize the shop instance with CLI arguments
    # The index name is automatically derived from shop_id
    shop_instance = ShopAgentInstance(
        shop_id=args.shop_id,
        index_name=f"{args.shop_id}_nlweb"
    )

    print(f"🚀 Starting {args.shop_id} on port {args.port}...")
    
    # Run uvicorn with the parsed port
    uvicorn.run(app, host="0.0.0.0", port=args.port)