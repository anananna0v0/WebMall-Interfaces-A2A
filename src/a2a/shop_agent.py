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

import re

def normalize_url(url: str) -> str:
    """Normalize URL for comparison."""
    if not url or not isinstance(url, str):
        return ""
    return url.rstrip('/').lower()

def extract_urls_from_response(response_text: str) -> set[str]:
    """Robust URL extraction ported from nlweb_mcp."""
    if not isinstance(response_text, str):
        return set()
    try:
        data = json.loads(response_text.strip())
        if isinstance(data, dict) and "urls" in data:
            return set(u for u in data["urls"] if isinstance(u, str))
        elif isinstance(data, list):
            return set(u for u in data if isinstance(u, str))
    except (json.JSONDecodeError, TypeError):
        pass
    
    json_pattern = r'\{"urls":\s*\[.*?\]\}'
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    for match in json_matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "urls" in data:
                return set(u for u in data["urls"] if isinstance(u, str))
        except: continue
            
    urls_found = re.findall(r'https?://\S+', response_text)
    return set([url.strip(')>."\',') for url in urls_found])

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
        
        self.llm = ChatOpenAI(model="gpt-5-mini")
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
            f"You are an advanced e-commerce Shop Agent for {self.shop_id}.\n\n"
            "TASK-SPECIFIC INSTRUCTIONS:\n"
            "- Extract precise keywords from the user wish and use the 'search_inventory' tool.\n"
            "- If no products are found, you may try ONLY ONCE with simpler or broader keywords.\n"
            "- After one retry, if still no results are found, you MUST STOP and return an empty list.\n"
            "- If the user asks for 'cheapest', only return the product(s) with the absolute lowest price.\n"
            "- Ensure the products strictly align with the user's requirements (color, model, specs).\n\n"
            "RESPONSE FORMAT REQUIREMENTS:\n"
            "- Your final response MUST be a valid JSON object ONLY.\n"
            "- Required format: {{\"urls\": [\"url1\", \"url2\", ...]}}\n"
            "- Do NOT include any explanatory text, conversational filler, or Markdown code blocks (e.g., ```json).\n"
            "- If no matches exist, return: {{\"urls\": []}}"
        )

        # Change 'state_modifier' to 'prompt' to fix the TypeError
        return create_react_agent(
            model=self.llm,
            tools=[search_inventory],
            prompt=system_message 
        )

    async def process_message(self, wish: str):
        """Processes the buyer wish with robust URL extraction."""
        # Force the model to think about JSON structure
        inputs = {"messages": [("user", f"{wish}\n\nIMPORTANT: Your final response must be a JSON object: {{\"urls\": [\"URL_HERE\"]}}")]}
        
        try:
            result = await self.agent_executor.ainvoke(
                inputs, 
                config={"recursion_limit": 25}
            )
            
            agent_output = result["messages"][-1].content
            
            # Use the ported robust extractor
            urls_set = extract_urls_from_response(agent_output)
            urls = list(urls_set)
            
            # Simple token tracking fallback
            usage = {"prompt_tokens": 0, "completion_tokens": 0}
            return urls, usage
            
        except Exception as e:
            print(f"⚠️ Agent failed: {e}")
            return [], {"error": str(e)}


app = FastAPI()

# Global variable to hold the agent instance
# This will be initialized in the main block below
shop_instance: ShopAgentInstance = None
@app.post("/messages")
async def handle_a2a_request(request: Request):
    """Standard A2A JSON-RPC interface with error protection."""
    try:
        body = await request.json()
        wish = body.get("params", {}).get("query", "")
        
        # Process via LangGraph
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
    except Exception as e:
        # --- Change: Ensure error response follows JSON-RPC 2.0 ---
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": str(e)},
            "id": None
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