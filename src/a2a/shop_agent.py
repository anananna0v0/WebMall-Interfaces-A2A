import sys
import os
import json
import logging
import argparse
import uvicorn
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv 

# LangChain & LangGraph Imports
from langchain_openai import ChatOpenAI 
from langchain_core.tools import tool 
from langchain_community.callbacks import get_openai_callback 
from langgraph.prebuilt import create_react_agent 

# Path Injection and Environment Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.dirname(current_dir)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(src_path), ".env"))

# Local Package Imports
from nlweb_mcp.search_engine import SearchEngine
from nlweb_mcp.elasticsearch_client import ElasticsearchClient
from nlweb_mcp.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShopAgentInstance:
    def __init__(self, shop_id: str, index_name: str):
        """
        Initializes the Shop Agent with search capabilities and LLM.
        """
        self.shop_id = shop_id
        self.index_name = index_name

        # Initialize core services for product retrieval
        self.es_client = ElasticsearchClient()
        self.embedding_service = EmbeddingService()
        self.search_engine = SearchEngine(
            elasticsearch_client=self.es_client,
            embedding_service=self.embedding_service,
            index_name=index_name
        )

        # Use GPT-4o-mini or similar for reliable JSON output
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.agent_executor = self._setup_agent()

    def _setup_agent(self):
        """Sets up the LangGraph ReAct agent with shop-specific tools and instructions."""
        
        @tool
        def search_inventory(query: str) -> str:
            """
            Search the shop's local database for products. 
            Returns raw product data including name, price, and description.
            """
            raw_results = self.search_engine.search(query=query)
            items = raw_results.get("marketplace_inventory", [])
            
            if not items:
                return "No products found. Try a broader search term (e.g., brand only)."
            
            return json.dumps(items)

        # System message defined for A2A Schema.org compliance
        # Updated system_message for shop_agent.py inspired by MCP benchmark
        system_message = (
            f"You are a professional e-commerce consultant for {self.shop_id}.\n\n"
            "SEARCH STRATEGY:\n"
            "- Extract exact product identifiers (models, specs) from the query.\n"
            "- Use 'search_inventory' to fetch results.\n"
            "- If no results match, relax constraints (e.g., remove color/size) and retry ONCE.\n"
            "- If STILL no matches, return an empty list: {\"offers\": []}.\n\n"
            "VALUATION RULES:\n"
            "- If user asks for 'cheapest', filter and return ONLY the lowest-priced item.\n"
            "- Do NOT suggest alternatives unless specifically asked.\n\n"
            "RESPONSE HYGIENE:\n"
            "- Return valid JSON ONLY. No markdown, no filler, no backticks."
        )

        return create_react_agent(
            model=self.llm,
            tools=[search_inventory],
            prompt=system_message 
        )

    async def process_message(self, wish: str) -> Tuple[List[Dict], Dict]:
        """Processes the wish and tracks token usage."""
        inputs = {"messages": [("user", wish)]}

        with get_openai_callback() as cb:
            try:
                # Increased recursion_limit from 15 to 25 to avoid premature failure 
                result = await self.agent_executor.ainvoke(
                    inputs, 
                    config={"recursion_limit": 25} 
                )
                agent_output = result["messages"][-1].content
                
                # ... (JSON parsing logic)
                return offers, usage
            except Exception as e:
                # Log the specific error for debugging 
                logger.error(f"Agent execution error: {e}")
                return [], {"prompt_tokens": cb.prompt_tokens, "completion_tokens": cb.completion_tokens, "error": str(e)}

app = FastAPI()
shop_instance: ShopAgentInstance = None
@app.post("/messages")
async def handle_a2a_request(request: Request):
    """Standard A2A JSON-RPC 2.0 interface."""
    try:
        body = await request.json()
        method = body.get("method")
        
        # 1. Direct response to ping to avoid LLM recursion 
        if method == "ping":
            return {"jsonrpc": "2.0", "result": "pong", "id": body.get("id")}
            
        wish = body.get("params", {}).get("query", "")
        
        # 2. Invoke internal LangGraph processing for real queries
        offers, usage = await shop_instance.process_message(wish)
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "agent_name": shop_instance.shop_id,
                "offers": offers,
                "tokens": usage
            },
            "id": body.get("id")
        }
    except Exception as e:
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(e)}, "id": None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2A Shop Agent Server")
    parser.add_argument("--shop_id", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    
    args = parser.parse_args()
    shop_instance = ShopAgentInstance(
        shop_id=args.shop_id,
        index_name=f"{args.shop_id}_nlweb"
    )

    print(f"🚀 {args.shop_id} ready for A2A requests on port {args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)