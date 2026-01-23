import sys
import os
import json
import logging
import argparse
import uvicorn
import re
from typing import List, Dict, Any, Tuple, Set
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

# Load environment variables from the project root .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(src_path), ".env"))

# Local Package Imports
from nlweb_mcp.search_engine import SearchEngine
from nlweb_mcp.elasticsearch_client import ElasticsearchClient
from nlweb_mcp.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_urls_from_response(response_text: str) -> Set[str]:
    """
    Directly embedded URL extraction logic from benchmark_nlweb_mcp.py.
    Extracts URLs from the agent's final response by parsing JSON or using regex fallback.
    """
    if not isinstance(response_text, str):
        return set()
    
    # Try to parse the entire response as JSON first
    try:
        data = json.loads(response_text.strip())
        if isinstance(data, dict) and "urls" in data:
            urls = data["urls"]
            if isinstance(urls, list):
                return set(u for u in urls if isinstance(u, str) and u.strip().lower() != "done")
        elif isinstance(data, list):
            return set(u for u in data if isinstance(u, str) and u.strip().lower() != "done")
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try to find JSON patterns within mixed content
    json_pattern = r'\{"urls":\s*\[.*?\]\}'
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    for match in json_matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "urls" in data and isinstance(data["urls"], list):
                return set(u for u in data["urls"] if isinstance(u, str) and u.strip().lower() != "done")
        except (json.JSONDecodeError, TypeError):
            continue
    
    # Final fallback to regex if no JSON patterns worked
    urls_found = re.findall(r'https?://\S+', response_text)
    return set([url.strip(')>."\',') for url in urls_found])

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

        # Configured with gpt-5-mini
        self.llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        self.agent_executor = self._setup_agent()

    def _setup_agent(self):
        """Sets up the LangGraph ReAct agent with shop-specific tools."""
        
        @tool
        def search_inventory(query: str) -> str:
            """
            Search for products in this shop's inventory index.
            Returns a list of products in JSON format including title, price, and url.
            """
            results = self.search_engine.search(query=query, top_k=5)
            return json.dumps(results)

        # Latest prompt logic from your image
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
            "- Do NOT include any explanatory text, conversational filler, or Markdown code blocks.\n"
            "- If no matches exist, you MUST return: {{\"urls\": []}}"
        )

        return create_react_agent(
            model=self.llm,
            tools=[search_inventory],
            prompt=system_message 
        )

    async def process_message(self, wish: str):
        """Processes the buyer wish and captures REAL token usage."""
        from langchain_community.callbacks import get_openai_callback
        inputs = {"messages": [("user", wish)]}

        with get_openai_callback() as cb:
            try:
                result = await self.agent_executor.ainvoke(
                    inputs, 
                    config={"recursion_limit": 25}
                )
                agent_output = result["messages"][-1].content
                # Use the embedded extraction function directly
                urls = list(extract_urls_from_response(agent_output))
                
                return urls, {
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens
                }
            except Exception as e:
                logger.error(f"Error in process_message: {e}")
                return [], {
                    "error": str(e),
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens
                }

app = FastAPI()
shop_instance: ShopAgentInstance = None

@app.post("/messages")
async def handle_a2a_request(request: Request):
    """Standard A2A JSON-RPC 2.0 interface."""
    try:
        body = await request.json()
        method = body.get("method")
        
        if method == "ping":
            return {"jsonrpc": "2.0", "result": "pong", "id": body.get("id")}
            
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