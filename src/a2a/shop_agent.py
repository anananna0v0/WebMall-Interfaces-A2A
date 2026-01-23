import sys
import os
import json
import logging
import argparse
import uvicorn
import re
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

# Load environment variables from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(src_path), ".env"))

# Local Package Imports
from nlweb_mcp.search_engine import SearchEngine
from nlweb_mcp.elasticsearch_client import ElasticsearchClient
from nlweb_mcp.embedding_service import EmbeddingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_json_payload(text: str) -> List[Dict]:
    """
    Extracts the structured JSON 'offers' array from the agent's output.
    Handles potential raw text or markdown wrapping if LLM deviates.
    """
    try:
        # Try to parse the entire text as JSON first
        data = json.loads(text.strip())
        return data.get("offers", [])
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: Find JSON-like object patterns in the response string
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("offers", [])
    except Exception:
        pass
    
    return []

class ShopAgentInstance:
    def __init__(self, shop_id: str, index_name: str):
        """
        Initializes the Shop Agent with search capabilities and gpt-5-mini.
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

        # Configured with gpt-5-mini as requested
        self.llm = ChatOpenAI(model="gpt-5-mini")
        self.agent_executor = self._setup_agent()

    def _setup_agent(self):
        """Sets up the LangGraph ReAct agent with shop-specific tools."""
        
        @tool
        def search_inventory(query: str) -> str:
            """
            Search for products in this shop's inventory index.
            Returns full product details including price and Schema.org fields.
            """
            results = self.search_engine.search(query=query, top_k=5)
            return json.dumps(results)

        # The requested latest prompt
        system_message = (
            f"You are an advanced e-commerce Shop Agent for {self.shop_id}.\n\n"
            "TASK-SPECIFIC INSTRUCTIONS:\n"
            "- Use 'search_inventory' to find products. This tool returns full product details.\n"
            "- If multiple products match, return them as a list of structured objects.\n\n"
            "RESPONSE FORMAT REQUIREMENTS:\n"
            "- Your final response MUST be a valid JSON object ONLY.\n"
            "- The 'offers' field must contain objects matching Schema.org 'Product' format.\n"
            "- Example Format:\n"
            "{\n"
            "  \"offers\": [\n"
            "    {\n"
            "      \"@context\": \"https://schema.org/\",\n"
            "      \"@type\": \"Product\",\n"
            "      \"name\": \"Product Name\",\n"
            "      \"offers\": { \"price\": 120.0, \"priceCurrency\": \"EUR\" },\n"
            "      \"url\": \"https://...\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "- Do NOT include Markdown code blocks or any conversational text.\n"
            "- If no matches exist, return: {{\"offers\": []}}"
        )

        return create_react_agent(
            model=self.llm,
            tools=[search_inventory],
            prompt=system_message 
        )

    async def process_message(self, wish: str):
        """Processes the buyer wish and captures full token usage."""
        inputs = {"messages": [("user", wish)]}

        with get_openai_callback() as cb:
            try:
                result = await self.agent_executor.ainvoke(
                    inputs, 
                    config={"recursion_limit": 25}
                )
                agent_output = result["messages"][-1].content
                
                # Extract full Schema.org objects from the response
                offers = extract_json_payload(agent_output)
                
                return offers, {
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
    """Standard A2A JSON-RPC 2.0 interface with full schema support."""
    try:
        body = await request.json()
        method = body.get("method")
        
        # 1. Health check
        if method == "ping":
            return {"jsonrpc": "2.0", "result": "pong", "id": body.get("id")}
            
        wish = body.get("params", {}).get("query", "")
        
        # 2. Invoke shop agent logic
        offers, usage = await shop_instance.process_message(wish)
        
        # Return full structure per 2026/01/22 protocol
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