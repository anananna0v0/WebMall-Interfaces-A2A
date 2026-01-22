import sys
import os
import json
import logging
import argparse
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- Path Configuration ---
# Current file: src/a2a/shop_agent.py (Go up 3 levels to reach project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.append(os.path.join(BASE_DIR, "src"))

load_dotenv(os.path.join(BASE_DIR, ".env"))

# Import required local modules
from nlweb_mcp.config import WEBMALL_SHOPS, ELASTICSEARCH_HOST
from nlweb_mcp.elasticsearch_client import ElasticsearchClient
from nlweb_mcp.embedding_service import EmbeddingService
from nlweb_mcp.search_engine import SearchEngine
from nlweb_mcp.woocommerce_client import WooCommerceClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Core Shop Logic ---
class ShopAgentInstance:
    def __init__(self, shop_id: str):
        self.shop_id = shop_id
        self.config = WEBMALL_SHOPS.get(shop_id)
        # Load identity from agent cards
        shop_num = shop_id.split('_')[-1]
        self.card_path = os.path.join(BASE_DIR, "src", "a2a", "cards", f"shop_{shop_num}_card.json")
        self.agent_card = self._load_agent_card()
        
        # Initialize search components
        self.es_client = ElasticsearchClient(host=ELASTICSEARCH_HOST)
        self.embedding_service = EmbeddingService()
        self.search_engine = SearchEngine(self.es_client, self.embedding_service, self.config["index_name"])
        self.woo_client = WooCommerceClient(base_url=self.config["url"])
        self.llm_client = AsyncOpenAI()

    def _load_agent_card(self):
        try:
            with open(self.card_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"name": self.shop_id}

    async def process_search(self, raw_wish: str):
        """Analyze wish, optimize query, and return JSON-LD products."""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        
        reasoning_prompt = f"""
            You are the highly experienced Head of Sales at {self.agent_card.get('name')}. 
            A buyer agent sent a 'Wish': "{raw_wish}"

            Your goal is to transform this wish into the most effective search query for our Elasticsearch semantic engine.

            Your reasoning steps:
            1. **Analyze Intent**: Determine if the buyer is looking for a specific model, comparing prices, or seeking alternatives (substitutes).
            2. **Extract Entities**: Identify and preserve crucial technical identifiers such as brand names, model numbers (e.g., 'S24+', 'RTX4070'), and physical dimensions (e.g., '360mm', '1TB').
            3. **Handle Specificity**: If the wish is vague, translate it into keywords that characterize our entry-level product lines.
            4. **Keyword Optimization**: Return only core semantic keywords for maximum precision.

            OUTPUT REQUIREMENT: 
            - Return ONLY the optimized search string.
            - Do NOT include any explanations or formatting.
            """
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": reasoning_prompt}]
            )
            refined_query = response.choices[0].message.content.strip()
            usage = {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens}
            logger.info(f"[{self.shop_id}] Optimized Query: {refined_query}")
        except Exception:
            refined_query = raw_wish

        res = self.search_engine.search(query=refined_query)
        search_results = res if isinstance(res, dict) else json.loads(res)
        raw_items = search_results.get("marketplace_inventory", [])
        products = [self.woo_client.get_schema_org_product(item) for item in raw_items]
        
        logger.info(f"[{self.shop_id}] Found {len(products)} products.")
        return products, usage

# --- FastAPI Implementation ---
app = FastAPI()
shop_instance = None

@app.post("/messages")
async def handle_a2a(request: Request):
    """JSON-RPC 2.0 Entry point for Buyer Agent."""
    global shop_instance
    if not shop_instance:
        shop_instance = ShopAgentInstance(app.state.args.shop_id)
        
    body = await request.json()
    if body.get("method") == "ask_webmall":
        query = body.get("params", {}).get("query", "")
        products, usage = await shop_instance.process_search(query)
        return {
            "jsonrpc": "2.0",
            "result": {"agent_name": shop_instance.agent_card.get("name"), "offers": products, "usage": usage},
            "id": body.get("id")
        }
    return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shop_id", type=str, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    app.state.args = args 
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)