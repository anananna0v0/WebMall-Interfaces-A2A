from fastapi import FastAPI, Request
from elasticsearch import Elasticsearch
from openai import OpenAI
from dotenv import load_dotenv
import json

# --- Environment setup ---
load_dotenv()
client = OpenAI()

# --- Elasticsearch setup ---
es = Elasticsearch("http://localhost:9200")  # default local ES endpoint
INDEX_NAME = "webmall_4_nlweb"

# --- FastAPI setup (A2A layer) ---
app = FastAPI()


@app.get("/.well-known/agent-card")
async def agent_card():
    return {
        "name": "Buyer Agent 1",
        "shop_domain": "webmall-4.informatik.uni-mannheim.de",
        "description": "Buyer agent for WebMall Shop 1 (handles search, add-to-cart, and checkout tasks).",
        "capabilities": [
            {"name": "search_offers", "description": "Search products by keyword"},
            {"name": "add_to_cart", "description": "Add a selected product to the cart"},
            {"name": "checkout", "description": "Simulate checkout for items in the cart"},
        ],
        "skills": [
            {"name": "elasticsearch_query", "description": "Query products from the offline Elasticsearch index webmall_1_nlweb"},
            {"name": "llm_reasoning", "description": "Use GPT-5-mini to select the best results based on search context"},
            {"name": "fastapi_endpoint", "description": "Expose A2A-compatible endpoints for communication"},
        ],
        "version": "0.4",
    }


@app.post("/a2a/sendMessage")
async def handle_message(request: Request):
    data = await request.json()
    query = data["input"]["text"]

    # --- Step 1: Search Elasticsearch ---
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "description"],
                "operator": "or",
            }
        },
        "size": 5,
    }

    try:
        results = es.search(index=INDEX_NAME, body=search_body)
        hits = results["hits"]["hits"]

        top_products = [
            {
                "name": h["_source"].get("title", "Unknown"),
                "price": h["_source"].get("price", "N/A"),
                "url": h["_source"].get("url", ""),
            }
            for h in hits
        ]
        import sys
        print("[DEBUG] top_products:", json.dumps(top_products, indent=2))
        sys.stdout.flush()

    except Exception as e:
        print(f"[Elasticsearch Error] {e}")
        top_products = []

    # --- Step 2: Reasoning with LLM ---
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a buyer agent helping a customer find products from '{INDEX_NAME}'. "
                        "You are given the top 5 search results from the local Elasticsearch index. "
                        "Pick the 2 most relevant ones to the user query and return them as JSON: "
                        "[{\"name\": ..., \"price\": ..., \"url\": ...}]. "
                        "If no relevant products are found, return []."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User query: {query}\n\nTop search results:\n{json.dumps(top_products, indent=2)}",
                },
            ],
        )

        # Try to parse model output as JSON
        raw_output = response.choices[0].message.content
        try:
            answer = json.loads(raw_output)
        except json.JSONDecodeError:
            print("[Warning] Model did not return valid JSON.")
            print("[RAW OUTPUT]:", raw_output)
            sys.stdout.flush()
            answer = []


        # Optional: print token usage
        usage = getattr(response, "usage", None)
        if usage:
            print(f"[Buyer Agent 1] Tokens used: {usage.total_tokens}")

    except Exception as e:
        print(f"[LLM Error] {e}")
        answer = []

    # --- Step 3: Return final A2A-compatible response ---
    return {"output": {"artifacts": answer}}


# --- Run with: uvicorn src.a2a.buyer_agent_4:app --port 10001 ---
