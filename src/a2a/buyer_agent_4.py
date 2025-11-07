"""
This is a modified version of the Buyer agent that uses an LLM to parse
incoming messages and decide what action to take (search, add to cart, checkout).

The code introduces a ``decide_task_with_llm`` function that uses a language model
(via LangChain's ChatOpenAI wrapper) to classify the user's intent and extract
a refined search query. You need to install ``langchain_community`` and
``openai`` in your environment for this to work. The environment should have
``OPENAI_API_KEY`` set.

The ``add_to_cart`` and ``checkout`` branches are simplified; use this as a
starting point for debugging and extending functionality.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# === setup results directory ===
BASE_DIR = Path(__file__).resolve().parent.parent  # 指到 WebMall-Interfaces-A2A/src 以外那層
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Optional: import ChatOpenAI from LangChain if available
try:
    from langchain_community.chat_models import ChatOpenAI
    LLM_AVAILABLE = True
except ImportError:
    ChatOpenAI = None
    LLM_AVAILABLE = False

# Elasticsearch setup
from elasticsearch import Elasticsearch
es = Elasticsearch(os.getenv("ES_HOST", "http://localhost:9200"))
INDEX_NAME = os.getenv("ES_INDEX", "webmall_4")

def process_query(state: dict) -> dict:
    """Simple query processing using Elasticsearch."""
    query = state.get("query", "")
    search_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^2", "description"],
                            "operator": "or",
                        }
                    }
                ]
            }
        },
        "size": 3,
    }
    try:
        results = es.search(index=INDEX_NAME, body=search_body)
        hits = results["hits"]["hits"]
        artifacts = []
        for hit in hits:
            source = hit["_source"]
            artifacts.append(
                {
                    "name": source.get("title", "Unknown product"),
                    "price": source.get("price", "N/A"),
                    "url": source.get("url", "N/A"),
                }
            )
        response_text = f"Found {len(artifacts)} results for query: '{query}'"
    except Exception as e:
        response_text = f"Error searching Elasticsearch: {str(e)}"
        artifacts = []
    return {"response": response_text, "artifacts": artifacts}

def invoke_workflow(state: dict) -> dict:
    return process_query(state)

def decide_task_with_llm(user_input: str):
    """
    Use an LLM to decide the user's intent (search/add_to_cart/checkout)
    and generate a concise search query. If the LLM is unavailable, this
    function defaults to always returning a search task with the original input.
    """
    default_task = "search"
    default_query = user_input
    default_reasoning = "(LLM not available; defaulting to search)"
    if not LLM_AVAILABLE:
        return default_task, default_query, default_reasoning
    prompt = f"""
You are an agent reasoning module for a shopping assistant.

For the given user message:
1. Briefly explain your reasoning (1-2 sentences).
2. Decide the task type (search / add_to_cart / checkout).
3. If applicable, produce a concise query (2–6 words).

Format your output as JSON:
{{
  "reasoning": "<short explanation>",
  "task": "search",
  "query": "wireless mouse"
}}

User message: "{user_input}"
"""
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.2)
    try:
        llm_response = llm.invoke(prompt)
        import json
        parsed = json.loads(llm_response.content.strip())
        reasoning = parsed.get("reasoning", default_reasoning)
        task = parsed.get("task", default_task).lower()
        refined_query = parsed.get("query", default_query)
        return task, refined_query, reasoning
    except Exception:
        return default_task, default_query, default_reasoning

app = FastAPI()

@app.get("/.well-known/agent-card")
async def agent_card():
    return {
        "name": "Buyer Agent",
        "description": "Improved buyer agent using LLM to decide tasks and search queries.",
        "service_endpoint": "/a2a/sendMessage",
        "capabilities": [
            {"name": "search_offers", "description": "Search products by keyword"},
            {"name": "add_to_cart", "description": "Add selected product(s) to cart"},
            {"name": "checkout", "description": "Simulate checkout with provided payment details"},
        ],
        "skills": [
            {"name": "elasticsearch_query", "description": "Query products in the local ES index"},
            {"name": "llm_reasoning", "description": "Use LLM to interpret user messages"},
        ],
        "version": "0.1",
    }

@app.post("/a2a/sendMessage")
async def send_message(request: Request):
    data = await request.json()
    text = data.get("input", {}).get("text", "").strip()
    task, refined_query, reasoning = decide_task_with_llm(text)
    print(f"[Buyer LLM] Reasoning: {reasoning}")

    log_path = RESULTS_DIR / "buyer_reasoning_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "input": text,
            "reasoning": reasoning
        }) + "\n")

    if task == "search":
        result = invoke_workflow({"query": refined_query})
        return JSONResponse({"output": {"text": result["response"], "artifacts": result["artifacts"]}})
    elif task == "add_to_cart":
        # Perform a search first to identify items, then simulate adding to cart
        result = invoke_workflow({"query": refined_query})
        artifacts = result["artifacts"]
        if artifacts:
            added_item = artifacts[0]
            response_text = f"Added '{added_item['name']}' ({added_item['price']} EUR) to cart successfully (LLM-driven)."
            return JSONResponse({"output": {"text": response_text, "artifacts": [added_item]}})
        else:
            return JSONResponse({"output": {"text": "No products found to add to cart (LLM-driven).", "artifacts": []}})
    elif task == "checkout":
        return JSONResponse({"output": {"text": "Checkout completed successfully (LLM-driven).", "artifacts": []}})
    else:
        result = invoke_workflow({"query": refined_query})
        return JSONResponse({"output": {"text": result["response"], "artifacts": result["artifacts"]}})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10004)
