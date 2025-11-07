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
CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = (CURRENT_DIR / "../../results").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === import ChatOpenAI  ===
try:
    from langchain_openai import ChatOpenAI
    print("[DEBUG] Imported ChatOpenAI from langchain_openai ")
    LLM_AVAILABLE = True
except ImportError as e:
    print("[DEBUG] Failed to import ChatOpenAI ", e)
    ChatOpenAI = None
    LLM_AVAILABLE = False

# Elasticsearch setup
from elasticsearch import Elasticsearch
es = Elasticsearch(os.getenv("ES_HOST", "http://localhost:9200"))
INDEX_NAME = os.getenv("ES_INDEX", "webmall_1")

def process_query(state: dict) -> dict:
    """
    Manual hybrid search (semantic + keyword) for ES 8.11.
    Uses Reciprocal Rank Fusion (RRF) to combine results and logs detailed ranking info.
    """
    query = state.get("query", "")
    artifacts = []

    try:
        # === 1. Generate embedding ===
        from openai import OpenAI
        client = OpenAI()
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        # === 2. Semantic search ===
        semantic_body = {
            "size": 5,
            "knn": {
                "field": "composite_embedding",
                "query_vector": embedding,
                "k": 5,
                "num_candidates": 100
            }
        }
        semantic_res = es.search(index=INDEX_NAME, body=semantic_body)
        semantic_hits = semantic_res["hits"]["hits"]

        # === 3. Keyword search ===
        keyword_body = {
            "size": 5,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "description"],
                    "operator": "or"
                }
            }
        }
        keyword_res = es.search(index=INDEX_NAME, body=keyword_body)
        keyword_hits = keyword_res["hits"]["hits"]

        # === 4. Build ranked lists ===
        semantic_list = semantic_hits
        keyword_list = keyword_hits

        # === 5. Reciprocal Rank Fusion (score = 1 / (k + rank)) ===
        k_rrf = int(os.getenv("RRF_K", "60"))

        def rrf_scores(hits):
            scores = {}
            for rank, hit in enumerate(hits, start=1):
                scores[id(hit)] = 1.0 / (k_rrf + rank)
            return scores

        rrf_sem = rrf_scores(semantic_list)
        rrf_key = rrf_scores(keyword_list)

        # === 6. Merge both lists (deduplicate by URL) ===
        seen_by_url = {}
        combined = []
        for src_list in (semantic_list, keyword_list):
            for hit in src_list:
                url = hit["_source"].get("url", "")
                if url not in seen_by_url:
                    seen_by_url[url] = hit
                    combined.append(hit)

        # === 7. Compute total RRF score ===
        def total_rrf(hit):
            hid = id(hit)
            return rrf_sem.get(hid, 0.0) + rrf_key.get(hid, 0.0)

        combined.sort(key=total_rrf, reverse=True)

        # === 8. Keep only top-scoring fusion group ===
        if combined:
            top_score = total_rrf(combined[0])
            eps = 1e-9
            top_hits = [h for h in combined if abs(total_rrf(h) - top_score) < eps]
        else:
            top_hits = []

        # === 9. Format final output ===
        artifacts = []
        for hit in top_hits:
            src = hit["_source"]
            artifacts.append({
                "name": src.get("title", "Unknown product"),
                "price": src.get("price", "N/A"),
                "url": src.get("url", "N/A"),
                "score": total_rrf(hit)
            })

        response_text = f"Found {len(artifacts)} top results for '{query}' (RRF)."

        # === 10. Debug logging (semantic / keyword / final) ===
        buyer_id = os.getenv("ES_INDEX", "unknown_index")
        debug_path = RESULTS_DIR / f"buyer_debug_scores_{buyer_id}.jsonl"

        def pack(hits):
            out = []
            for rank, h in enumerate(hits, start=1):
                s = h["_source"]
                out.append({
                    "rank": rank,
                    "title": s.get("title"),
                    "url": s.get("url"),
                    "bm25_or_knn_score": h.get("_score", 0)
                })
            return out

        with open(debug_path, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "rrf_k": k_rrf,
                "semantic_ranking": pack(semantic_list),
                "keyword_ranking": pack(keyword_list),
                "final_top": [
                    {"title": a["name"], "url": a["url"], "rrf_score": a["score"]}
                    for a in artifacts
                ]
            }) + "\n")

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

    buyer_id = os.getenv("ES_INDEX", "unknown_index")
    log_path = RESULTS_DIR / f"buyer_reasoning_{buyer_id}.jsonl"

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
    uvicorn.run(app, host="0.0.0.0", port=10001)
