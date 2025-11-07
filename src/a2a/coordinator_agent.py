from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import os
import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
log_path = RESULTS_DIR / "coordinator_reasoning_log.jsonl"

# === import ChatOpenAI  ===
try:
    from langchain_openai import ChatOpenAI
    print("[DEBUG] Imported ChatOpenAI from langchain_openai ")
    LLM_AVAILABLE = True
except ImportError as e:
    print("[DEBUG] Failed to import ChatOpenAI ", e)
    ChatOpenAI = None
    LLM_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

BUYER_URLS = [
    "http://localhost:10001/a2a/sendMessage",
    "http://localhost:10002/a2a/sendMessage",
    "http://localhost:10003/a2a/sendMessage",
    "http://localhost:10004/a2a/sendMessage"
]

def decide_task_with_llm(user_input: str):
    default_task = "search"
    default_query = user_input
    default_reasoning = "(LLM not available; defaulting to search)"
    if not LLM_AVAILABLE:
        return default_task, default_query, default_reasoning

    prompt = f"""
You are a coordinator agent reasoning module.

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
        parsed = json.loads(llm_response.content.strip())
        reasoning = parsed.get("reasoning", default_reasoning)
        task = parsed.get("task", default_task).lower()
        refined_query = parsed.get("query", default_query)
        return task, refined_query, reasoning
    except Exception:
        return default_task, default_query, default_reasoning

@app.get("/.well-known/agent-card")
async def agent_card():
    return {
        "name": "Coordinator Agent",
        "description": "Coordinates multiple Buyer agents, aggregates results, and selects the best offer.",
        "capabilities": [
            {"name": "assign_tasks", "description": "Send user queries to Buyer agents"},
            {"name": "aggregate_results", "description": "Aggregate and compare Buyer responses"}
        ],
        "skills": [
            {"name": "LLM_reasoning", "description": "Use LLM to analyze user requests"},
            {"name": "multi_agent_coordination", "description": "Coordinate multiple Buyer agents"}
        ],
        "version": "0.2"
    }

@app.post("/a2a/sendMessage")
async def send_message(request: Request):
    data = await request.json()
    text = data.get("input", {}).get("text", "").strip()
    task, refined_query, reasoning = decide_task_with_llm(text)

    # === write reasoning log ===
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "input": text,
            "reasoning": reasoning,
            "task": task,
            "refined_query": refined_query
        }) + "\n")

    results = []
    for url in BUYER_URLS:
        try:
            response = requests.post(url, json={"input": {"text": refined_query}}, timeout=15)
            buyer_data = response.json().get("output", {})
            artifacts = buyer_data.get("artifacts", [])
            results.extend(artifacts)
        except Exception as e:
            print(f"[Coordinator] Failed to contact {url}: {e}")

    # === Decide output type based on user input ===
    user_lower = text.lower()

    def is_cheapest_query(q: str) -> bool:
        """Return True if the query asks for the cheapest or lowest price."""
        keywords = ["cheapest", "lowest", "best offer", "lowest price"]
        return any(k in q for k in keywords)

    def is_all_offers_query(q: str) -> bool:
        """Return True if the query asks for all offers or a full list."""
        keywords = ["all offers", "list", "show all"]
        return any(k in q for k in keywords)

    # === Branch selection ===
    if is_all_offers_query(user_lower):
        # Return all valid offers, sorted by price
        valid_results = []
        for item in results:
            try:
                price = float(item.get("price", 0))
                if price > 0:
                    valid_results.append(item)
            except Exception:
                continue

        if valid_results:
            results_sorted = sorted(valid_results, key=lambda x: float(x["price"]))
            response_text = f"Found {len(results_sorted)} offers for '{refined_query}'."
            return JSONResponse({"output": {"text": response_text, "artifacts": results_sorted}})
        else:
            return JSONResponse({"output": {"text": "No results found from Buyers.", "artifacts": []}})

    elif is_cheapest_query(user_lower):
        # Find the cheapest offer(s)
        valid_results = []
        for item in results:
            try:
                price = float(item.get("price", 0))
                if price > 0:
                    valid_results.append(item)
            except Exception:
                continue

        if not valid_results:
            return JSONResponse({"output": {"text": "No valid offers found.", "artifacts": []}})

        min_price = min(float(item["price"]) for item in valid_results)
        cheapest_items = [item for item in valid_results if float(item["price"]) == min_price]

        if len(cheapest_items) == 1:
            response_text = f"Best offer: {cheapest_items[0]['name']} ({cheapest_items[0]['price']} EUR)"
        else:
            response_text = f"Found {len(cheapest_items)} offers sharing the lowest price ({min_price} EUR)."

        return JSONResponse({"output": {"text": response_text, "artifacts": cheapest_items}})

    else:
        # Default: return all results
        if results:
            response_text = f"Found {len(results)} offers for '{refined_query}'."
            return JSONResponse({"output": {"text": response_text, "artifacts": results}})
        else:
            return JSONResponse({"output": {"text": "No results found from Buyers.", "artifacts": []}})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11000)
