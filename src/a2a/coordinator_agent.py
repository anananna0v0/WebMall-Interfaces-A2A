from fastapi import FastAPI, Request
from openai import OpenAI
import aiohttp, asyncio, json
from dotenv import load_dotenv

# --- Environment setup ---
load_dotenv()
client = OpenAI()

# --- FastAPI app ---
app = FastAPI()

# --- Buyer agent endpoints ---
BUYER_URLS = [
    "http://localhost:10001/a2a/sendMessage",
    "http://localhost:10002/a2a/sendMessage",
    "http://localhost:10003/a2a/sendMessage",
    "http://localhost:10004/a2a/sendMessage",
]


@app.get("/.well-known/agent-card")
async def agent_card():
    return {
        "name": "Coordinator Agent",
        "description": "Coordinates all Buyer agents and aggregates results.",
        "capabilities": [
            {"name": "coordinate_search", "description": "Send user query to all buyers and merge their responses"},
            {"name": "summarize_results", "description": "Use GPT-5-mini to summarize and rank offers"},
        ],
        "version": "0.4",
    }


# --- helper to query a single buyer ---
async def query_buyer(session, url, query):
    try:
        async with session.post(url, json={"input": {"text": query}}, timeout=60) as resp:
            data = await resp.json()
            artifacts = data.get("output", {}).get("artifacts", [])
            if isinstance(artifacts, str):
                try:
                    artifacts = json.loads(artifacts)
                except Exception:
                    artifacts = []
            return {"url": url, "artifacts": artifacts}
    except Exception as e:
        print(f"[Coordinator] Error contacting {url}: {e}")
        return {"url": url, "artifacts": []}


@app.post("/a2a/sendMessage")
async def handle_message(request: Request):
    data = await request.json()
    query = data["input"]["text"]

    # --- Step 1: Query all buyers in parallel ---
    async with aiohttp.ClientSession() as session:
        tasks = [query_buyer(session, url, query) for url in BUYER_URLS]
        buyer_results = await asyncio.gather(*tasks)

    # --- Step 2: Flatten all offers ---
    all_artifacts = []
    for r in buyer_results:
        for a in r["artifacts"]:
            if isinstance(a, dict):
                a["source_buyer"] = r["url"]
                all_artifacts.append(a)

    print(f"[Coordinator] Received {len(all_artifacts)} total offers from buyers.")

    # --- Step 3: Ask LLM to summarize / rank best 3 offers ---
    if not all_artifacts:
        return {"output": {"artifacts": []}}

    prompt = (
        "You are a shopping coordinator combining offers from multiple stores. "
        "Given these product options, select the 3 best matches for the user query. "
        "Return them as JSON list with fields: name, price, url, source_buyer.\n\n"
        f"User query: {query}\n\nOffers:\n{json.dumps(all_artifacts, indent=2)}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a neutral and precise shopping assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        raw_output = response.choices[0].message.content
        try:
            summary = json.loads(raw_output)
        except json.JSONDecodeError:
            print("[Coordinator Warning] LLM returned non-JSON output.")
            summary = []
        usage = getattr(response, "usage", None)
        if usage:
            print(f"[Coordinator] Tokens used: {usage.total_tokens}")
    except Exception as e:
        print(f"[Coordinator Error] {e}")
        summary = []

    return {"output": {"artifacts": summary}}


# Run with: uvicorn src.a2a.coordinator_agent:app --port 11000 --reload
