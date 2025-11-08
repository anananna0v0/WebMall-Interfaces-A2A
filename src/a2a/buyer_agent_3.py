# --- 1. Imports (from user's file) ---
import os
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any

load_dotenv()

# --- Import LangChain/OpenAI components ---
from langchain_community.callbacks import get_openai_callback
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.output_parsers import JsonOutputParser
    print("[DEBUG] Buyer: Imported ChatOpenAI from langchain_openai")
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] Buyer: Failed to import ChatOpenAI, {e}")
    ChatOpenAI = None
    LLM_AVAILABLE = False

# --- Import Elasticsearch (from user's file) ---
from elasticsearch import Elasticsearch


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! --- CHANGE THIS ONE LINE FOR EACH AGENT --- !!!
AGENT_NUMBER = "3"
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# --- 2. Configuration (based on user's file and AGENT_NUMBER) ---

# --- Agent-specific settings ---
AGENT_ID = f"buyer_agent_1000{AGENT_NUMBER}"
PORT = 10000 + int(AGENT_NUMBER)

# --- Path setup (from user's file) ---
CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = (CURRENT_DIR / "../../results").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- New: Reasoning Log Path (per-agent) ---
log_path = RESULTS_DIR / f"buyer_{AGENT_NUMBER}_reasoning.jsonl"

# --- Elasticsearch setup (from user's file, now parameterized) ---
es = Elasticsearch(os.getenv("ES_HOST", "http://localhost:9200"))
# Uses 'ES_INDEX_1', 'ES_INDEX_2' etc. from .env, or defaults to 'webmall_1', 'webmall_2'
INDEX_NAME = os.getenv(f"ES_INDEX_{AGENT_NUMBER}", f"webmall_{AGENT_NUMBER}")

app = FastAPI(
    title=f"{AGENT_ID} @ {INDEX_NAME}",
    description="Receives tasks, uses LLM to query Elasticsearch, returns results."
)

# --- LLM Initialization ---
if LLM_AVAILABLE:
    # This LLM is for making decisions (translating query)
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0, model_kwargs={"response_format": {"type": "json_object"}}) 
else:
    llm = None

# --- 3. Pydantic Models (Consistent with Coordinator) ---

class A2AMessage(BaseModel):
    user_query: str

class BuyerResponse(BaseModel):
    agent_id: str
    status: str
    content: Any

# --- 4. Logging Utility (New) ---

def log_reasoning(log_data: Dict[str, Any]):
    """
    Appends a log entry to this agent's reasoning log file.
    """
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                **log_data
            }
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"[{AGENT_ID}] [ERROR] Failed to write to log file: {e}")

# --- 5. Core Logic: LLM Decision Maker (Gold-Tier Prompt) ---

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!! AGENT 3 SPECIALIZED PROMPT (V8) - F1 SCORE OPTIMIZED           !!!
# !!! Teaches "match/operator: and" to kill noise / false positives  !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ES_TRANSLATION_PROMPT = """
You are an expert "Agentic" assistant for 'Shop 3'. Your sole purpose is to translate a user's natural language query into a precise Elasticsearch DSL JSON query object for the 'webmall_3' index.

# 'webmall_3' Index Rules:
# The "category.keyword" field is INCONSISTENT.
# - AMD CPUs are "Processors".
# - Canon Cameras can be "Visual Art" OR "Digital Cameras".
# The "title" field requires a "match" query with the "and" operator to be precise.

# Based on this mapping, here are my rules:
1.  You must ONLY respond with the JSON object for the query. Do not add any conversational text or explanations.
2.  If the user asks for "cheapest", "budget", etc., you MUST add a `"sort": [{"price": "asc"}]`.
3.  **CRITICAL RULE:** For "title" searches, you MUST use a `"match"` query with `"operator": "and"`. This finds all keywords but avoids noise. DO NOT use "match_phrase".
4.  You MUST infer the correct PRODUCT TYPE(S) and use them in a `"filter"`.
5.  If a product can have MULTIPLE categories, you must use a "bool/should" query in the filter.
6.  Always limit the results, set `"size": 5`.

# --- EXAMPLES (Based on 'webmall_3' rules, F1-Optimized) ---

# Example 1: User query "Find all offers for the AMD Ryzen 9 5900X."
# (Product Type: "Processors", Use "match" + "and")
User: "Find all offers for the AMD Ryzen 9 5900X."
Response:
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": {"query": "AMD Ryzen 9 5900X", "operator": "and"} } }
      ],
      "filter": [
        { "term": { "category.keyword": "Processors" } }
      ]
    }
  },
  "size": 5
}

# Example 2: User query "Find all offers for the Canon EOS R5 Mark II."
# (Product Types: "Visual Art" OR "Digital Cameras", Use "match" + "and")
User: "Find all offers for the Canon EOS R5 Mark II."
Response:
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": {"query": "Canon EOS R5 Mark II", "operator": "and"} } }
      ],
      "filter": [
        {
          "bool": {
            "should": [
              { "term": { "category.keyword": "Visual Art" } },
              { "term": { "category.keyword": "Digital Cameras" } }
            ],
            "minimum_should_match": 1
          }
        }
      ]
    }
  },
  "size": 5
}
"""

async def get_es_query_from_llm(user_query: str) -> Dict[str, Any]:
    # [AGENTIC BEHAVIOUR]
    # The core of the agent. It decides *how* to fulfill the task.
    if not llm:
        raise ValueError("LLM is not available.")

    print(f"[{AGENT_ID}] LLM is translating query: {user_query}")
    
    parser = JsonOutputParser()
    prompt = [
        SystemMessage(content=ES_TRANSLATION_PROMPT), # This correctly reads the specialized prompt (V4, V10, V8, V9)
        HumanMessage(content=user_query)
    ]
    
    try:
        token_usage = 0
        # --- New: Add token tracking ---
        with get_openai_callback() as cb:
            chain = llm | parser
            query_dsl = await chain.ainvoke(prompt)
            token_usage = cb.total_tokens # Get token count
            print(f"[{AGENT_ID}] LLM Token Usage: {token_usage}")
        
        print(f"[{AGENT_ID}] LLM produced ES DSL: {query_dsl}")
        
        # --- Log the decision ---
        log_reasoning({
            "step": "llm_translation",
            "user_query": user_query,
            "generated_dsl": query_dsl,
            "token_usage": token_usage # Add to log
        })
        return query_dsl
    
    except Exception as e:
        print(f"[{AGENT_ID}] LLM translation failed: {e}")
        log_reasoning({
            "step": "llm_translation",
            "status": "error",
            "user_query": user_query,
            "error": str(e)
        })
        raise

# --- 6. Core Logic: Action (Database) ---

async def execute_es_query(query_dsl: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    [REAL FUNCTION]
    This function executes the query against the REAL database.
    It will raise an exception if it fails.
    """
    print(f"[{AGENT_ID}] EXECUTING real ES query on index '{INDEX_NAME}'.")
    
    # The 'try/except' is now in the main handler.
    # We let errors propagate up.
    response = es.search(
        index=INDEX_NAME,
        body=query_dsl
    )
    hits = response['hits']['hits']
    
    results = [{"title": hit['_source'].get('title'), 
                "price": hit['_source'].get('price'), 
                "url": hit['_source'].get('url'), 
                "store": AGENT_ID} 
               for hit in hits]
    
    log_reasoning({
        "step": "execute_es_query",
        "status": "success",
        "found_hits": len(results)
    })
    return results

# --- 7. FastAPI Endpoint (A2A Server) ---
# --- THIS IS THE "FAIL-FAST" VERSION FOR EASIER DEBUGGING ---

@app.post("/a2a/sendMessage", response_model=BuyerResponse)
async def handle_a2a_message(request: A2AMessage):
    """
    This is the main A2A entry point.
    It will try the "real path". If it fails, it will FAIL LOUDLY
    by returning a 500 error, which is better for debugging.
    """
    print(f"[{AGENT_ID}] Received task: {request.user_query}")
    
    if not LLM_AVAILABLE or not es:
        # This is a critical configuration error. Fail hard.
        log_reasoning({
            "step": "pre-check", "status": "error",
            "error": "LLM or ES client is not available."
        })
        return JSONResponse(
            status_code=500,
            content={
                "agent_id": AGENT_ID, "status": "error",
                "content": "Agent is misconfigured: LLM or DB not available."
            }
        )

    try:
        # --- HAPPY PATH ---
        # Step 1: Decide (Use LLM to get ES query)
        es_query_dsl = await get_es_query_from_llm(request.user_query)
        
        # Step 2: Act (Execute the REAL ES query)
        search_results = await execute_es_query(es_query_dsl)
        
        if not search_results:
            return BuyerResponse(
                agent_id=AGENT_ID,
                status="success",
                content=[] # Real query, but no results found
            )

        # Step 3: Respond (Send real, structured data back)
        return BuyerResponse(
            agent_id=AGENT_ID,
            status="success",
            content=search_results
        )
        
    except Exception as e:
        # --- REAL FAILURE PATH ---
        # Something failed (LLM translation, ES query, etc.)
        # Log it and return a 500 error so the developer knows.
        print(f"[{AGENT_ID}] [ERROR] Real pipeline failed: {e}")
        log_reasoning({
            "step": "pipeline_failure",
            "status": "error",
            "error": str(e)
        })
        
        # Return a 500 error to the Coordinator (or curl)
        return JSONResponse(
            status_code=500,
            content={
                "agent_id": AGENT_ID, "status": "error",
                "content": f"An internal error occurred: {str(e)}"
            }
        )
    
# --- 8. Run Server ---

if __name__ == "__main__":
    print(f"[Buyer Agent] Starting {AGENT_ID} on http://0.0.0.0:{PORT} (Index: {INDEX_NAME})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)