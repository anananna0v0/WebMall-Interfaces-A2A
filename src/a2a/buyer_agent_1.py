# --- 1. Imports (from user's file) ---
import os
import json
import uvicorn
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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
AGENT_NUMBER = "1"
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

# --- Final Agent Card Model (A2A Protocol Compliance) ---
class AgentCard(BaseModel):
    id: str = Field(description="The unique identifier of the agent.")
    name: str = Field(description="Human-readable name of the agent.")
    description: str = Field(description="A short description of the agent's function.")
    webmall_id: str = Field(description="The identifier of the webmall database this agent operates on.")
    skills: List[str] = Field(description="List of core skills, e.g., SEARCH, ADD_TO_CART, CHECKOUT.")
    data_schema_hint: str = Field(description="The functional hint about the unique data structure this agent handles.")

# --- 3. Pydantic Models (Consistent with Coordinator) ---

class A2AMessage(BaseModel):
    """
    The A2A Protocol payload sends a structured task.
    The Coordinator determines the action (e.g., SEARCH) and the target.
    """
    action: str = Field(description="The high-level action to perform (e.g., SEARCH, ADD_TO_CART).")
    target: str = Field(description="The primary target of the action (e.g., user query, product URL).")

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
# !!! AGENT 1 SPECIALIZED PROMPT (V4 - F1 Optimized)                 !!!
# !!! Teaches "match/operator: and" AND "category = brand" rules     !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ES_TRANSLATION_PROMPT = """
You are an expert "Agentic" assistant for 'Shop 1'. Your sole purpose is to translate a user's natural language query into a precise Elasticsearch DSL JSON query object for the 'webmall_1' index.

# 'webmall_1' Index Rules:
# The "category.keyword" field contains the BRAND NAME (e.g., "Asus", "Canon").
# The "title" field requires "match" with "and" operator to be precise.

# Based on this mapping, here are my rules:
1.  You must ONLY respond with the JSON object for the query. Do not add any conversational text or explanations.
2.  If the user asks for "cheapest", "budget", etc., you MUST add a `"sort": [{"price": "asc"}]`.
3.  **CRITICAL RULE:** For "title" searches, you MUST use a `"match"` query with `"operator": "and"`.
4.  **CRITICAL RULE 2:** You MUST infer the BRAND NAME (e.g., "ASUS", "Canon") and use it in a `"filter"` on `"category.keyword"`.
5.  Always limit the results, set `"size": 5`.

# --- EXAMPLES (Based on 'webmall_1' rules, F1-Optimized) ---

# Example 1: User query "Find all offers for the AMD Ryzen 9 5900X."
# (Brand: "AMD", Use "match" + "and")
User: "Find all offers for the AMD Ryzen 9 5900X."
Response:
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": {"query": "AMD Ryzen 9 5900X", "operator": "and"} } }
      ],
      "filter": [
        { "term": { "category.keyword": "AMD" } }
      ]
    }
  },
  "size": 5
}

# Example 2: User query "Find all offers for the Canon EOS R5 Mark II."
# (Brand: "Canon", Use "match" + "and")
User: "Find all offers for the Canon EOS R5 Mark II."
Response:
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": {"query": "Canon EOS R5 Mark II", "operator": "and"} } }
      ],
      "filter": [
        { "term": { "category.keyword": "Canon" } }
      ]
    }
  },
  "size": 5
}

# Example 3: User query "Find the cheapest offer for an ASUS ProArt RTX4070 SUPER OC."
# (Brand: "Asus", Use "match" + "and")
User: "Find the cheapest offer for an ASUS ProArt RTX4070 SUPER OC."
Response:
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": {"query": "ASUS ProArt RTX4070 SUPER OC", "operator": "and"} } }
      ],
      "filter": [
        { "term": { "category.keyword": "Asus" } }
      ]
    }
  },
  "sort": [ { "price": "asc" } ],
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

async def mock_add_to_cart(url: str) -> bool:
    """
    [MOCK ACTION]
    Simulates the action of adding a specific product URL to the cart.
    In a real system, this would be an RPA step (e.g., Selenium click).
    """
    print(f"[{AGENT_ID}] MOCK: Attempting to add product at URL: {url} to cart.")
    # Simulate success or failure based on the URL structure, if desired, 
    # but for simplicity, we mock success.
    
    # We assume the action takes some time
    await asyncio.sleep(0.5) 
    
    log_reasoning({
        "step": "add_to_cart_action",
        "product_url": url,
        "status": "success"
    })
    return True

# --- 7. FastAPI Endpoint (A2A Server) ---

# --- New: Agent Capability Endpoint (Simulated Agent Card) ---
@app.get("/capability", response_model=AgentCard)
async def get_capability():
    # WEBMALL_ID and INDEX_NAME are determined at the top of the file
    
    # Simple check for agent number, assuming AGENT_NUMBER is defined as a string
    return AgentCard(
        id=AGENT_ID,
        name=f"WebMall Buyer Agent {AGENT_NUMBER}",
        description=f"A specialized agent for querying and interacting with WebMall {AGENT_NUMBER}. Handles specific data rules.",
        webmall_id=INDEX_NAME, # Dynamic index name (e.g., webmall_1)
        skills=["SEARCH", "ADD_TO_CART", "CHECKOUT"],
        data_schema_hint=ES_TRANSLATION_PROMPT[:200] # Use the first 200 chars of the custom prompt
    )

# --- THIS IS THE "FAIL-FAST" VERSION FOR EASIER DEBUGGING ---

@app.post("/a2a/sendMessage", response_model=BuyerResponse)
async def handle_a2a_message(request: A2AMessage):
    """
    The main A2A entry point. It routes based on the 'action' field.
    """
    print(f"[{AGENT_ID}] Received action: {request.action} with target: {request.target}")
    
    if not LLM_AVAILABLE or not es:
        # Critical configuration check (Fail fast)
        # ... (logging code remains the same) ...
        return JSONResponse(
            status_code=500,
            content={
                "agent_id": AGENT_ID, "status": "error",
                "content": "Agent is misconfigured: LLM or DB not available."
            }
        )

    try:
        if request.action == "SEARCH":
            # --- SEARCH PATH (Requires LLM and ES) ---
            es_query_dsl = await get_es_query_from_llm(request.target)
            search_results = await execute_es_query(es_query_dsl)
            
            return BuyerResponse(
                agent_id=AGENT_ID,
                status="success",
                content=search_results
            )
        
        elif request.action == "ADD_TO_CART":
            # --- ACTION PATH (Uses MOCK RPA) ---
            success = await mock_add_to_cart(request.target)
            if success:
                 return BuyerResponse(
                    agent_id=AGENT_ID,
                    status="success",
                    content=f"Product added to cart successfully. URL: {request.target}"
                )
            
        else:
            # UNKNOWN ACTION
             return BuyerResponse(
                agent_id=AGENT_ID,
                status="error",
                content=f"Unknown action: {request.action}. Only SEARCH and ADD_TO_CART are supported."
            )
            
    except Exception as e:
        # REAL FAILURE PATH (LLM or ES related)
        # ... (logging code remains the same) ...
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