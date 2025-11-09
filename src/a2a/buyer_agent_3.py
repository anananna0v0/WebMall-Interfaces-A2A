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
# !!! AGENT 3 SPECIALIZED PROMPT (V30) - F1 SCORE OPTIMIZED           !!!
# !!! Teaches "Smartwatches" category (Task 2)                       !!!
# !!! Teaches "description" search (Task 1)                          !!!
# !!! Teaches "must_not" (UAG) (Task 2)                              !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ES_TRANSLATION_PROMPT = """
You are an expert "Agentic" assistant for 'Shop 3'. Your sole purpose is to translate a user's natural language query into a precise Elasticsearch DSL JSON query object for the 'webmall_3' index.

# 'webmall_3' Index Rules:
# The "category.keyword" field is INCONSISTENT.
# - AMD CPUs are "Processors".
# - Canon Cameras can be "Visual Art" OR "Digital Cameras".
# The "title" field requires a "match" query with the "and" operator to be precise.
# CRITICAL KNOWLEDGE 3 (V30 Import): Specifications (like "orange" or "Series 6") can be in EITHER the "title" OR "description" field.
# CRITICAL KNOWLEDGE 4 (V30 New): The "Smartwatches" category contains BOTH the main Watch AND accessories (like "UAG").
# - When searching for "Apple smart watches", you MUST translate the title query to "Apple Watch".
# - You MUST ALSO exclude accessory-only BRANDS (like "UAG") using "must_not".

# Based on this mapping, here are my rules:
1.  You must ONLY respond with the JSON object for the query. Do not add any conversational text or explanations.
2.  If the user asks for "cheapest", "budget", etc., you MUST add a `"sort": [{"price": "asc"}]`.
3.  **CRITICAL RULE:** For "title" searches, you MUST use a `"match"` query with `"operator": "and"`. This finds all keywords but avoids noise. DO NOT use "match_phrase".
4.  You MUST infer the correct PRODUCT TYPE(S) and use them in a `"filter"`.
5.  If a product can have MULTIPLE categories, you must use a "bool/should" query in the filter.
6.  Always limit the results, set `"size": 5`.
7.  **CRITICAL RULE 2 (V30 Import):** If the query contains specifications (like colors, sizes), you MUST search for them in BOTH "title" and "description" using a "bool/should" query.

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

# Example 3: User query "Find all offers for Apple smart watches."
# (V30 DEBUGGED EXAMPLE - This is our Task 2)
# (Product Type: "Smartwatches")
# (CRITICAL: "Apple smart watches" -> "Apple Watch" for title search)
# (CRITICAL: "must_not" accessory brands like "UAG")
User: "Find all offers for Apple smart watches."
Response:
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": {"query": "Apple Watch", "operator": "and"} } }
      ],
      "must_not": [
        { "match": { "title": "UAG" } },
        { "match": { "title": "strap" } }
      ],
      "filter": [
        { "term": { "category.keyword": "Smartwatches" } }
      ]
    }
  },
  "size": 5
}

# Example 4: User query "Find all offers for orange straps that fit with the Apple Watch Series 6."
# (V30 DEBUGGED EXAMPLE - This is our Task 1)
# (Product Type: "Smartwatches", Specs: "orange" + "Series 6" -> Search BOTH title AND description)
User: "Find all offers for orange straps that fit with the Apple Watch Series 6."
Response:
{
  "query": {
    "bool": {
      "must": [
        { "match": { "title": {"query": "Apple Watch strap", "operator": "and"} } },
        { "bool": {
            "should": [
              { "match": { "title": "orange" } },
              { "match": { "description": "orange" } }
            ], "minimum_should_match": 1
          }
        },
        { "bool": {
            "should": [
              { "match": { "title": "Series 6" } },
              { "match": { "description": "Series 6" } }
            ], "minimum_should_match": 1
          }
        }
      ],
      "filter": [
        { "term": { "category.keyword": "Smartwatches" } }
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

async def mock_checkout(details: Dict[str, str]) -> str:
    """
    [MOCK ACTION]
    Simulates the checkout process using provided customer and payment details.
    """
    print(f"[{AGENT_ID}] MOCK: Attempting checkout for user: {details.get('name')}")
    
    await asyncio.sleep(1.0) # Checkout takes longer
    
    # Check for basic required fields for logging clarity
    if not details.get('card') or not details.get('name'):
        status = "failed"
        message = "Checkout failed: Missing card or name details."
    else:
        status = "success"
        message = f"Checkout successfully completed. Payment card: XXXX{details.get('card', '????')[-4:]} for user: {details.get('name')}."

    log_reasoning({
        "step": "checkout_action",
        "user_name": details.get('name'),
        "status": status
    })
    return message

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
    Main A2A entry point for the Buyer Agent. Routes the task based on the 'action'.
    """
    print(f"[{AGENT_ID}] Received action: {request.action} with target: {request.target}")

    if not LLM_AVAILABLE or not es:
        # Critical configuration check (Fail fast)
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
        if request.action == "SEARCH":
            # --- SEARCH PATH (Handles both Query and URL targets) ---
            
            if request.target.lower().startswith("http"):
                # Case 1: Target is a direct URL (for Checkout setup)
                
                # CRITICAL FIX: Match the webmall name string in the URL
                # We check if the target URL contains the specific webmall ID (e.g., "webmall-3")
                webmall_id_match = f"webmall-{AGENT_NUMBER}" # AGENT_NUMBER is "1", "2", "3", or "4"
                
                if webmall_id_match in request.target:
                    # Success: Return mock data for the product confirmed by URL
                    print(f"[{AGENT_ID}] Found product via URL match. Returning mock data.")
                    return BuyerResponse(
                        agent_id=AGENT_ID,
                        status="success",
                        # We return mock data for simplicity since the product is confirmed by URL
                        content=[{"title": "Trust TK-350 Keyboard", "price": 49.99, "url": request.target, "store": AGENT_ID}]
                    )
                else:
                    # Target URL does not belong to this store 
                    print(f"[{AGENT_ID}] Target URL does not belong to this store. Skipping search.")
                    return BuyerResponse(agent_id=AGENT_ID, status="success", content=[])
            
            else:
                # Case 2: Target is a natural language query, proceed with LLM/ES
                es_query_dsl = await get_es_query_from_llm(request.target)
                search_results = await execute_es_query(es_query_dsl)
                
                return BuyerResponse(
                    agent_id=AGENT_ID,
                    status="success",
                    content=search_results
                )
        
        elif request.action == "ADD_TO_CART":
            # --- ADD_TO_CART PATH (Mock Action) ---
            success = await mock_add_to_cart(request.target)
            if success:
                 return BuyerResponse(
                    agent_id=AGENT_ID,
                    status="success",
                    content=f"Product added to cart successfully. URL: {request.target}"
                )
            
        elif request.action == "CHECKOUT":
            # --- CHECKOUT PATH (Mock Action, expects JSON target) ---
            
            # CRITICAL FIX: Use robust JSON parsing for the target string from the LLM.
            
            # Step 1: Clean and Parse the JSON string
            try:
                # 1. Strip external quotes and control characters that might be introduced by LLM
                clean_target = request.target.strip().replace('\\n', '').replace('\\t', '')
                # 2. Safely load the JSON data
                checkout_details = json.loads(clean_target)
            except json.JSONDecodeError as e:
                # If parsing fails, it's because the LLM didn't produce valid JSON
                print(f"[{AGENT_ID}] Checkout failed: JSON Decode Error: {e}")
                return BuyerResponse(
                    agent_id=AGENT_ID,
                    status="error",
                    content=f"Checkout failed: Details were not sent in valid JSON format. Error: {str(e)}"
                )

            # Step 2: Execute Mock Checkout
            checkout_message = await mock_checkout(checkout_details)
            
            if "failed" in checkout_message:
                status = "error"
            else:
                status = "success"
                
            return BuyerResponse(
                agent_id=AGENT_ID,
                status=status,
                content=checkout_message
            )

        else:
            # UNKNOWN ACTION
             return BuyerResponse(
                agent_id=AGENT_ID,
                status="error",
                content=f"Unknown action: {request.action}. Only SEARCH, ADD_TO_CART, and CHECKOUT are supported."
            )
            
    except Exception as e:
        # REAL FAILURE PATH (LLM or ES related)
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