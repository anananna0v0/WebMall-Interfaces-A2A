# --- 1. Imports ---
import os
import json
import httpx  # For async HTTP requests
import asyncio  # For concurrent delegation
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any

# --- Imports from user's file ---
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
try:
    # Using LangChain as per the provided file
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.output_parsers import JsonOutputParser
    print("[DEBUG] Imported ChatOpenAI from langchain_openai")
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"[DEBUG] Failed to import ChatOpenAI, {e}")
    ChatOpenAI = None
    LLM_AVAILABLE = False

# --- 2. Configuration (from user's file) ---
load_dotenv()

app = FastAPI()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
log_path = RESULTS_DIR / "coordinator_reasoning_log.jsonl"

# --- Buyer URLs ---
BUYER_URLS = [
    "http://localhost:10001/a2a/sendMessage",
    "http://localhost:10002/a2a/sendMessage",
    "http://localhost:10003/a2a/sendMessage",
    "http://localhost:10004/a2a/sendMessage",
]

# --- LLM Initialization ---
if LLM_AVAILABLE:
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0, model_kwargs={"response_format": {"type": "json_object"}}) # <-- 已修正
    synthesizer_llm = ChatOpenAI(model="gpt-5-mini", temperature=0.2)
else:
    llm = None
    synthesizer_llm = None

# --- 3. Pydantic Models (for A2A Protocol and API) ---

class A2AMessage(BaseModel):
    """
    The "A2A Protocol" payload we send to Buyers.
    We send the raw query to let the Buyer Agent use its own "agentic behaviour".
    """
    user_query: str

class BuyerResponse(BaseModel):
    """
    The standardized response we expect back from each Buyer Agent.
    """
    agent_id: str  # e.g., "buyer_agent_10001"
    status: str    # "success" or "error"
    content: Any   # Can be a list of products, a confirmation, or an error message

class UserRequest(BaseModel):
    query: str

class FinalResponse(BaseModel):
    natural_language_response: str
    raw_data: List[BuyerResponse]

# --- 4. Logging Utility ---

def log_reasoning(log_data: Dict[str, Any]):
    """
    Appends a log entry to the coordinator_reasoning_log.jsonl file.
    """
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                **log_data
            }
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"[ERROR] Failed to write to log file: {e}")

# --- 5. Core Logic: LLM Helpers ---

async def get_task_intent(user_query: str) -> Dict[str, Any]:
    # [LLM DECISION 1: Classify Intent]
    # Uses gpt-5-mini to classify the user's intent.
    if not llm:
        return {"intent": "UNKNOWN", "error": "LLM not available"}

    print(f"[Coordinator] Classifying intent for: {user_query}")
    
    # Define the parser
    parser = JsonOutputParser()
    
    # Define the prompt
    prompt = [
        SystemMessage(content=(
            "You are an expert system. Your job is to classify the user's intent for a web mall. "
            "Possible intents are: 'SEARCH' (finding products), "
            "'ADD_TO_CART' (adding a specific item), "
            "'CHECKOUT' (completing the purchase), "
            "or 'UNKNOWN' (for anything else). "
            "Respond ONLY with a JSON object like {\"intent\": \"YOUR_CLASSIFICATION\"}."
        )),
        HumanMessage(content=user_query)
    ]
    
    try:
        token_usage = 0
        # --- New: Add token tracking ---
        with get_openai_callback() as cb:
            # Create the chain: prompt | model | parser
            chain = llm | parser
            result_json = await chain.ainvoke(prompt)
            token_usage = cb.total_tokens # Get token count
            print(f"[Coordinator] Intent LLM Token Usage: {token_usage}")
        
        log_reasoning({
            "step": "intent_classification",
            "query": user_query,
            "result": result_json,
            "token_usage": token_usage # Add to log
        })
        return result_json
    except Exception as e:
        print(f"[Coordinator] LLM Intent classification failed: {e}")
        return {"intent": "UNKNOWN", "error": str(e)}
    
async def synthesize_results(buyer_responses: List[BuyerResponse], user_query: str) -> str:
    # [LLM DECISION 2: Synthesize Results]
    # Uses gpt-5-mini to summarize the findings from all buyers.
    if not synthesizer_llm:
        return "LLM not available. Found some results."

    print(f"[Coordinator] Synthesizing results from all Buyers...")
    
    successful_responses = [r.content for r in buyer_responses if r.status == "success" and r.content]
    
    if not successful_responses:
        return "I'm sorry, I wasn't able to find that item or fulfill the request in any of the stores."
    
    # Create a prompt for the synthesizer LLM
    prompt_context = (
        f"You are a helpful shopping assistant. You asked 4 independent shops (agents) to handle a user request. "
        f"The original user request was: \"{user_query}\"\n\n"
        f"Here are the successful JSON-formatted results from the shops:\n\n"
        f"{successful_responses}\n\n"
        "Your task is to analyze these (potentially different) results and provide a single, comprehensive, and helpful natural language answer to the user. "
        "If the query was to 'find the cheapest', make sure you compare prices and explicitly state which one is cheapest and from which shop. "
        "Summarize the findings clearly. Be friendly and concise. DO NOT make up information or offer to search other stores."
    )

    try:
        token_usage = 0
        # --- New: Add token tracking ---
        with get_openai_callback() as cb:
            response = await synthesizer_llm.ainvoke([
                SystemMessage(content=prompt_context)
            ])
            token_usage = cb.total_tokens # Get token count
            print(f"[Coordinator] Synthesis LLM Token Usage: {token_usage}")

        final_answer = response.content
        
        log_reasoning({
            "step": "synthesis",
            "query": user_query,
            "buyer_results": successful_responses,
            "synthesized_answer": final_answer,
            "token_usage": token_usage # Add to log
        })
        return final_answer
    except Exception as e:
        print(f"[Coordinator] LLM Synthesis failed: {e}")
        return "I found some results, but I had trouble summarizing them for you."

# --- 6. Core Logic: A2A Client ---

async def delegate_to_buyers(payload: A2AMessage) -> List[BuyerResponse]:
    """
    Concurrently sends the A2A task (raw user query) to all Buyer Agents.
    """
    print(f"[Coordinator] Delegating task to {len(BUYER_URLS)} Buyers...")
    
    async with httpx.AsyncClient() as client:
        tasks = []
        for url in BUYER_URLS:
            # We are sending the request to the /a2a/sendMessage endpoint
            tasks.append(
                client.post(url, json=payload.model_dump(), timeout=30.0)
            )

        # Gather results, allowing for exceptions
        results = await asyncio.gather(*tasks, return_exceptions=True)

        parsed_responses = []
        for i, res in enumerate(results):
            agent_url = BUYER_URLS[i]
            # Extract agent ID from URL for logging
            try:
                agent_id = f"buyer_agent_{agent_url.split(':')[2].split('/')[0]}"
            except Exception:
                agent_id = f"buyer_agent_{i+1}"

            if isinstance(res, httpx.Response):
                try:
                    # We expect the Buyer to return a valid BuyerResponse
                    parsed_responses.append(BuyerResponse(**res.json()))
                except Exception:
                    # Buyer returned invalid JSON or format
                    parsed_responses.append(BuyerResponse(
                        agent_id=agent_id, 
                        status="error", 
                        content=f"Agent responded with invalid format: {res.text[:150]}"
                    ))
            else:
                # Request failed (e.g., httpx.ConnectError, Timeout)
                parsed_responses.append(BuyerResponse(
                    agent_id=agent_id, 
                    status="error", 
                    content=f"Agent did not respond: {str(res)}"
                ))
        
        print(f"[Coordinator] Received all Buyer responses: {parsed_responses}")
        return parsed_responses

# --- 7. FastAPI Endpoint ---

@app.post("/process_query", response_model=FinalResponse)
async def process_user_query(request: UserRequest):
    """
    Main user-facing endpoint.
    Orchestrates the 1. Intent -> 2. Delegate -> 3. Synthesize process.
    """
    
    if not LLM_AVAILABLE:
        return JSONResponse(status_code=500, content={"error": "LLM is not available. Check configuration."})

    # Step 1: Use LLM to classify intent
    intent_data = await get_task_intent(request.query)
    intent = intent_data.get("intent", "UNKNOWN")
    
    print(f"[Coordinator] Intent classified as: {intent}")

    if intent == "SEARCH":
        # Step 2: Create the A2A task payload
        # We send the RAW query, letting the Buyer Agents do the "agentic" work
        task_payload = A2AMessage(user_query=request.query)

        # Step 3: Concurrently delegate to all Buyers
        buyer_responses = await delegate_to_buyers(task_payload)
        
        # Step 4: Use LLM to synthesize results into a final answer
        final_answer = await synthesize_results(buyer_responses, request.query)
        
        return FinalResponse(
            natural_language_response=final_answer,
            raw_data=buyer_responses
        )
    
    elif intent == "ADD_TO_CART" or intent == "CHECKOUT":
        # TODO: Implement logic for these intents.
        # This is more complex as it might require context (which item? which store?)
        # or it might need to be sent to a specific Buyer Agent.
        
        # For a 1-day demo, focusing on SEARCH is the priority.
        # We can implement a simple version that sends it to all agents
        # and lets the synthesis LLM figure it out.
        
        print(f"[Coordinator] '{intent}' intent is not fully implemented. Forwarding to all agents as a general task.")
        
        task_payload = A2AMessage(user_query=request.query)
        buyer_responses = await delegate_to_buyers(task_payload)
        final_answer = await synthesize_results(buyer_responses, request.query)

        return FinalResponse(
            natural_language_response=final_answer,
            raw_data=buyer_responses
        )
        
    else:
        # UNKNOWN intent
        return FinalResponse(
            natural_language_response="I'm sorry, I'm not sure how to help with that. Can you rephrase your request?",
            raw_data=[]
        )

# --- 8. Run Server (from user's file) ---

if __name__ == "__main__":
    # This import is here to match the user's provided file
    import uvicorn
    print(f"[Coordinator] Starting Coordinator Agent on http://0.0.0.0:11000")
    uvicorn.run(app, host="0.0.0.0", port=11000)