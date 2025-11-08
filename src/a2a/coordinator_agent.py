# --- 1. Imports ---
import os
import json
import httpx  # For async HTTP requests
import asyncio  # For concurrent delegation
import contextlib
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

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

# Global variable: stores discovered Agent Cards
BUYER_CAPABILITIES: Dict[str, 'AgentCard'] = {} # Forward reference 'AgentCard' needed here

# --- LLM Initialization ---
if LLM_AVAILABLE:
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    synthesizer_llm = ChatOpenAI(model="gpt-5-mini", temperature=0.2)
else:
    llm = None
    synthesizer_llm = None

# --- 3. Pydantic Models (for A2A Protocol and API) ---

# --- Agent Card Model (A2A Protocol Compliance) ---
class AgentCard(BaseModel):
    id: str = Field(description="The unique identifier of the agent.")
    name: str = Field(description="Human-readable name of the agent.")
    description: str = Field(description="A short description of the agent's function.")
    webmall_id: str = Field(description="The identifier of the webmall database this agent operates on.")
    skills: List[str] = Field(description="List of core skills, e.g., SEARCH, ADD_TO_CART, CHECKOUT.")
    data_schema_hint: str = Field(description="The functional hint about the unique data structure this agent handles.")

class A2AMessage(BaseModel):
    """
    The A2A Protocol payload sends a structured task.
    The Coordinator determines the action (e.g., SEARCH) and the target.
    """
    action: str = Field(description="The high-level action to perform (e.g., SEARCH, ADD_TO_CART).")
    target: str = Field(description="The primary target of the action (e.g., user query, product URL).")

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

# ---  Models for LLM Plan Generation ---
class PlanStep(BaseModel):
    step_id: int
    action: str = Field(description="The high-level action to perform (e.g., SEARCH, ADD_TO_CART, CHECKOUT).")
    query: str = Field(description="The specific natural language query or target URL for this step.")
    
class ActionPlan(BaseModel):
    # The output model for the LLM when intent is complex
    intent: str = Field(description="The overall intent (e.g., 'PLAN').")
    plan: List[PlanStep] = Field(description="The sequence of actions needed to fulfill the user's request.")

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
    # [LLM DECISION 1: Classify Intent and Generate Plan]
    if not llm:
        return {"intent": "UNKNOWN", "error": "LLM not available"}

    print(f"[Coordinator] Classifying intent and generating plan for: {user_query}")
    
    # We define the JSON structure we want the LLM to output (ActionPlan)
    parser = JsonOutputParser(pydantic_object=ActionPlan)
    
    # The core instruction for generating the plan
    system_message_content = (
        "You are an expert planning system for a multi-agent web mall. Your task is to analyze the user's request "
        "and generate a structured sequence of actions (a 'plan') to fulfill it. "
        "The primary action types are: 'SEARCH', 'ADD_TO_CART', and 'CHECKOUT'.\n"
        
        # --- CRITICAL CONSOLIDATION RULE ADDED ---
        "CRITICAL CONSOLIDATION: If the request includes both 'Add to Cart' AND 'Checkout', you MUST generate a plan "
        "with ONLY TWO steps: 1) SEARCH, and 2) CHECKOUT. DO NOT include an intermediate ADD_TO_CART step in the final plan.\n"
        # --- END CRITICAL CONSOLIDATION RULE ---

        "If the final step is CHECKOUT, you MUST include ALL user and payment details (address, card number, CVV, expiry) "
        "in the final CHECKOUT step's 'query' field, formatted as a single JSON string. "
        "This requires precise data extraction from the user's input.\n"
        
        "Your final output MUST be a single JSON object conforming to the following schema:\n"
        f"{parser.get_format_instructions()}\n\n"
        "Example of a multi-step plan for CHECKOUT (Find X -> Checkout):"
        
        '{"intent": "PLAN", "plan": ['
        '{"step_id": 1, "action": "SEARCH", "query": "Find the cheapest offer for the Asrock B550 PHANTOM GAMING 4."},'
        '{"step_id": 2, "action": "CHECKOUT", "query": "{\\"name\\": \\"Jessica Morgan\\", \\"email\\": \\"jessica.morgan@yahoo.com\\", \\"street\\": \\"Maple Avenue\\", \\"house_number\\": \\"742\\", \\"zip\\": \\"60614\\", \\"city\\": \\"Chicago\\", \\"state\\": \\"IL\\", \\"country\\": \\"USA\\", \\"card\\": \\"4242424242424242\\", \\"cvv\\": \\"123\\", \\"expiry_date\\": \\"12/28\\"}"}'
        ']}'

        "Example of a multi-step plan for CHECKOUT (Find URL -> Checkout):"
    
        '{"intent": "PLAN", "plan": ['
        '{"step_id": 1, "action": "SEARCH", "query": "https://webmall-3.informatik.uni-mannheim.de/product/trust-tk-350-wireless-membrane-keyboard-spill-proof-silent-keys-media-keys-black"},'
        '{"step_id": 2, "action": "CHECKOUT", "query": "{\"name\":\"Jessica Morgan\"...}"}' 
        ']}'
    )
    
    prompt = [
        SystemMessage(content=system_message_content),
        HumanMessage(content=f"User Request: {user_query}")
    ]
    
    try:
        token_usage = 0
        with get_openai_callback() as cb:
            chain = llm | parser
            result_json_obj = await chain.ainvoke(prompt)
            token_usage = cb.total_tokens
            print(f"[Coordinator] Intent LLM Token Usage: {token_usage}")
        
        # Check if the result is a plan (simple search tasks will also be wrapped in a plan now)
        if result_json_obj.get("intent") == "PLAN" and "plan" in result_json_obj:
             log_reasoning({
                "step": "plan_generation",
                "query": user_query,
                "plan": result_json_obj.get("plan"),
                "token_usage": token_usage
            })
             # We return the whole plan object
             return result_json_obj
        else:
            raise ValueError("LLM did not return a valid structured plan.")
            
    except Exception as e:
        print(f"[Coordinator] LLM Planning failed: {e}")
        # If planning fails, treat as UNKNOWN
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
    
async def discover_buyer_capabilities():
    """
    [A2A DISCOVERY STEP]
    Contacts all known Buyer Agents' /capability endpoint to fetch their Agent Cards.
    This simulates the Discovery layer of the A2A protocol.
    """
    global BUYER_CAPABILITIES
    print("[Coordinator] Starting A2A Discovery: Fetching Agent Cards...")
    
    # We set a timeout for the discovery process
    async with httpx.AsyncClient(timeout=10.0) as client:
        tasks = []
        # NOTE: We need to use the base URL for capability discovery, not the /sendMessage endpoint
        base_urls = [url.split('/a2a/sendMessage')[0] for url in BUYER_URLS] 
        
        for url in base_urls:
            # Calls the /capability endpoint
            capability_url = f"{url}/capability"
            tasks.append(client.get(capability_url))
        
        # Gather responses concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for url_index, response in enumerate(responses):
            buyer_url = base_urls[url_index] # Use the base URL for logging context
            if isinstance(response, httpx.Response) and response.status_code == 200:
                try:
                    # Parse and validate the Agent Card Pydantic model
                    card = AgentCard.model_validate(response.json())
                    BUYER_CAPABILITIES[card.id] = card
                    print(f"[Coordinator] Discovery SUCCESS: {card.name} (Webmall: {card.webmall_id})")
                except Exception as e:
                    print(f"[Coordinator] Discovery ERROR: Could not parse Agent Card from {buyer_url}/capability. Error: {e}")
            else:
                print(f"[Coordinator] Discovery ERROR: Agent at {buyer_url}/capability is unreachable or failed to respond. Response: {response}")
    
    print(f"[Coordinator] A2A Discovery Complete. Found {len(BUYER_CAPABILITIES)} active Agents.")

async def mock_add_to_cart_execution(agent_id: str, url: str) -> BuyerResponse:
    """
    Simulates the process of delegating the ADD_TO_CART action to the specific agent.
    Returns a BuyerResponse object for the Coordinator's synthesis step.
    """
    # NOTE: This function would normally call delegate_to_buyers with a targeted URL.
    
    # We assume success and log the action
    print(f"[Coordinator] MOCK ACTION: Delegating ADD_TO_CART to {agent_id} for URL: {url}")
    await asyncio.sleep(0.1) # Simulate network delay
    
    return BuyerResponse(
        agent_id=agent_id,
        status="success",
        content=f"Product added to cart successfully. URL: {url}"
    )

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
            # NOTE: The agent ID is now better determined via the Agent Card, 
            # but we use a robust method here since we expect 4 responses.
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

# --- New: Lifespan Context Manager (Replaces on_event("startup")) ---
@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup events (like A2A Discovery) and shutdown events.
    """
    print("Coordinator starting...")
    
    # --- A2A Discovery Step ---
    await discover_buyer_capabilities() # <-- Execute Discovery on startup
    
    print("Coordinator Agent started successfully.")
    
    yield # <-- Application runs here

    # --- Shutdown (optional) ---
    print("Coordinator shutting down...")


# --- Apply Lifespan processor to FastAPI app ---
app = FastAPI(
    title="Coordinator Agent (A2A)",
    description="Receives user queries, delegates tasks to Buyer Agents, and synthesizes results.",
    lifespan=lifespan
)

@app.post("/process_query", response_model=FinalResponse)
async def process_user_query(request: UserRequest):
    """
    Main user-facing endpoint. Orchestrates the Multi-step Plan execution.
    """
    
    if not LLM_AVAILABLE:
        return JSONResponse(status_code=500, content={"error": "LLM is not available. Check configuration."})

    # Step 1: Use LLM to generate the action plan (Plan JSON)
    plan_data = await get_task_intent(request.query)
    intent = plan_data.get("intent", "UNKNOWN")
    
    print(f"[Coordinator] Intent classified as: {intent}")

    if intent == "PLAN" and "plan" in plan_data:
        # State tracking: urls_to_action stores {URL: agent_id} mapping for subsequent actions
        all_buyer_responses: List[BuyerResponse] = []
        urls_to_action: Dict[str, str] = {} 
        final_synthesis_query = request.query
        
        # --- EXECUTE PLAN STEPS ---
        for step in plan_data["plan"]:
            step_action = step["action"]
            step_query = step["query"]
            
            print(f"[Coordinator] Executing Step {step['step_id']}: {step_action} on target: {step_query}")
            
            if step_action == "SEARCH":
                # Step 2: Execute SEARCH (Delegate to all buyers)
                task_payload = A2AMessage(action="SEARCH", target=step_query)
                search_responses = await delegate_to_buyers(task_payload)
                all_buyer_responses.extend(search_responses)
                
                # Process SEARCH results to find URLs for next steps (ADD_TO_CART/CHECKOUT)
                for res in search_responses:
                    if res.status == "success" and res.content:
                        # Collect all URLs and their corresponding Agent ID (Store)
                        for item in res.content:
                            urls_to_action[item['url']] = res.agent_id
                            
                final_synthesis_query = step_query

            elif step_action == "ADD_TO_CART":
                # Step 3: Execute ADD_TO_CART (Delegate to specific agents for each URL found)
                
                if not urls_to_action:
                    print("[Coordinator] ADD_TO_CART skipped: SEARCH yielded no results for action.")
                    all_buyer_responses.append(BuyerResponse(agent_id="Coordinator", status="info", content="ADD_TO_CART skipped: No targets from SEARCH step."))
                    continue

                print(f"[Coordinator] Delegating ADD_TO_CART for {len(urls_to_action)} unique offers.")
                
                add_to_cart_tasks = []
                for url, agent_id in urls_to_action.items():
                    add_to_cart_tasks.append(
                         mock_add_to_cart_execution(agent_id, url)
                    )
                
                mock_results = await asyncio.gather(*add_to_cart_tasks)
                all_buyer_responses.extend(mock_results)

            elif step_action == "CHECKOUT":
                # Step 4: Execute CHECKOUT (The final action)
                if not urls_to_action:
                    all_buyer_responses.append(BuyerResponse(agent_id="Coordinator", status="info", content="CHECKOUT skipped: No items selected for purchase."))
                    continue
                
                try:
                    # The LLM must provide the structured details in the step_query
                    checkout_details = json.loads(step_query)
                except json.JSONDecodeError:
                    all_buyer_responses.append(BuyerResponse(agent_id="Coordinator", status="error", content="CHECKOUT failed: LLM did not provide valid JSON details for payment."))
                    continue
                
                # --- ACTION EXECUTION: Targeted Checkout Delegation ---
                
                # Find the target Buyer ID (We assume the first item found is the target for checkout)
                first_url, target_agent_id = next(iter(urls_to_action.items())) 
                
                # --- CRITICAL BUG FIX (Replaces flawed index logic) ---
                # Find the target Buyer's /a2a/sendMessage URL from the BUYER_URLS list
                
                agent_port_str = target_agent_id.split('_')[-1] # e.g., "10003"
                target_base_url = None
                for url in BUYER_URLS:
                    if agent_port_str in url:
                        target_base_url = url
                        break
                # --- END OF BUG FIX ---

                if target_base_url:
                    print(f"[Coordinator] Delegating CHECKOUT to {target_agent_id} at {target_base_url}")
                    
                    # The payload contains the ACTION and the JSON details (target)
                    checkout_payload = A2AMessage(action="CHECKOUT", target=step_query) 
                    
                    try:
                        async with httpx.AsyncClient(timeout=30.0) as client:
                            checkout_response = await client.post(target_base_url, json=checkout_payload.model_dump())
                            checkout_result = BuyerResponse.model_validate(checkout_response.json())
                            all_buyer_responses.append(checkout_result)
                    except Exception as e:
                        all_buyer_responses.append(BuyerResponse(agent_id="Coordinator", status="error", content=f"CHECKOUT routing/execution failed: {str(e)}"))
                else:
                    all_buyer_responses.append(BuyerResponse(agent_id="Coordinator", status="error", content=f"CHECKOUT failed: Could not find target Buyer URL for agent {target_agent_id}"))

                # Clear action list after checkout
                urls_to_action = {}


        # Step 5: Final Synthesis
        # We synthesize results from ALL steps 
        final_answer = await synthesize_results(all_buyer_responses, final_synthesis_query)
        
        return FinalResponse(
            natural_language_response=final_answer,
            raw_data=all_buyer_responses
        )
    
    else:
        # UNKNOWN or failed plan generation
        return FinalResponse(
            natural_language_response=f"I'm sorry, I couldn't generate a plan to fulfill your complex request: {request.query}",
            raw_data=[]
        )

# --- 8. Run Server (from user's file) ---

if __name__ == "__main__":
    # This import is here to match the user's provided file
    import uvicorn
    print(f"[Coordinator] Starting Coordinator Agent on http://0.0.0.0:11000")
    # NOTE: The 'app = FastAPI(...)' call is now above the endpoints,
    # and the lifespan handler is correctly applied.
    uvicorn.run(app, host="0.0.0.0", port=11000)