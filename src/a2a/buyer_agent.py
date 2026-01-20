import json
import logging
from typing import Dict, List, Any, Optional
import tiktoken
from openai import OpenAI

from a2a.config import OPENAI_API_KEY, EMBEDDING_MODEL
from a2a.protocol import A2AProtocol

logger = logging.getLogger("buyer_agent")

class BuyerAgent:
    """
    BuyerAgent: The decision-making center that coordinates multi-step 
    procurement tasks across decentralized ShopAgents.
    """

    def __init__(self, registry):
        self.registry = registry
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.target_model = "gpt-5-mini"
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o") # Use gpt-4o encoder for gpt-5-mini
        
        # Performance tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _count_tokens(self, text: str) -> int:
        """Counts actual tokens using tiktoken."""
        return len(self.tokenizer.encode(text))

    def _generate_embedding(self, text: str) -> List[float]:
        """Calculates embedding once to be shared with all shops."""
        response = self.client.embeddings.create(
            input=[text],
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    def execute_procurement_task(self, instruction: str) -> Dict[str, Any]:
        """
        Main entry point for benchmark tasks. 
        Detects task type and executes the appropriate Action Chain.
        """
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        
        # 1. Vector Pre-processing (Buyer computes once)
        query_embedding = self._generate_embedding(instruction)
        
        # 2. Skill Discovery
        # Filter shops that declare 'product_search' capability
        target_shops = []
        for shop_id, agent in self.registry.get_all_agents().items():
            card = agent.get_agent_card()
            if "product_search" in card.get("skills", []):
                target_shops.append((shop_id, agent))

        # 3. Phase 1: Search and Selection
        all_candidates = self._broadcast_search(instruction, query_embedding, target_shops)
        
        # 4. Phase 2: Decision Making (Selection & Action Planning)
        # Use LLM to pick the best items and decide if checkout is needed
        decision = self._reason_and_plan(instruction, all_candidates)
        
        final_results = []
        
        # 5. Phase 3: Execution (Action Chain for Order tasks)
        if "order" in instruction.lower() or "checkout" in instruction.lower() or "buy" in instruction.lower():
            # If the task requires buying, execute Add to Cart and Checkout
            for item in decision.get("selected_items", []):
                shop_id = item.get("shop_id")
                target_agent = self.registry.get_all_agents().get(shop_id)
                
                if target_agent:
                    # Step: Add to Cart
                    self._send_rpc(target_agent, "add_to_cart", {"product_url": item["url"]})
                    # Step: Checkout
                    checkout_resp = self._send_rpc(target_agent, "checkout", {
                        "payment_info": "Encrypted_Card_Data",
                        "shipping_address": "User_Profile_Address"
                    })
                    if checkout_resp.get("result"):
                        final_results.append(item["url"]) # Return URL as success marker for CR
        else:
            # For search-only tasks, return all selected matches
            final_results = [item["url"] for item in decision.get("selected_items", [])]

        return {
            "results": list(set(final_results)),
            "metrics": {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens
            }
        }

    def _broadcast_search(self, query: str, embedding: List[float], shops: list) -> List[Dict[str, Any]]:
        """Broadcasts search request with pre-computed embedding to selected shops."""
        aggregated_results = []
        for shop_id, agent in shops:
            req = A2AProtocol.create_request("search_product", {
                "query": query,
                "query_embedding": embedding,
                "max_results": 5
            })
            
            resp = agent.handle_rpc_request(req)
            if resp.get("result"):
                # Annotate results with shop_id for later transaction targeting
                for product in resp["result"]:
                    product["shop_id"] = shop_id
                    aggregated_results.append(product)
        
        return aggregated_results

    def _reason_and_plan(self, instruction: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Uses LLM to evaluate all offers and select the best matches."""
        prompt = (
            f"User Task: {instruction}\n"
            f"Aggregated Shop Offers: {json.dumps(candidates)}\n\n"
            "Task: Select the most relevant products that perfectly fulfill the requirement. "
            "If the task asks for 'cheapest', select only one. If it asks for 'all offers', select all relevant. "
            "Return a JSON object: {\"selected_items\": [{\"url\": \"...\", \"shop_id\": \"...\"}]}"
        )
        
        self.total_input_tokens += self._count_tokens(prompt)
        
        try:
            response = self.client.chat.completions.create(
                model=self.target_model,
                messages=[{"role": "system", "content": "You are a professional procurement agent."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            self.total_output_tokens += self._count_tokens(content)
            return json.loads(content)
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {"selected_items": []}

    def _send_rpc(self, agent, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Internal helper to send and track RPC calls."""
        req = A2AProtocol.create_request(method, params)
        resp = agent.handle_rpc_request(req)
        # In A2A, we also track the communication payload size if needed for reporting
        return resp