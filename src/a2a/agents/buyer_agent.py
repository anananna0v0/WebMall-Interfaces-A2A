import json
import logging
from typing import Dict, List, Any
from openai import OpenAI
from a2a.config import OPENAI_API_KEY
from a2a.protocol import A2AProtocol

logger = logging.getLogger("buyer_agent")

class BuyerAgent:
    """
    BuyerAgent that aggregates and filters products from ShopAgents using LLM reasoning.
    """

    def __init__(self, registry):
        self.registry = registry
        # Use gpt-5-mini as the reasoning engine
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.target_model = "gpt-5-mini"

    def execute_procurement_task(self, instruction: str) -> Dict[str, Any]:
        """
        Step 1: Aggregate raw results from all ShopAgents.
        Step 2: Filter results using LLM to match the instruction precisely.
        """
        raw_candidates = []
        
        # Step 1: Broadcast to all ShopAgents (Registry provides all agents)
        for shop_id, agent in self.registry.get_all_agents().items():
            # Based on standard settings, each shop returns top 3 results
            req = A2AProtocol.create_request(
                "search_product", 
                {"query": instruction, "max_results": 3}
            )
            resp = agent.handle_rpc_request(req)
            
            if resp and resp.get("result"):
                raw_candidates.extend(resp["result"])

        if not raw_candidates:
            return {"results": []}

        # Step 2: LLM Reasoning Stage (Filtering out noise)
        filtered_results = self._filter_with_llm(instruction, raw_candidates)
        
        return {"results": filtered_results}

    def _filter_with_llm(self, instruction: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prompt gpt-5-mini to select only the items that perfectly match the instruction.
        """
        prompt = f"""
        User Task: {instruction}
        
        Available Candidate Products:
        {json.dumps(candidates, indent=2)}
        
        Evaluation Rules:
        1. Select ONLY products that accurately match the user's requirement.
        2. If multiple versions or prices exist, keep only the most relevant ones.
        3. Remove irrelevant noise (e.g., cables or cases if the user wants a CPU).
        4. Return the results as a JSON array of objects, each containing 'name' and 'url'.
        
        Output format:
        {{"results": [{{"name": "Product Name", "url": "Exact URL"}}]}}
        """

        try:
            # Using JSON mode for structured output
            response = self.client.chat.completions.create(
                model=self.target_model,
                messages=[{"role": "system", "content": "You are a precise procurement assistant."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            output_json = json.loads(response.choices[0].message.content)
            return output_json.get("results", [])
            
        except Exception as e:
            logger.error(f"LLM Filtering failed: {e}")
            # Fallback to raw candidates if LLM fails
            return raw_candidates