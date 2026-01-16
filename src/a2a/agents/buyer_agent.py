import json
import logging
from typing import Dict, List, Any
from openai import OpenAI
from a2a.config import OPENAI_API_KEY
from a2a.protocol import A2AProtocol

logger = logging.getLogger("buyer_agent")

class BuyerAgent:
    def __init__(self, registry):
        self.registry = registry
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.target_model = "gpt-5-mini"

    def execute_procurement_task(self, instruction: str) -> Dict[str, Any]:
        """
        Executes the task in two steps: 
        1. Retrieval: Aggregates raw results from ShopAgents.
        2. Reasoning: Filters results using LLM.
        Tracks internal token usage for both steps.
        """
        raw_candidates = []
        internal_in_chars = 0
        
        # Step 1: Retrieval (Communication with ShopAgents)
        for shop_id, agent in self.registry.get_all_agents().items():
            # Requesting top 3 results from each shop
            req = A2AProtocol.create_request(
                "search_product", 
                {"query": instruction, "max_results": 3}
            )
            # Log inter-agent request size
            internal_in_chars += len(json.dumps(req))
            
            resp = agent.handle_rpc_request(req)
            
            if resp and resp.get("result"):
                raw_candidates.extend(resp["result"])
                # Log inter-agent response size
                internal_in_chars += len(json.dumps(resp["result"]))

        if not raw_candidates:
            return {
                "results": [], 
                "internal_usage": {"in": internal_in_chars // 4, "out": 0}
            }

        # Step 2: Reasoning (Filtering via LLM)
        # Input to LLM includes instruction and the aggregated raw data
        llm_prompt = self._build_prompt(instruction, raw_candidates)
        internal_in_chars += len(llm_prompt)
        
        filtered_results = self._filter_with_llm(llm_prompt)
        
        # Output from LLM
        internal_out_chars = len(json.dumps(filtered_results))
        
        return {
            "results": filtered_results,
            "internal_usage": {
                "in": internal_in_chars // 4,
                "out": internal_out_chars // 4
            }
        }

    def _build_prompt(self, instruction: str, candidates: List[Dict[str, Any]]) -> str:
        return f"""
        User Task: {instruction}
        Candidate Products: {json.dumps(candidates)}
        Task: Return ONLY products that perfectly match the User Task as a JSON list.
        Format: {{"results": [{{"name": "...", "url": "..."}}]}}
        """

    def _filter_with_llm(self, prompt: str) -> List[Dict[str, Any]]:
        try:
            response = self.client.chat.completions.create(
                model=self.target_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("results", [])
        except Exception as e:
            logger.error(f"LLM filtering failed: {e}")
            return []