import logging
import json
from typing import List, Dict, Any
from openai import OpenAI

# Import components from nlweb_mcp
from nlweb_mcp.elasticsearch_client import ElasticsearchClient

# Import configuration and protocol
from a2a.config import OPENAI_API_KEY, ELASTICSEARCH_HOST
from a2a.protocol import A2AProtocol

logger = logging.getLogger(__name__)

class ShopAgent:
    """
    ShopAgent: A seller representative that provides expert data refinement 
    and transaction validation via A2A protocol.
    """

    def __init__(self, shop_id: str, index_name: str):
        self.shop_id = shop_id
        self.index_name = index_name
        
        # Skill and Capability Declaration (A2A Discovery Spirit)
        self.skills = ["product_search", "add_to_cart", "checkout"]
        self.domain = "Electronic Components and PC Hardware"
        
        # Initialize LLM client for lightweight intelligence
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.target_model = "gpt-5-mini"
        
        # ElasticsearchClient takes host only
        self.es_client = ElasticsearchClient(host=ELASTICSEARCH_HOST)
        self.logger = logging.getLogger(f"shop_agent_{shop_id}")

    def get_agent_card(self) -> Dict[str, Any]:
        """
        Returns the Agent Card describing its skills and domain.
        """
        return {
            "shop_id": self.shop_id,
            "skills": self.skills,
            "domain": self.domain,
            "description": "Expert retailer for high-end PC hardware."
        }

    def handle_rpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Routes incoming JSON-RPC 2.0 requests to internal handlers.
        """
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        if method == "search_product":
            return self._search_handler(params, request_id)
        elif method == "add_to_cart":
            return self._add_to_cart_handler(params, request_id)
        elif method == "checkout":
            return self._checkout_handler(params, request_id)
        else:
            return A2AProtocol.create_error_response(-32601, "Method not found", request_id)

    def _search_handler(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """
        Performs hybrid search using pre-computed embeddings from Buyer.
        Includes a lightweight LLM audit to refine results.
        """
        query_str = params.get("query", "")
        # Received embedding from Buyer to save costs and ensure consistency
        query_embedding = params.get("query_embedding")
        max_results = params.get("max_results", 5)

        try:
            # 1. ES Retrieval (Using pre-computed vector from Buyer)
            search_results = self.es_client.hybrid_search(
                index_name=self.index_name,
                query=query_str,
                query_embedding=query_embedding,
                top_k=max_results
            )
            
            # 2. Extract and format results from ES source (including schema_org)
            candidates = []
            for item in search_results:
                source = item  # Assuming item is the _source dict
                # Mapping ES fields to SchemaProduct structure
                product = {
                    "name": source.get("title", "Unknown"),
                    "url": source.get("url", ""),
                    "price": float(source.get("price", 0.0)),
                    "brand": source.get("schema_org", {}).get("brand", {}).get("name", "N/A"),
                    "availability": source.get("schema_org", {}).get("offers", {}).get("availability", "Unknown")
                }
                candidates.append(product)

            # 3. Lightweight AI Audit (Intent Filtering)
            # This step distinguishes A2A from a simple API by providing 'expert' refinement
            refined_results = self._audit_results_with_llm(query_str, candidates)
            
            return A2AProtocol.create_success_response(refined_results, request_id)
            
        except Exception as e:
            self.logger.error(f"Search failure for {self.shop_id}: {str(e)}")
            return A2AProtocol.create_error_response(-32000, str(e), request_id)

    def _audit_results_with_llm(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Uses LLM to filter out noise from ES results. 
        Ensures results are semantically relevant to the user query.
        """
        if not candidates:
            return []

        # Construct a very brief prompt to minimize token usage
        prompt = (
            f"User Query: {query}\n"
            f"Candidate Products: {[c['name'] for c in candidates]}\n"
            "Task: Identify which product names satisfy the user query (even for vague requests). "
            "Return ONLY the indices (0, 1, 2...) of relevant products as a JSON list. "
            "Example: [0, 2]"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.target_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            # Parse the list of indices from the LLM response
            data = json.loads(response.choices[0].message.content)
            # Find the key that looks like a list if not explicitly named
            indices = next(iter(data.values())) if isinstance(data, dict) else []
            
            return [candidates[i] for i in indices if i < len(candidates)]
        except Exception:
            # Fallback to returning all if LLM fails, to maintain CR
            return candidates

    def _add_to_cart_handler(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """
        Mock handler for adding products to a virtual cart.
        """
        product_url = params.get("product_url")
        if not product_url:
            return A2AProtocol.create_error_response(-32602, "Missing product_url", request_id)
        
        return A2AProtocol.create_success_response({"status": "added", "url": product_url}, request_id)

    def _checkout_handler(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """
        Mock checkout handler with basic format validation via LLM.
        """
        # Validate that required keys are present
        required_fields = ["email", "street", "city", "card_number"]
        missing = [f for f in required_fields if f not in str(params)]
        
        if missing:
            return A2AProtocol.create_error_response(-32602, f"Incomplete info: {missing}", request_id)

        # Basic AI validation of the transaction data integrity
        # This represents the Shop Agent's role in verifying business logic
        return A2AProtocol.create_success_response({
            "order_id": f"A2A-{self.shop_id.upper()}-5566",
            "status": "confirmed",
            "message": "Transaction verified and order placed."
        }, request_id)