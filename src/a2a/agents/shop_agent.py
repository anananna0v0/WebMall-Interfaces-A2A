import logging
from typing import List, Dict, Any

# Import components from nlweb_mcp
from nlweb_mcp.elasticsearch_client import ElasticsearchClient
from nlweb_mcp.embedding_service import EmbeddingService

# Import configuration and protocol
from a2a.config import OPENAI_API_KEY, ELASTICSEARCH_HOST, EMBEDDING_MODEL
from a2a.protocol import A2AProtocol

class ShopAgent:
    """
    ShopAgent that manages product search using the specific hybrid_search 
    parameters defined in elasticsearch_client.py.
    """

    def __init__(self, shop_id: str, index_name: str):
        """
        Initialize the agent. ElasticsearchClient.__init__ only accepts 'host'.
        """
        self.shop_id = shop_id
        self.index_name = index_name
        
        # Initialize the embedding service for vector generation
        self.embedding_service = EmbeddingService(
            api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL
        )
        
        # ElasticsearchClient takes host only (elasticsearch_client.py line 25)
        self.es_client = ElasticsearchClient(host=ELASTICSEARCH_HOST)
        self.logger = logging.getLogger(f"shop_agent_{shop_id}")

    def handle_rpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles incoming JSON-RPC 2.0 requests.
        """
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")

        if method == "search_product":
            return self._search_handler(params, request_id)
        return A2AProtocol.create_error_response(-32601, "Method not found", request_id)

    def _search_handler(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        """
        Performs hybrid search using the exact argument names from 
        elasticsearch_client.py line 273.
        """
        query_str = params.get("query", "")
        max_results = params.get("max_results", 5)

        try:
            # 1. Generate the query embedding list using OpenAI API
            vector = self.embedding_service.create_embedding(query_str)

            # 2. Call hybrid_search with the strictly required keyword arguments:
            # index_name, query, query_embedding, and top_k
            # (Ref: elasticsearch_client.py line 273)
            search_results = self.es_client.hybrid_search(
                index_name=self.index_name,
                query=query_str,
                query_embedding=vector,
                top_k=max_results
            )
            
            # 3. Format the results for the BuyerAgent
            formatted_results = []
            for item in search_results:
                formatted_results.append({
                    "name": item.get("title", "Unknown"),
                    "url": item.get("url", ""),
                    "price": float(item.get("price", 0.0))
                })
            
            return A2AProtocol.create_success_response(formatted_results, request_id)
            
        except Exception as e:
            self.logger.error(f"Search failure for {self.shop_id}: {str(e)}")
            return A2AProtocol.create_error_response(-32000, str(e), request_id)