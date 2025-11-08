from a2a.ingestion.elasticsearch_client_woo import ElasticsearchClient
from a2a.ingestion.embedding_service import EmbeddingService

es = ElasticsearchClient()
embed = EmbeddingService()

query = "wireless keyboard"
indices = ["webmall_1", "webmall_2", "webmall_3", "webmall_4"]

for index in indices:
    print(f"\n=== Fallback Hybrid Search for index '{index}' and query '{query}' ===")
    try:
        results = es.search_hybrid_fallback(index_name=index, query=query, embedding_service=embed, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']} | €{r['price']} | {r['url']}")
    except Exception as e:
        print(f"Error searching {index}: {e}")
