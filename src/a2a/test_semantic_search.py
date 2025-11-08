from a2a.ingestion.elasticsearch_client_woo import ElasticsearchClient
from a2a.ingestion.embedding_service import EmbeddingService

def test_semantic_search(index_name: str, query_text: str, top_k: int = 5):
    """
    Perform a semantic search on the specified Elasticsearch index.
    """
    # Initialize Elasticsearch and embedding services
    es = ElasticsearchClient()
    embed = EmbeddingService()

    # Create query embedding from user input
    query_embedding = embed.create_embedding(query_text)

    # Execute semantic search using the composite embedding field
    results = es.search_semantic(index_name, query_embedding, top_k=top_k)

    # Display results
    print(f"\n=== Semantic Search Results for index '{index_name}' and query '{query_text}' ===\n")
    for i, r in enumerate(results):
        title = r.get("title", "Unknown Title")
        price = r.get("price", "N/A")
        url = r.get("url", "")
        print(f"{i+1}. {title} | €{price} | {url}")


if __name__ == "__main__":
    # Test queries for all four stores
    query_text = "wireless keyboard"
    indices = ["webmall_1", "webmall_2", "webmall_3", "webmall_4"]

    for index in indices:
        try:
            test_semantic_search(index, query_text)
        except Exception as e:
            print(f"\nError testing {index}: {e}")
