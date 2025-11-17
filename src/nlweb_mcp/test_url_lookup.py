#!/usr/bin/env python3
"""
Simple test script for get_products_by_urls function
Tests URL lookup directly without running MCP servers
"""

import json
import sys
from typing import List
from dotenv import load_dotenv

# Add parent directory to Python path
sys.path.append('.')

from elasticsearch_client import ElasticsearchClient
from embedding_service import EmbeddingService
from search_engine import SearchEngine

# Load environment variables
load_dotenv(dotenv_path="/Users/aaronsteiner/Documents/GitHub/webmall-alternative-interfaces/.env")


def test_url_lookup(urls: List[str], index_name: str = "webmall_3"):
    """Test get_products_by_urls function directly"""
    
    print(f"Testing URL lookup for index: {index_name}")
    print(f"URLs to test: {urls}\n")
    
    try:
        # Initialize components
        print("Initializing Elasticsearch client...")
        es_client = ElasticsearchClient()
        
        print("Initializing embedding service...")
        embedding_service = EmbeddingService()
        
        # Test embedding service
        print("Testing embedding service...")
        if not embedding_service.test_embedding_service():
            print("ERROR: Embedding service test failed")
            return
        
        print("Initializing search engine...")
        search_engine = SearchEngine(
            es_client,
            embedding_service,
            index_name
        )
        
        # Perform health check
        print("Performing health check...")
        health = search_engine.health_check()
        print(f"Health status: {health.get('status')}")
        if health.get('status') != 'healthy':
            print(f"WARNING: Health check issues: {health}")
        
        print("\n" + "="*60)
        print("TESTING get_products_by_urls")
        print("="*60 + "\n")
        
        # Call the function
        results = search_engine.get_products_by_urls(urls)
        
        # Display results
        print(f"Total URLs requested: {len(urls)}")
        print(f"Total results returned: {len(results)}")
        print("\nDetailed Results:")
        print("-" * 60)
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            if "error" in result:
                print(f"  ERROR: {result.get('error')}")
                print(f"  URL: {result.get('url')}")
            else:
                print(f"  Title: {result.get('name', 'N/A')}")
                print(f"  URL: {result.get('url', 'N/A')}")
                print(f"  Price: {result.get('offers', {}).get('price', 'N/A')}")
                print(f"  Category: {result.get('category', 'N/A')}")
                if 'description' in result:
                    desc = result['description']
                    if len(desc) > 200:
                        desc = desc[:200] + "..."
                    print(f"  Description: {desc}")
        
        # Test variations
        print("\n" + "="*60)
        print("TESTING URL VARIATIONS")
        print("="*60 + "\n")
        
        # Test each URL with and without trailing slash
        for url in urls:
            print(f"\nTesting variations for: {url}")
            
            # Test without trailing slash
            url_no_slash = url.rstrip('/')
            print(f"  Without slash: {url_no_slash}")
            results_no_slash = search_engine.get_products_by_urls([url_no_slash])
            print(f"    Found: {len([r for r in results_no_slash if 'error' not in r])} products")
            
            # Test with trailing slash
            url_with_slash = url_no_slash + '/'
            print(f"  With slash: {url_with_slash}")
            results_with_slash = search_engine.get_products_by_urls([url_with_slash])
            print(f"    Found: {len([r for r in results_with_slash if 'error' not in r])} products")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'es_client' in locals() and hasattr(es_client.client, 'close'):
            es_client.client.close()
            print("\nClosed Elasticsearch connection")


if __name__ == "__main__":
    # Default test URLs
    test_urls = [
        "https://webmall-3.informatik.uni-mannheim.de/product/trust-tk-350-wireless-membrane-keyboard-spill-proof-silent-keys-media-keys-black/"
    ]
    
    index_name = "webmall_3_nlweb"
    test_url_lookup(test_urls, index_name)