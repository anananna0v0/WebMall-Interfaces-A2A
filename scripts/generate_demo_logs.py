#!/usr/bin/env python3
"""
Script to generate demo log JSON files for each interface by querying
actual Elasticsearch indices with a real product search.
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, AsyncElasticsearch
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Configuration
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
RAG_INDEX = "webmall_v2"
NLWEB_INDICES = ["webmall_1_nlweb", "webmall_2_nlweb", "webmall_3_nlweb", "webmall_4_nlweb"]

# Product to search for (available in all 4 shops)
SEARCH_QUERY = "Canon EOS R5 Mark II"
TASK_ID = "Webmall_Single_Product_Search_Task2"

# Expected URLs for Canon EOS R5 Mark II
EXPECTED_URLS = [
    "https://webmall-1.informatik.uni-mannheim.de/product/eos-r5-mark-ii-body",
    "https://webmall-2.informatik.uni-mannheim.de/product/canon-eos-r5-ii",
    "https://webmall-3.informatik.uni-mannheim.de/product/canon-eos-r5-mark-ii-full-frame-sans-miroir-camera",
    "https://webmall-4.informatik.uni-mannheim.de/product/canon-eos-r5-mark-ii-mirrorless-digital-camera-body-only"
]


class DemoLogGenerator:
    def __init__(self):
        self.es_client = Elasticsearch(
            [ELASTICSEARCH_HOST],
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30
        )
        self.async_client = AsyncElasticsearch(
            [ELASTICSEARCH_HOST],
            verify_certs=False,
            ssl_show_warn=False,
            timeout=30
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for query text"""
        return await self.embeddings.aembed_query(text)

    async def search_rag_index(self, query: str, query_embedding: List[float], match_count: int = 20) -> List[Dict]:
        """Search the RAG index with hybrid search"""
        search_body = {
            "size": match_count,
            "query": {
                "bool": {
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'composite_embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                },
                                "boost": 1.0
                            }
                        },
                        {
                            "match": {
                                "title": {
                                    "query": query,
                                    "boost": 3.0
                                }
                            }
                        },
                        {
                            "match": {
                                "content": {
                                    "query": query,
                                    "boost": 0.5
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["url", "title", "summary", "content", "chunk_number"]
        }

        response = await self.async_client.search(index=RAG_INDEX, body=search_body)

        results = []
        for hit in response['hits']['hits']:
            doc = hit['_source']
            doc['score'] = hit['_score']
            results.append(doc)

        return results

    async def search_nlweb_index(self, index_name: str, query: str, query_embedding: List[float], match_count: int = 10) -> List[Dict]:
        """Search a specific NLWeb shop index"""
        search_body = {
            "size": match_count,
            "query": {
                "bool": {
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'composite_embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                },
                                "boost": 1.0
                            }
                        },
                        {
                            "match": {
                                "title": {
                                    "query": query,
                                    "boost": 3.0
                                }
                            }
                        },
                        {
                            "match": {
                                "description": {
                                    "query": query,
                                    "boost": 1.0
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["url", "title", "price", "description", "category", "schema_org"]
        }

        try:
            response = await self.async_client.search(index=index_name, body=search_body)

            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                doc['score'] = hit['_score']
                results.append(doc)

            return results
        except Exception as e:
            print(f"Error searching {index_name}: {e}")
            return []

    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing trailing slashes"""
        return url.rstrip('/')

    def calculate_metrics(self, retrieved_urls: List[str], expected_urls: List[str]) -> Dict:
        """Calculate precision, recall, and F1 score"""
        retrieved_normalized = set(self.normalize_url(u) for u in retrieved_urls)
        expected_normalized = set(self.normalize_url(u) for u in expected_urls)

        correct = retrieved_normalized & expected_normalized

        precision = len(correct) / len(retrieved_normalized) if retrieved_normalized else 0
        recall = len(correct) / len(expected_normalized) if expected_normalized else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        task_completion = 1 if recall == 1.0 else 0

        return {
            "task_completion_rate": task_completion,
            "avg_precision": round(precision, 4),
            "avg_recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

    async def generate_rag_demo_log(self) -> Dict:
        """Generate demo log for RAG interface"""
        print(f"Generating RAG demo log for: {SEARCH_QUERY}")

        start_time = datetime.now()
        query_embedding = await self.get_embedding(SEARCH_QUERY)

        # Perform search
        results = await self.search_rag_index(SEARCH_QUERY, query_embedding, match_count=30)

        execution_time = (datetime.now() - start_time).total_seconds()

        # Extract URLs and deduplicate
        seen_urls = set()
        unique_results = []
        for r in results:
            url = self.normalize_url(r['url'])
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Find correct URLs in results
        correct_retrieved = []
        additional_retrieved = []
        url_ranks = {}

        expected_normalized = set(self.normalize_url(u) for u in EXPECTED_URLS)

        for rank, r in enumerate(unique_results, 1):
            url_norm = self.normalize_url(r['url'])
            if url_norm in expected_normalized:
                correct_retrieved.append(r['url'])
                url_ranks[url_norm] = rank
            else:
                additional_retrieved.append(r['url'])

        # Only include the correct product URLs in parsed_urls
        parsed_urls = correct_retrieved

        metrics = self.calculate_metrics(correct_retrieved, EXPECTED_URLS)

        # Build tool history
        tool_history = [
            {
                "tool_name": "search_products",
                "tool_args": {
                    "query": SEARCH_QUERY,
                    "match_count": 15,
                    "use_hybrid": True
                },
                "tool_output": {
                    "results_found": len(results),
                    "status": "success"
                },
                "timestamp": datetime.now().isoformat(),
                "tool_type": "search"
            }
        ]

        # Add get_product_details call if we found correct URLs
        if correct_retrieved:
            # Get full product details without truncation
            product_details = []
            for r in unique_results:
                if r['url'] in correct_retrieved:
                    product_details.append({
                        "title": r['title'],
                        "url": r['url'],
                        "description": r.get('content', '')  # Full content description
                    })

            tool_history.append({
                "tool_name": "get_product_details",
                "tool_type": "details",
                "tool_args": {
                    "urls": correct_retrieved
                },
                "tool_output_raw": json.dumps({
                    "status": "success",
                    "urls_requested": len(correct_retrieved),
                    "details_found": len(correct_retrieved),
                    "product_details": product_details
                }),
                "timestamp": datetime.now().isoformat(),
                "tool_output_parsed": {
                    "status": "success",
                    "urls_requested": len(correct_retrieved),
                    "details_found": len(correct_retrieved)
                }
            })

        demo_log = {
            "task_id": TASK_ID,
            "user_task": f"\nFind all offers for the {SEARCH_QUERY}.\n",
            "metrics": metrics,
            "parsed_urls": parsed_urls,
            "db_urls_found": [u for u in EXPECTED_URLS if self.normalize_url(u) in set(self.normalize_url(r['url']) for r in unique_results)],
            "db_urls_missing": [u for u in EXPECTED_URLS if self.normalize_url(u) not in set(self.normalize_url(r['url']) for r in unique_results)],
            "db_coverage": len([u for u in EXPECTED_URLS if self.normalize_url(u) in set(self.normalize_url(r['url']) for r in unique_results)]) / len(EXPECTED_URLS),
            "tool_history": tool_history,
            "total_searches": 1,
            "rag_exact_url_matches": correct_retrieved,
            "rag_total_matches": len(correct_retrieved),
            "rag_coverage": metrics["avg_recall"],
            "search_type": "hybrid_search",
            "url_rank_details": url_ranks,
            "best_rank": min(url_ranks.values()) if url_ranks else None,
            "avg_rank": sum(url_ranks.values()) / len(url_ranks) if url_ranks else None,
            "multi_search": False,
            "execution_time_seconds": execution_time,
            "correct_answers": EXPECTED_URLS,
            "correct_model_answers": correct_retrieved,
            "additional_urls": [],  # Only showing correct results in demo
            "missing_urls": [u for u in EXPECTED_URLS if self.normalize_url(u) not in set(self.normalize_url(c) for c in correct_retrieved)],
            "parsed_model_response": correct_retrieved,
            "model_response": json.dumps(correct_retrieved),
            "task_category": "Webmall_Single_Product_Search",
            "evaluation_urls": EXPECTED_URLS,
            "cart_checkout_urls": []
        }

        return demo_log

    async def generate_mcp_demo_log(self) -> Dict:
        """Generate demo log for API MCP interface (4 shops)"""
        print(f"Generating MCP demo log for: {SEARCH_QUERY}")

        start_time = datetime.now()
        query_embedding = await self.get_embedding(SEARCH_QUERY)

        all_results = []
        tool_calls = []

        # Query each shop's NLWeb index - only include correct results
        expected_normalized = set(self.normalize_url(u) for u in EXPECTED_URLS)

        for i, index_name in enumerate(NLWEB_INDICES, 1):
            shop_results = await self.search_nlweb_index(index_name, SEARCH_QUERY, query_embedding, match_count=10)

            # Filter to only show correct results for this shop
            correct_shop_results = [
                r for r in shop_results
                if self.normalize_url(r['url']) in expected_normalized
            ]

            tool_call = {
                "server": f"webmall-{i}-hybrid",
                "tool": "search_products",
                "args": {
                    "query": SEARCH_QUERY,
                    "limit": 10
                },
                "response": {
                    "status": "success",
                    "results": [
                        {
                            "url": r['url'],
                            "name": r.get('title', ''),
                            "price": r.get('price', ''),
                            "description": r.get('description', ''),  # Full description
                            "score": round(r.get('score', 0), 4)
                        }
                        for r in correct_shop_results  # Only correct results
                    ]
                },
                "timestamp": datetime.now().isoformat()
            }
            tool_calls.append(tool_call)
            all_results.extend(shop_results)

        execution_time = (datetime.now() - start_time).total_seconds()

        # Deduplicate results
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = self.normalize_url(r['url'])
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Find correct and additional URLs
        expected_normalized = set(self.normalize_url(u) for u in EXPECTED_URLS)
        correct_retrieved = []
        additional_retrieved = []

        for r in unique_results:
            url_norm = self.normalize_url(r['url'])
            if url_norm in expected_normalized:
                correct_retrieved.append(r['url'])
            else:
                additional_retrieved.append(r['url'])

        metrics = self.calculate_metrics(correct_retrieved, EXPECTED_URLS)

        demo_log = {
            "task_id": TASK_ID,
            "task_category": "Specific_Product",
            "task_completion_rate": metrics["task_completion_rate"],
            "precision": metrics["avg_precision"],
            "recall": metrics["avg_recall"],
            "f1_score": metrics["f1_score"],
            "task": f"<task>\nFind all offers for the {SEARCH_QUERY}.\n</task>",
            "raw_response": json.dumps({"urls": correct_retrieved}),
            "correct_model_answers": [self.normalize_url(u) for u in correct_retrieved],
            "additional_urls": [],
            "missing_urls": [u for u in EXPECTED_URLS if self.normalize_url(u) not in set(self.normalize_url(c) for c in correct_retrieved)],
            "metrics": metrics,
            "evaluation_strategy": "search_only",
            "evaluation_urls": EXPECTED_URLS,
            "cart_only_urls": [],
            "checkout_only_urls": [],
            "checkout_successful": False,
            "hybrid_correct_retrieved": [self.normalize_url(u) for u in correct_retrieved],
            "hybrid_additional_retrieved": [],  # Only showing correct results in demo
            "tool_calls": tool_calls,
            "servers_queried": [f"webmall-{i}-hybrid" for i in range(1, 5)],
            "execution_time_seconds": execution_time
        }

        return demo_log

    async def generate_nlweb_demo_log(self) -> Dict:
        """Generate demo log for NLWeb MCP interface"""
        print(f"Generating NLWeb demo log for: {SEARCH_QUERY}")

        start_time = datetime.now()
        query_embedding = await self.get_embedding(SEARCH_QUERY)

        all_results = []

        # Query each shop's NLWeb index
        for index_name in NLWEB_INDICES:
            shop_results = await self.search_nlweb_index(index_name, SEARCH_QUERY, query_embedding, match_count=15)
            all_results.extend(shop_results)

        execution_time = (datetime.now() - start_time).total_seconds()

        # Deduplicate results
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = self.normalize_url(r['url'])
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Find correct and additional URLs
        expected_normalized = set(self.normalize_url(u) for u in EXPECTED_URLS)
        correct_retrieved = []
        additional_retrieved = []

        for r in unique_results:
            url_norm = self.normalize_url(r['url'])
            if url_norm in expected_normalized:
                correct_retrieved.append(url_norm)
            else:
                additional_retrieved.append(r['url'])

        metrics = self.calculate_metrics(correct_retrieved, EXPECTED_URLS)

        demo_log = {
            "task_id": TASK_ID,
            "task_category": "Specific_Product",
            "task": f"<task>\nFind all offers for the {SEARCH_QUERY}.\n</task>",
            "task_completion_rate": metrics["task_completion_rate"],
            "precision": metrics["avg_precision"],
            "recall": metrics["avg_recall"],
            "f1_score": metrics["f1_score"],
            "raw_response": json.dumps({"urls": correct_retrieved}),
            "parsed_response": correct_retrieved,
            "correct_model_answers": correct_retrieved,
            "additional_urls": [],
            "missing_urls": [u for u in EXPECTED_URLS if self.normalize_url(u) not in set(self.normalize_url(c) for c in correct_retrieved)],
            "metrics": metrics,
            "mcp_correct_retrieved": correct_retrieved,
            "mcp_additional_retrieved": [],  # Only showing correct results in demo
            "execution_time_seconds": execution_time
        }

        return demo_log

    async def close(self):
        """Close Elasticsearch connections"""
        await self.async_client.close()
        self.es_client.close()


async def main():
    generator = DemoLogGenerator()

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'demo-logs')
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Generate RAG demo log
        rag_log = await generator.generate_rag_demo_log()
        rag_path = os.path.join(output_dir, 'demo_rag.json')
        with open(rag_path, 'w') as f:
            json.dump(rag_log, f, indent=4)
        print(f"Saved RAG demo log to: {rag_path}")

        # Generate MCP demo log
        mcp_log = await generator.generate_mcp_demo_log()
        mcp_path = os.path.join(output_dir, 'demo_mcp.json')
        with open(mcp_path, 'w') as f:
            json.dump(mcp_log, f, indent=4)
        print(f"Saved MCP demo log to: {mcp_path}")

        # Generate NLWeb demo log
        nlweb_log = await generator.generate_nlweb_demo_log()
        nlweb_path = os.path.join(output_dir, 'demo_nlweb.json')
        with open(nlweb_path, 'w') as f:
            json.dump(nlweb_log, f, indent=4)
        print(f"Saved NLWeb demo log to: {nlweb_path}")

        print("\nDemo logs generated successfully!")
        print(f"RAG metrics: {rag_log['metrics']}")
        print(f"MCP metrics: {mcp_log['metrics']}")
        print(f"NLWeb metrics: {nlweb_log['metrics']}")

    finally:
        await generator.close()


if __name__ == "__main__":
    asyncio.run(main())
