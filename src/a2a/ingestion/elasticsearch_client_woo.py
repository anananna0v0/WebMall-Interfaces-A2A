"""
elasticsearch_client_woo.py
-------------------------------------
Simplified Elasticsearch client for WooCommerce ingestion.
Used together with data_ingestion_woo.py.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from elasticsearch8 import Elasticsearch
from elasticsearch8.exceptions import NotFoundError, RequestError

try:
    from .config import ELASTICSEARCH_HOST, EMBEDDING_DIMENSIONS
except ImportError:
    from config import ELASTICSEARCH_HOST, EMBEDDING_DIMENSIONS

TITLE_WEIGHT = 0.6
CONTENT_WEIGHT = 0.4

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    def __init__(self, host: str = ELASTICSEARCH_HOST):
        """Initialize Elasticsearch client."""
        self.client = Elasticsearch(
            [host],
            verify_certs=False,
            ssl_show_warn=False,
            retry_on_timeout=True,
            max_retries=3,
            timeout=30,
        )
        self.test_connection()

    def test_connection(self):
        """Test connection to Elasticsearch."""
        try:
            info = self.client.info()
            logger.info(f"Connected to Elasticsearch: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise

    def create_index(self, index_name: str, force_recreate: bool = False):
        """Create index with simple mappings for WooCommerce products."""
        if force_recreate and self.client.indices.exists(index=index_name):
            logger.info(f"Deleting existing index: {index_name}")
            self.client.indices.delete(index=index_name)

        if self.client.indices.exists(index=index_name):
            logger.info(f"Index {index_name} already exists")
            return

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "product_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        },
                        "category_analyzer": {
                            "tokenizer": "path_hierarchy",
                            "filter": ["lowercase"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "product_id": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "product_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "price": {"type": "float"},
                    "description": {
                        "type": "text",
                        "analyzer": "product_analyzer"
                    },
                    "category": {
                        "type": "text",
                        "analyzer": "category_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "title_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMENSIONS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "content_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMENSIONS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "composite_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMENSIONS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
        }

        try:
            self.client.indices.create(index=index_name, **mapping)
            logger.info(f"Created index: {index_name}")
        except RequestError as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            raise

    def create_composite_embedding(
        self, title_embedding: List[float], content_embedding: List[float]
    ) -> List[float]:
        """Combine title and content embeddings with weights."""
        try:
            title_vec = np.array(title_embedding)
            content_vec = np.array(content_embedding)
            composite = TITLE_WEIGHT * title_vec + CONTENT_WEIGHT * content_vec
            norm = np.linalg.norm(composite)
            if norm > 0:
                composite = composite / norm
            return composite.tolist()
        except Exception as e:
            logger.warning(f"Composite embedding failed: {e}")
            return [(t + c) / 2 for t, c in zip(title_embedding, content_embedding)]

    def bulk_index_products(self, index_name: str, products: List[Dict[str, Any]]):
        """Bulk index multiple products."""
        from elasticsearch8.helpers import bulk

        actions = [
            {"_index": index_name, "_id": p["product_id"], "_source": p}
            for p in products
        ]

        try:
            success, failed = bulk(self.client, actions, chunk_size=100)
            logger.info(f"Bulk indexed {success} products to {index_name}")
            if failed:
                logger.warning(f"Failed to index {len(failed)} products")
            return success, failed
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            raise

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """Get document count and index size."""
        try:
            stats = self.client.indices.stats(index=index_name)
            return {
                "document_count": stats["indices"][index_name]["total"]["docs"]["count"],
                "size_in_bytes": stats["indices"][index_name]["total"]["store"]["size_in_bytes"],
                "index_name": index_name
            }
        except NotFoundError:
            return {"error": f"Index {index_name} not found"}
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise

    def search_keyword(self, index_name: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Simple keyword-based search (for testing)."""
        try:
            response = self.client.search(
                index=index_name,
                query={
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "description", "category"]
                    }
                },
                size=top_k
            )
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def search_semantic(self, index_name: str, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search using composite embeddings."""
        try:
            response = self.client.search(
                index=index_name,
                knn={
                    "field": "composite_embedding",
                    "query_vector": query_embedding,
                    "k": top_k,
                    "num_candidates": top_k * 2
                },
                _source=["product_id", "title", "price", "url", "description", "category"],
                size=top_k
            )
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
        
    def search_hybrid_fallback(self, index_name: str, query: str, embedding_service, top_k: int = 10):
        """Keyword-first search; fallback to semantic search if too few results."""
        # Step 1: keyword search
        keyword_results = self.search_keyword(index_name, query, top_k)
        if len(keyword_results) >= 3:
            return keyword_results

        # Step 2: fallback to semantic search
        query_embedding = embedding_service.create_embedding(query)
        semantic_results = self.search_semantic(index_name, query_embedding, top_k)
        return semantic_results

