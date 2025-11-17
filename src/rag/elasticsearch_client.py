import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from elasticsearch import AsyncElasticsearch, Elasticsearch
from elasticsearch.exceptions import RequestError, NotFoundError
from dotenv import load_dotenv
import asyncio
import re

load_dotenv()

logger = logging.getLogger(__name__)

# Constants
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
EMBEDDING_DIMENSIONS = 1536  # OpenAI text-embedding-3-small
INDEX_NAME = "webmall_v2"

# Embedding weights
TITLE_WEIGHT = 0.5
SUMMARY_WEIGHT = 0.3
CONTENT_WEIGHT = 0.2


class ElasticsearchRAGClient:
    def __init__(self, host: str = ELASTICSEARCH_HOST, index_name: str = INDEX_NAME):
        self.host = host
        self.index_name = index_name

        # Create both sync and async clients
        self.client = Elasticsearch(
            [host],
            verify_certs=False,
            ssl_show_warn=False,
            max_retries=3,
            retry_on_timeout=True,
            timeout=30,
            maxsize=25
        )

        self.async_client = AsyncElasticsearch(
            [host],
            verify_certs=False,
            ssl_show_warn=False,
            max_retries=3,
            retry_on_timeout=True,
            timeout=30,
            maxsize=25
        )

        # Test connection and create index
        self.test_connection()
        self.create_index()

    def test_connection(self):
        """Test connection to Elasticsearch"""
        try:
            info = self.client.info()
            logger.info(
                f"Connected to Elasticsearch: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise

    def create_index(self, force_recreate: bool = False):
        """Create index with V2 mappings supporting multiple embeddings and keyword search"""

        if force_recreate and self.client.indices.exists(index=self.index_name):
            logger.info(f"Deleting existing index: {self.index_name}")
            self.client.indices.delete(index=self.index_name)

        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index {self.index_name} already exists")
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
                        "keyword_analyzer": {
                            "tokenizer": "keyword",
                            "filter": ["lowercase", "trim"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    # Core fields
                    "url": {"type": "keyword"},
                    "chunk_number": {"type": "integer"},
                    "title": {
                        "type": "text",
                        "analyzer": "product_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "summary": {
                        "type": "text",
                        "analyzer": "product_analyzer"
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "product_analyzer"
                    },

                    # Separate embeddings for each field
                    "title_embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMENSIONS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "summary_embedding": {
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

                    # Metadata
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "keyword"},
                            "content_type": {"type": "keyword"},
                            "chunk_size": {"type": "integer"},
                            "crawled_at": {"type": "date"},
                            "url_path": {"type": "keyword"}
                        }
                    }
                }
            }
        }

        try:
            self.client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def create_composite_embedding(self, title_emb: List[float], summary_emb: List[float],
                                   content_emb: List[float]) -> List[float]:
        """Create weighted composite embedding from individual field embeddings"""
        # Convert to numpy arrays
        title_vec = np.array(title_emb)
        summary_vec = np.array(summary_emb)
        content_vec = np.array(content_emb)

        # Apply weights and combine
        composite = (TITLE_WEIGHT * title_vec +
                     SUMMARY_WEIGHT * summary_vec +
                     CONTENT_WEIGHT * content_vec)

        # Normalize the composite vector
        norm = np.linalg.norm(composite)
        if norm > 0:
            composite = composite / norm

        return composite.tolist()

    async def insert_chunk(self, data: Dict[str, Any]) -> bool:
        """Insert a chunk with multiple embeddings"""
        try:
            # Create composite embedding
            composite_embedding = self.create_composite_embedding(
                data['title_embedding'],
                data['summary_embedding'],
                data['content_embedding']
            )

            # Prepare document
            doc = {
                "url": data["url"],
                "chunk_number": data["chunk_number"],
                "title": data["title"],
                "summary": data["summary"],
                "content": data["content"],
                "title_embedding": data["title_embedding"],
                "summary_embedding": data["summary_embedding"],
                "content_embedding": data["content_embedding"],
                "composite_embedding": composite_embedding,
                "metadata": data.get("metadata", {})
            }

            # Create unique ID
            doc_id = f"{data['url']}_{data['chunk_number']}"

            # Index document
            await self.async_client.index(
                index=self.index_name,
                id=doc_id,
                document=doc
            )

            return True

        except Exception as e:
            logger.error(f"Failed to insert chunk: {e}")
            return False

    async def hybrid_search(self, query: str, query_embedding: List[float],
                            match_count: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword matching"""

        # Build the search query
        search_body = {
            "size": match_count * 2,  # Get more results for re-ranking
            "query": {
                "bool": {
                    "should": [
                        # Semantic search on composite embedding
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
                        # Keyword matching on title with high boost
                        {
                            "match": {
                                "title": {
                                    "query": query,
                                    "boost": 3.0
                                }
                            }
                        },
                        # Text search on summary
                        {
                            "match": {
                                "summary": {
                                    "query": query,
                                    "boost": 1.5
                                }
                            }
                        },
                        # Text search on content
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
            "_source": ["url", "title", "summary", "content", "chunk_number"],
            "highlight": {
                "fields": {
                    "title": {"fragment_size": 200},
                    "summary": {"fragment_size": 200},
                    "content": {"fragment_size": 200}
                }
            }
        }

        try:
            response = await self.async_client.search(
                index=self.index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                doc['score'] = hit['_score']
                doc['similarity'] = hit['_score']  # For compatibility

                # Add highlights if available
                if 'highlight' in hit:
                    doc['highlights'] = hit['highlight']

                results.append(doc)

            # Sort by score and return top matches
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:match_count]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def hybrid_search_title_content(self, query: str, query_embedding: List[float],
                                          match_count: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword matching"""

        # Build the search query
        search_body = {
            "size": match_count * 2,  # Get more results for re-ranking
            "query": {
                "bool": {
                    "should": [
                        # Semantic search on composite embedding
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
                        # Keyword matching on title with high boost
                        {
                            "match": {
                                "title": {
                                    "query": query,
                                    "boost": 3.0
                                }
                            }
                        },
                        # Text search on content
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
            "_source": ["url", "title", "summary", "content", "chunk_number"],
            "highlight": {
                "fields": {
                    "title": {"fragment_size": 200},
                    "summary": {"fragment_size": 200},
                    "content": {"fragment_size": 200}
                }
            }
        }

        try:
            response = await self.async_client.search(
                index=self.index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                doc['score'] = hit['_score']
                doc['similarity'] = hit['_score']  # For compatibility

                # Add highlights if available
                if 'highlight' in hit:
                    doc['highlights'] = hit['highlight']

                results.append(doc)

            # Sort by score and return top matches
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:match_count]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def hybrid_search_content_only(self, query: str, query_embedding: List[float],
                                         match_count: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword matching"""

        # Build the search query
        search_body = {
            "size": match_count * 2,  # Get more results for re-ranking
            "query": {
                "bool": {
                    "should": [
                        # Semantic search on composite embedding
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
                        # Text search on content
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
            "_source": ["url", "title", "summary", "content", "chunk_number"],
            "highlight": {
                "fields": {
                    "title": {"fragment_size": 200},
                    "summary": {"fragment_size": 200},
                    "content": {"fragment_size": 200}
                }
            }
        }

        try:
            response = await self.async_client.search(
                index=self.index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                doc['score'] = hit['_score']
                doc['similarity'] = hit['_score']  # For compatibility

                # Add highlights if available
                if 'highlight' in hit:
                    doc['highlights'] = hit['highlight']

                results.append(doc)

            # Sort by score and return top matches
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:match_count]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    async def semantic_search(self, query_embedding: List[float], match_count: int = 10) -> List[Dict[str, Any]]:
        """Perform pure semantic search using composite embeddings"""
        search_body = {
            "size": match_count,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'composite_embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "_source": ["url", "title", "summary", "content", "chunk_number", "keywords"]
        }

        try:
            response = await self.async_client.search(
                index=self.index_name,
                body=search_body
            )

            results = []
            for hit in response['hits']['hits']:
                doc = hit['_source']
                # Remove the +1.0 we added
                doc['similarity'] = (hit['_score'] - 1.0)
                results.append(doc)

            return results

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    async def get_documents_by_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Fetch full document details for specific URLs"""
        try:
            if not urls:
                return []

            query = {
                "query": {
                    "terms": {"url": urls}
                },
                "size": len(urls)
            }

            response = await self.async_client.search(
                index=self.index_name,
                body=query
            )

            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                results.append({
                    "url": source.get('url', ''),
                    "title": source.get('title', ''),
                    "content": source.get('content', ''),
                    "summary": source.get('summary', ''),
                    "shop": source.get('shop', ''),
                    "price": source.get('price', ''),
                    "score": hit['_score']
                })

            return results

        except Exception as e:
            logger.error(f"Error fetching documents by URLs: {e}")
            return []

    async def check_url_exists(self, url: str) -> bool:
        """Check if a URL exists in the index"""
        try:
            query = {
                "query": {
                    "term": {"url": url}
                }
            }

            response = await self.async_client.count(
                index=self.index_name,
                body=query
            )

            return response['count'] > 0

        except Exception as e:
            logger.error(f"Error checking URL existence: {e}")
            return False

    async def reset_index(self):
        """Delete and recreate the index"""
        try:
            if await self.async_client.indices.exists(index=self.index_name):
                await self.async_client.indices.delete(index=self.index_name)
                logger.info(f"Deleted index: {self.index_name}")

            # Recreate with sync client
            self.create_index()
            logger.info(f"Recreated index: {self.index_name}")

        except Exception as e:
            logger.error(f"Failed to reset index: {e}")
            raise

    async def close(self):
        """Close the Elasticsearch connections"""
        await self.async_client.close()
        self.client.close()
