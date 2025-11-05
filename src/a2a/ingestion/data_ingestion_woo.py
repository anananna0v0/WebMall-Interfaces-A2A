"""
data_ingestion_woo.py
-------------------------------------
Ingest data from the 4 WebMall WooCommerce APIs into Elasticsearch.
- Fetches real product data (id, name, permalink, price, description)
- Generates embeddings (if USE_EMBEDDINGS=true)
- Creates a clean, benchmark-compatible ES index (webmall_x)
"""

import logging
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv("../../.env")

from a2a.ingestion.woocommerce_client import WooCommerceClient
from a2a.ingestion.elasticsearch_client_woo import ElasticsearchClient
from a2a.ingestion.embedding_service import EmbeddingService
from a2a.ingestion.config import WEBMALL_SHOPS

logger = logging.getLogger(__name__)

class WooDataIngestion:
    def __init__(self, es_client: ElasticsearchClient, embed_service: EmbeddingService):
        self.es_client = es_client
        self.embed_service = embed_service
        self.shop_clients = {
            sid: WooCommerceClient(
                base_url=sc["url"],
                consumer_key=sc["consumer_key"],
                consumer_secret=sc["consumer_secret"]
            )
            for sid, sc in WEBMALL_SHOPS.items()
        }

    # ---------------------------------------------
    # Ingest a single shop
    # ---------------------------------------------
    def ingest_shop(self, shop_id: str, recreate_index: bool = False) -> Dict[str, Any]:
        if shop_id not in WEBMALL_SHOPS:
            raise ValueError(f"Invalid shop_id: {shop_id}")

        shop_cfg = WEBMALL_SHOPS[shop_id]
        index_name = shop_cfg["index_name"]

        logger.info(f"Ingesting {shop_id} → index {index_name}")
        self.es_client.create_index(index_name, force_recreate=recreate_index)

        wc = self.shop_clients[shop_id]
        if not wc.test_connection():
            logger.error(f"Cannot connect to WooCommerce API for {shop_id}")
            return {"success": False, "error": "WooCommerce connection failed"}

        raw_products = wc.get_all_products()
        if not raw_products:
            logger.warning(f"No products found for {shop_id}")
            return {"success": True, "products_processed": 0}

        processed, failed = self._process_batch(shop_id, raw_products, index_name)
        stats = self.es_client.get_index_stats(index_name)

        return {
            "success": True,
            "shop_id": shop_id,
            "products_total": len(raw_products),
            "products_processed": processed,
            "products_failed": failed,
            "index_stats": stats,
            "timestamp": datetime.now().isoformat()
        }

    # ---------------------------------------------
    # Internal batch processor
    # ---------------------------------------------
    def _process_batch(self, shop_id: str, products: List[Dict[str, Any]], index_name: str) -> tuple[int, int]:
        extracted = []
        for p in products:
            try:
                data = {
                    "product_id": str(p["id"]),
                    "title": p.get("name", ""),
                    "price": float(p.get("price", 0) or 0),
                    "url": p.get("permalink", ""),
                    "description": p.get("description", "")[:2000],
                    "category": ", ".join([c["name"] for c in p.get("categories", [])]),
                }
                extracted.append(data)
            except Exception as e:
                logger.error(f"Extract failed for product {p.get('id')}: {e}")

        if not extracted:
            return 0, len(products)

        try:
            embeddings = self.embed_service.create_separate_embeddings_batch(extracted)
        except Exception as e:
            logger.warning(f"Embedding generation failed ({e}), continuing without embeddings")
            embeddings = [{"title_embedding": [], "content_embedding": []} for _ in extracted]

        docs = []
        for i, prod in enumerate(extracted):
            try:
                emb = embeddings[i]
                doc = {
                    "product_id": prod["product_id"],
                    "title": prod["title"],
                    "price": prod["price"],
                    "url": prod["url"],
                    "description": prod["description"],
                    "category": prod["category"],
                    "title_embedding": emb.get("title_embedding", []),
                    "content_embedding": emb.get("content_embedding", []),
                    "composite_embedding": self.es_client.create_composite_embedding(
                        emb.get("title_embedding", []),
                        emb.get("content_embedding", [])
                    ),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                docs.append(doc)
            except Exception as e:
                logger.error(f"Failed to prepare doc for {prod['product_id']}: {e}")

        if docs:
            success, failed = self.es_client.bulk_index_products(index_name, docs)
            return success, len(failed or [])
        return 0, len(products)

    # ---------------------------------------------
    # Ingest all shops
    # ---------------------------------------------
    def ingest_all(self, recreate_indices: bool = False) -> Dict[str, Any]:
        results = {}
        logger.info("Starting ingestion for all WebMall shops")

        for sid in WEBMALL_SHOPS.keys():
            try:
                results[sid] = self.ingest_shop(sid, recreate_indices)
            except Exception as e:
                logger.error(f"Failed to ingest {sid}: {e}")
                results[sid] = {"success": False, "error": str(e)}

        summary = {
            "shops_total": len(WEBMALL_SHOPS),
            "shops_successful": sum(1 for r in results.values() if r.get("success")),
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        logger.info(f"Ingestion complete: {summary}")
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    es = ElasticsearchClient()
    embed = EmbeddingService()
    pipeline = WooDataIngestion(es, embed)
    pipeline.ingest_all(recreate_indices=True)
