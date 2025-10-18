#!/usr/bin/env python3
"""
Data ingestion script for NLWeb MCP implementation
"""

import logging
import argparse
import json
from datetime import datetime
from dotenv import load_dotenv

# load_dotenv("../../.env")
load_dotenv()

# --- Offline mode flag ---
import os

OFFLINE_MODE = os.getenv("OFFLINE_MODE", "false").lower() == "true"
OFFLINE_DATA_DIR = os.getenv("OFFLINE_DATA_DIR", "./data")

if OFFLINE_MODE:
    print(f"[Offline mode enabled] Loading data from {OFFLINE_DATA_DIR}")
else:
    print("[Online mode] Using WooCommerce API as data source")

# Handle both relative and absolute imports
try:
    from .elasticsearch_client import ElasticsearchClient
    from .embedding_service import EmbeddingService
    from .data_ingestion import DataIngestionPipeline
    from .config import WEBMALL_SHOPS
except ImportError:
    from elasticsearch_client import ElasticsearchClient
    from embedding_service import EmbeddingService
    from data_ingestion import DataIngestionPipeline
    from config import WEBMALL_SHOPS

def setup_logging(debug: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

import pandas as pd

def offline_import_to_elasticsearch(es_client, data_dir):
    """
    Read local CSVs from data_dir and index them into Elasticsearch.
    Each CSV (webmall_1.csv ... webmall_4.csv) becomes its own index.
    """
    csv_files = [f"webmall_{i}.csv" for i in range(1, 5)]
    for csv_name in csv_files:
        csv_path = os.path.join(data_dir, csv_name)
        index_name = csv_name.replace(".csv", "_nlweb")
        print(f"[Offline import] Processing {csv_path} → index: {index_name}")

        if not os.path.exists(csv_path):
            print(f"⚠️  File not found: {csv_path}, skipping.")
            continue

        df = pd.read_csv(csv_path)
        df = df.fillna("") # Replace NaN with empty strings
        print(f"  Loaded {len(df)} rows.")

        # Create or reset index before inserting
        es_client.create_index(index_name, force_recreate=True)

        # Take first row and convert to dict
        sample_doc = df.iloc[0].to_dict()

        # Index into Elasticsearch
        es_client.index_document(index_name, sample_doc)
        print(f"  Indexed 1 sample document into {index_name}")


def main():
    parser = argparse.ArgumentParser(description='Ingest data for NLWeb MCP servers')
    parser.add_argument('--shop', choices=list(WEBMALL_SHOPS.keys()) + ['all'], 
                       default='all', help='Shop to ingest data for')
    parser.add_argument('--force-recreate', action='store_true', 
                       help='Force recreate indices (will delete existing data)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        logger.info("Initializing Elasticsearch client...")
        es_client = ElasticsearchClient()

        # --- Offline ingestion shortcut ---
        if OFFLINE_MODE:
            offline_import_to_elasticsearch(es_client, OFFLINE_DATA_DIR)
            print("[Offline ingestion completed]")
            return 0

        import os

        if os.getenv("USE_EMBEDDINGS", "true").lower() == "false":
            logger.info("Skipping embedding initialization (USE_EMBEDDINGS=false)")
            embedding_service = None
        else:
            logger.info("Initializing embedding service...")
            from embedding_service import EmbeddingService
            embedding_service = EmbeddingService()

        # logger.info("Initializing embedding service...")
        # embedding_service = EmbeddingService()
        
        # Test embedding service
        # if not embedding_service.test_embedding_service():
        #     logger.error("Embedding service test failed")
        #     return 1

        if embedding_service:
            if not embedding_service.test_embedding_service():
                logger.error("Embedding service test failed")
                return 1
            else:
                logger.info("Embedding service disabled (USE_EMBEDDINGS=false)")


        # Initialize ingestion pipeline
        logger.info("Initializing data ingestion pipeline...")
        pipeline = DataIngestionPipeline(es_client, embedding_service)
        
        # Perform ingestion
        if args.shop == 'all':
            logger.info("Starting ingestion for all shops...")
            results = pipeline.ingest_all_shops(args.force_recreate)
        else:
            logger.info(f"Starting ingestion for {args.shop}...")
            results = pipeline.ingest_shop_data(args.shop, args.force_recreate)
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'ingestion_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Ingestion completed. Results saved to {results_file}")
        
        # Print summary
        if args.shop == 'all':
            print(f"\n=== Ingestion Summary ===")
            print(f"Total shops: {results.get('total_shops', 0)}")
            print(f"Successful shops: {results.get('successful_shops', 0)}")
            print(f"Total products processed: {results.get('total_products_processed', 0)}")
            print(f"Total products failed: {results.get('total_products_failed', 0)}")
            
            for shop_id, shop_result in results.get('shop_results', {}).items():
                status = "✓" if shop_result.get('success', False) else "✗"
                processed = shop_result.get('products_processed', 0)
                print(f"  {status} {shop_id}: {processed} products")
        else:
            status = "✓" if results.get('success', False) else "✗"
            processed = results.get('products_processed', 0)
            print(f"\n{status} {args.shop}: {processed} products processed")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

