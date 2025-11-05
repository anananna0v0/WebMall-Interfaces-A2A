import os
from dotenv import load_dotenv

# Dynamically locate the project root (the folder containing .env)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))  # up to WebMall-Interfaces-A2A
env_path = os.path.join(PROJECT_ROOT, ".env")

# print("Looking for .env at:", env_path)

load_dotenv(env_path)

WOOCOMMERCE_TIMEOUT = int(os.getenv("WOOCOMMERCE_TIMEOUT", "30"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_CHOICE", "text-embedding-3-small")

# WooCommerce stores
WEBMALL_SHOPS = {
    "webmall_1": {
        "url": os.getenv("WOO_STORE_URL_1"),
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_1"),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_1"),
        "index_name": "webmall_1"
    },
    "webmall_2": {
        "url": os.getenv("WOO_STORE_URL_2"),
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_2"),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_2"),
        "index_name": "webmall_2"
    },
    "webmall_3": {
        "url": os.getenv("WOO_STORE_URL_3"),
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_3"),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_3"),
        "index_name": "webmall_3"
    },
    "webmall_4": {
        "url": os.getenv("WOO_STORE_URL_4"),
        "consumer_key": os.getenv("WOO_CONSUMER_KEY_4"),
        "consumer_secret": os.getenv("WOO_CONSUMER_SECRET_4"),
        "index_name": "webmall_4"
    }
}

ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
EMBEDDING_DIMENSIONS = 1536
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "true").lower() == "true"

#
