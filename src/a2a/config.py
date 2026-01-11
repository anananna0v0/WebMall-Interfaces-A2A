import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory is the project root (three levels up from src/a2a/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")

# --- Elasticsearch and AI Configuration ---
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Infrastructure Mapping ---
WEBMALL_SHOPS = {
    "webmall_1": {"index_name": "webmall_1_nlweb", "url": "https://webmall-1.informatik.uni-mannheim.de"},
    "webmall_2": {"index_name": "webmall_2_nlweb", "url": "https://webmall-2.informatik.uni-mannheim.de"},
    "webmall_3": {"index_name": "webmall_3_nlweb", "url": "https://webmall-3.informatik.uni-mannheim.de"},
    "webmall_4": {"index_name": "webmall_4_nlweb", "url": "https://webmall-4.informatik.uni-mannheim.de"}
}

# --- Task Data Path ---
TASK_SET_FILE = "task_sets_33.json"
TASK_SET_PATH = BASE_DIR / "task_sets" / TASK_SET_FILE