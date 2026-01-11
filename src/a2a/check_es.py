import logging
from elasticsearch import Elasticsearch
from a2a.config import ELASTICSEARCH_HOST, WEBMALL_SHOPS

# Set up simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("es_checker")

def check_index_structure():
    es = Elasticsearch([ELASTICSEARCH_HOST], verify_certs=False, ssl_show_warn=False)
    
    for shop_id, info in WEBMALL_SHOPS.items():
        index_name = info["index_name"]
        print(f"\n{'='*50}")
        print(f"Checking Index: {index_name} ({shop_id})")
        
        if not es.indices.exists(index=index_name):
            print(f"❌ Error: Index '{index_name}' does not exist!")
            continue

        # 1. Check Mapping
        mapping = es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]["mappings"].get("properties", {})
        
        target_field = "composite_embedding"
        if target_field in properties:
            print(f"✅ Field '{target_field}' exists.")
            print(f"   - Type: {properties[target_field]['type']}")
            print(f"   - Dims: {properties[target_field].get('dims')}")
        else:
            print(f"❌ Error: Missing '{target_field}' field!")
            print(f"   Available fields: {list(properties.keys())}")

        # 2. Check Sample Document Data
        sample = es.search(index=index_name, body={"query": {"match_all": {}}, "size": 1})
        hits = sample.get("hits", {}).get("hits", [])
        
        if hits:
            source = hits[0]["_source"]
            print(f"✅ Sample document found (ID: {hits[0]['_id']})")
            print(f"   - Title: {source.get('title', 'N/A')}")
            print(f"   - URL: {source.get('url', 'N/A')}")
            print(f"   - Has composite_embedding: {target_field in source}")
            
            # Check if embedding is all zeros (common issue)
            if target_field in source:
                vec = source[target_field]
                if all(v == 0 for v in vec[:10]):
                    print(f"⚠️ Warning: Embedding starts with all zeros. Data might be corrupted.")
        else:
            print(f"⚠️ Warning: Index is empty (no documents found).")

if __name__ == "__main__":
    check_index_structure()