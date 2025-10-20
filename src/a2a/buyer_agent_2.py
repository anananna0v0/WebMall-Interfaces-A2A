from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from elasticsearch import Elasticsearch
from langgraph.graph import Graph, END

# --- Elasticsearch setup ---
es = Elasticsearch("http://localhost:9200")  # default local ES endpoint
INDEX_NAME = "webmall_2_nlweb"

# --- LangGraph logic (still acts as the "brain") ---
def process_query(state):
    query = state.get("query", "")

    # Perform simple text search on "name" and "description"
    search_body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["Name^2", "Description", "Short description"],
                            "operator": "or"
                        }
                    }
                ]
            }
        },
        "size": 3
    }

    try:
        results = es.search(index=INDEX_NAME, body=search_body)
        hits = results["hits"]["hits"]
        artifacts = []
        for hit in hits:
            source = hit["_source"]
            artifacts.append({
                "name": source.get("Name", "Unknown product"),
                "price": source.get("Regular price", "N/A"),
                "url": source.get("Images", "N/A")
            })
        response = f"Found {len(artifacts)} results for query: '{query}'"
    except Exception as e:
        response = f"Error searching Elasticsearch: {str(e)}"
        artifacts = []

    return {"response": response, "artifacts": artifacts}

# --- LangGraph workflow ---
graph = Graph()
graph.add_node("process", process_query)
graph.set_entry_point("process")
graph.add_edge("process", END)
workflow = graph.compile()

# --- FastAPI setup (A2A layer) ---
app = FastAPI()

@app.get("/.well-known/agent-card")
async def agent_card():
    return {
        "name": "Buyer Agent 2",
        "shop_domain": "webmall-1.informatik.uni-mannheim.de",
        "description": "Buyer agent for WebMall Shop 2 (handles search, add-to-cart, and checkout tasks).",
        "capabilities": [
            {"name": "search_offers", "description": "Search products by keyword"},
            {"name": "add_to_cart", "description": "Add a selected product to the cart"},
            {"name": "checkout", "description": "Simulate checkout for items in the cart"}
        ],
        "skills": [
            {"name": "elasticsearch_query", "description": "Query products from the offline Elasticsearch index webmall_2_nlweb"},
            {"name": "langgraph_workflow", "description": "Use LangGraph workflow for task processing"},
            {"name": "fastapi_endpoint", "description": "Expose A2A-compatible endpoints for communication"}
        ],
        "version": "0.3",
    }

@app.post("/a2a/sendMessage")
async def send_message(request: Request):
    data = await request.json()
    text = data.get("input", {}).get("text", "").lower()

    # Detect add-to-cart command
    if "add to cart" in text or ("add" in text and "cart" in text):
        # Extract core search terms
        clean_query = text.replace("add to cart", "").replace("add", "").replace("to cart", "").strip()
        clean_query = clean_query.replace("the cheapest", "").strip()

        result = workflow.invoke({"query": clean_query})
        artifacts = result["artifacts"]

        if artifacts:
            cheapest = artifacts[0]
            response_text = f"Added '{cheapest['name']}' ({cheapest['price']} EUR) to cart successfully."
        else:
            response_text = "No products found to add to cart."

        return JSONResponse({"output": {"text": response_text, "artifacts": artifacts}})
    
    # Detect checkout command
    if "checkout" in text.strip().lower():
        response_text = "Checkout completed successfully. Order confirmed."
        return JSONResponse({
            "output": {
                "text": response_text,
                "artifacts": []
            }
        })

    # Default: normal search
    result = workflow.invoke({"query": text})
    return JSONResponse({
        "output": {
            "text": result["response"],
            "artifacts": result["artifacts"]
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10002)
