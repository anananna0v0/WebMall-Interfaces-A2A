from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from elasticsearch import Elasticsearch
from langgraph.graph import Graph, END

# --- Elasticsearch setup ---
es = Elasticsearch("http://localhost:9200")  # default local ES endpoint
INDEX_NAME = "webmall_1_nlweb"

# --- LangGraph logic (still acts as the "brain") ---
def process_query(state):
    query = state.get("query", "")

    # Perform simple text search on "name" and "description"
    search_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["Name^2", "Description", "Short description"],
                "fuzziness": "AUTO"
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
        "name": "Buyer Agent 1",
        "description": "A buyer agent that queries offline Elasticsearch index webmall_1_nlweb.",
        "capabilities": [{"name": "search_offers", "description": "Search products by keyword"}],
        "version": "0.3"
    }

@app.post("/a2a/sendMessage")
async def send_message(request: Request):
    data = await request.json()
    query = data.get("input", {}).get("text", "")
    result = workflow.invoke({"query": query})
    response_text = result["response"]
    artifacts = result["artifacts"]

    return JSONResponse({
        "output": {
            "text": response_text,
            "artifacts": artifacts
        }
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10001)
