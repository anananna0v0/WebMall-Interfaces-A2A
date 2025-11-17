import os
import sys
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Import calculation function from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import calculation_results, interface_results_dir, extract_reasoning_effort
from rag.elasticsearch_client import ElasticsearchRAGClient
import json
import time
from typing import Dict, List, Any, Tuple
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import csv


# LangChain imports for model-agnostic approach
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

# Import cart tools
from rag.rag_cart_tools import get_cart_tools, reset_all_carts



load_dotenv()

# Configuration: webmall URLs
URLS = {
    "URL_1": "https://webmall-1.informatik.uni-mannheim.de",
    "URL_2": "https://webmall-2.informatik.uni-mannheim.de",
    "URL_3": "https://webmall-3.informatik.uni-mannheim.de",
    "URL_4": "https://webmall-4.informatik.uni-mannheim.de",
    "URL_5": "https://webmall-solution.informatik.uni-mannheim.de"
}

# Initialize Elasticsearch client for RAG
es_client = ElasticsearchRAGClient()

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing trailing slashes and converting to lowercase."""
    return url.rstrip('/').lower()


def fill_urls(text: str, urls: Dict[str, str]) -> str:
    """Replace URL placeholders with actual URLs."""
    for key, val in urls.items():
        text = text.replace("{{" + key + "}}", val)
    return text


async def get_embedding(text: str) -> Tuple[List[float], int]:
    """Get embedding vector from OpenAI and return tokens used."""
    try:
        # Use LangChain embeddings
        embedding = await embeddings_model.aembed_query(text)
        # Estimate tokens (roughly 1 token per 4 characters)
        tokens_used = len(text) // 4
        return embedding, tokens_used
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536, 0  # Return zero vector and zero tokens on error


# Global variables for tracking
search_history = []
search_results_cache = []  # Store actual results for easy access
token_tracker = {"embedding_tokens": 0}


@tool
async def search_products(query: str, match_count: int = 30, use_hybrid: bool = True) -> str:
    """Search for products in the webmall database. Use this to find specific products, compatible items, or browse categories.

    Args:
        query: The search query. Be specific and use exact product names when possible.
        match_count: Number of results to retrieve (1-100). Use more results for broad searches, fewer for specific items. Default is 30.
        use_hybrid: Whether to use hybrid search (combines semantic + keyword matching). Generally recommended. Default is True.

    Returns:
        JSON string containing search results with status, query, results count, and product information.
    """
    global search_history, search_results_cache, token_tracker

    # Validate match_count
    match_count = max(1, min(100, match_count))

    print(
        f"\n🔍 SEARCH TOOL: Query='{query}', Results={match_count}, Mode={'hybrid' if use_hybrid else 'semantic'}")

    # Get embedding for the query
    query_embedding, embedding_tokens = await get_embedding(query)
    token_tracker["embedding_tokens"] += embedding_tokens

    # Perform search
    if use_hybrid:
        # results = await es_client.hybrid_search(query, query_embedding, match_count)
        # results = await es_client.hybrid_search_content_only(query, query_embedding, match_count)
        results = await es_client.hybrid_search_title_content(query, query_embedding, match_count)
    else:
        results = await es_client.semantic_search(query_embedding, match_count)

    # Store results in cache for easy access
    search_results_cache.append(results)

    # Track search in history
    search_record = {
        "query": query,
        "match_count": match_count,
        "use_hybrid": use_hybrid,
        "results_found": len(results),
        "timestamp": datetime.now().isoformat()
    }
    search_history.append(search_record)

    print(f"✅ Found {len(results)} results")

    # Return structured response with full results for the agent
    return_string = json.dumps({
        "status": "success",
        "query": query,
        "results_count": len(results),
        "results": [
            {
                "title": r["title"],
                "url": r["url"],
                # "description": r.get("content", "N/A")[:250]
            }
            # Return lightweight results, let the agent fetch details for specific products
            for r in results
        ],
        "search_mode": "hybrid" if use_hybrid else "semantic"
    })
    # print(return_string)
    return return_string


@tool
async def get_product_details(urls: List[str]) -> str:
    """Get detailed information for specific product URLs. Use this after search_products to get full descriptions and details for products you're interested in.

    Args:
        urls: List of product URLs to fetch details for. Maximum 20 URLs per request.

    Returns:
        JSON string containing detailed product information including descriptions, content, summaries, prices, and shop information.
    """
    global token_tracker

    # Validate and limit URLs
    if not urls:
        return json.dumps({"status": "error", "error": "No URLs provided"})

    urls = urls[:20]  # Limit to 20 URLs to prevent excessive token usage

    print(f"\n📋 DETAILS TOOL: Fetching details for {len(urls)} URLs")

    try:
        # Fetch detailed information from Elasticsearch
        detailed_results = await es_client.get_documents_by_urls(urls)

        print(f"✅ Retrieved details for {len(detailed_results)} products")

        # Return structured response with detailed information
        return_string = json.dumps({
            "status": "success",
            "urls_requested": len(urls),
            "details_found": len(detailed_results),
            "product_details": [
                {
                    "title": r["title"],
                    "url": r["url"],
                    "description": r.get("content", "N/A")
                }
                for r in detailed_results
            ]
        })

        return return_string

    except Exception as e:
        print(f"❌ Error fetching product details: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e),
            "urls_requested": len(urls)
        })


def aggregate_search_results(all_results: List[List[Dict]], expected_urls: List[str]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Aggregate results from multiple searches, removing duplicates and tracking ranks.

    Returns:
        - Deduplicated results sorted by best score
        - Mapping of expected URLs to their best ranks across all searches
    """
    # Dictionary to track best result for each URL
    url_best_results = {}
    url_best_ranks = {}

    # Process each search result set
    for search_idx, results in enumerate(all_results):
        for rank, doc in enumerate(results, 1):
            url = doc['url']
            score = doc.get('score', doc.get('similarity', 0))

            # Track best result for this URL
            if url not in url_best_results or score > url_best_results[url].get('score', 0):
                url_best_results[url] = doc
                url_best_results[url]['best_search_idx'] = search_idx
                url_best_results[url]['aggregated_score'] = score

            # Track best rank for expected URLs
            normalized_url = normalize_url(url)
            for expected_url in expected_urls:
                if normalize_url(expected_url) == normalized_url:
                    if expected_url not in url_best_ranks or rank < url_best_ranks[expected_url]:
                        url_best_ranks[expected_url] = rank

    # Sort aggregated results by score
    aggregated_results = sorted(
        url_best_results.values(),
        key=lambda x: x.get('aggregated_score', 0),
        reverse=True
    )

    return aggregated_results, url_best_ranks


def extract_urls_from_cart_tool_output(tool_output: str, tool_name: str) -> List[str]:
    """Extract URLs from cart/checkout tool output."""
    try:
        if isinstance(tool_output, str):
            response_data = json.loads(tool_output)
        else:
            response_data = tool_output

        urls = set()  # Use set to avoid duplicates

        # For cart tools, prioritize cart_urls if available
        if "cart_urls" in response_data:
            urls.update(response_data["cart_urls"])
        elif "cart" in response_data and isinstance(response_data["cart"], list):
            # Fallback to extracting from cart items if cart_urls not available
            for item in response_data["cart"]:
                if "url" in item:
                    urls.add(item["url"])

        # For checkout tools, prioritize product_urls if available
        if "product_urls" in response_data:
            urls.update(response_data["product_urls"])
        elif "items" in response_data and isinstance(response_data["items"], list):
            # Fallback to extracting from items if product_urls not available
            for item in response_data["items"]:
                if "url" in item:
                    urls.add(item["url"])

        return list(urls)

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Warning: Could not extract URLs from {tool_name} output: {e}")
        return []


def message_content_to_text(content: Any) -> str:
    """Normalize LangChain/SSE message content to plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get(
                    "content") or item.get("value")
                if isinstance(text_value, str):
                    parts.append(text_value)
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    return str(content)


def parse_model_answer(answer: Any) -> List[str]:
    """Parse the model answer and return the list of URLs."""
    normalized_answer = message_content_to_text(answer)

    try:
        # Try to extract JSON array from the response
        if normalized_answer.startswith('[') and normalized_answer.endswith(']'):
            return json.loads(normalized_answer)
        else:
            # If response contains JSON within text, try to find it
            import re
            json_match = re.search(r'\[.*\]', normalized_answer, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: treat as single item or "Done"
                cleaned = normalized_answer.strip()
                return [cleaned] if cleaned.lower() != "done" else ["Done"]
    except json.JSONDecodeError:
        print("Warning: Could not parse JSON response, treating as plain text")
        # Fallback to original ### splitting method
        return [part.strip() for part in normalized_answer.split("###") if part.strip()]


def create_fallback_result(user_task: str, urls_in_db: List[str], expected_flat: List[str],
                           total_tokens_used: Dict, error_message: str, execution_time: float) -> Dict:
    """Create a fallback result structure when the agent fails."""
    print(f"🚨 CREATING FALLBACK RESULT: {error_message}")

    return {
        "parsed_urls": [],
        "answer": f"AGENT_FAILED: {error_message}",
        "search_history": [],
        "total_searches": 0,
        "aggregated_results": [],
        "rag_exact_url_matches": [],
        "rag_total_matches": 0,
        "rag_coverage": 0.0,
        "url_ranks": {},
        "best_rank": None,
        "avg_rank": None,
        "db_coverage": len(urls_in_db) / len(expected_flat) if expected_flat else 0,
        "search_type": "multi_search_failed",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "execution_time_seconds": execution_time,
        "tool_calls_log": [],
        "cart_checkout_urls": [],
        "error_occurred": True,
        "error_message": error_message,
        "error_type": "GraphRecursionError"
    }


async def get_model_answer(user_task: str, urls_in_db: List[str], expected_flat: List[str],
                           total_tokens_used: Dict, chat_model: Any,
                           model_name: str = "gpt-4") -> Dict:
    """Get model answer using the LangGraph RAG system with tool binding."""

    # Start execution timer
    task_start_time = time.time()

    print("Starting RAG workflow...")

    # Reset global variables for this task
    global search_history, search_results_cache
    search_history = []
    search_results_cache = []

    # Create the system prompt with intelligent search strategy guidance
    system_prompt = """You are an advanced RAG-capable agent that can browse four webshops, find product offers, manage shopping carts, and complete purchases.
You have access to search functions, product detail fetching, cart management tools, and checkout capabilities for all four shops.

AVAILABLE TOOLS:
- search_products: Search for products across all shops (returns title + URL only for token efficiency)
- get_product_details: Get detailed descriptions for specific URLs (use after search to get full info)
- add_to_cart_webmall_1 through add_to_cart_webmall_4: Add products to specific shop carts
- checkout_webmall_1 through checkout_webmall_4: Complete purchases with customer details

EFFICIENT SEARCH WORKFLOW:
1. Use search_products first to get an overview of available products (lightweight: title + URL only)
2. Review search results and identify promising products
3. Use get_product_details for URLs you're interested in to get full descriptions, specs, and pricing
4. Make decisions based on detailed information

TASK-SPECIFIC INSTRUCTIONS:

FOR SEARCH TASKS:
1. OPTIMIZE FOR EFFICIENCY: Use the two-step search approach to minimize token usage.
2. QUERY ANALYSIS: 
   - Simple specific product searches (e.g., "AMD Ryzen 9 5900X"): Start with 10-15 results
   - Complex/compatibility queries (e.g., "cables compatible with..."): May need 30 results
   - Comparison queries (e.g., "better than X"): Break into separate searches
3. Get details only for products that seem relevant from titles
4. After analysis, return a JSON array with the EXACT URLs of all relevant products found.

FOR ADD TO CART TASKS:
1. First search for the requested products
2. Extract the product URLs from search results
3. Group URLs by shop (webmall_1, webmall_2, etc.)
4. Call the appropriate add_to_cart tool for each shop with their respective URLs
5. Return the URLs of products successfully added to carts

FOR CHECKOUT TASKS:
1. If products are already in cart, proceed directly to checkout
2. If starting fresh, first add products to cart
3. Call the checkout tool with all required customer and payment information
4. Return the product URLs from the completed order

RESPONSE FORMAT:
- For all tasks: Return a JSON array containing the EXACT URLs
- Format: ["url1", "url2", ...] or ["Done"] if task completed/no results"""

    # Get cart tools
    cart_tools = get_cart_tools()

    # Create a React agent that handles tool calling automatically
    # Set recursion limit to 50 (double the default) to handle complex tasks
    agent = create_react_agent(
        model=chat_model, tools=[search_products, get_product_details, *cart_tools])

    # Run the agent with proper token tracking and error handling
    try:
        with get_usage_metadata_callback() as cb:
            result = await agent.ainvoke(
                {"messages": [SystemMessage(
                    content=system_prompt), HumanMessage(content=user_task)]},
                config={"recursion_limit": 50}  # Increase recursion limit
            )
    except GraphRecursionError as e:
        execution_time = time.time() - task_start_time
        error_msg = f"GraphRecursionError: Recursion limit exceeded - {str(e)}"
        print(f"❌ AGENT RECURSION ERROR: {error_msg}")
        print(f"⏱️  Failed after {execution_time:.2f} seconds")
        return create_fallback_result(user_task, urls_in_db, expected_flat, total_tokens_used, error_msg, execution_time)
    except Exception as e:
        execution_time = time.time() - task_start_time
        error_msg = f"Unexpected agent error: {str(e)}"
        print(f"❌ AGENT UNEXPECTED ERROR: {error_msg}")
        print(f"⏱️  Failed after {execution_time:.2f} seconds")
        return create_fallback_result(user_task, urls_in_db, expected_flat, total_tokens_used, error_msg, execution_time)

    # Extract token usage from callback
    usage_data = cb.usage_metadata
    print(f"📊 Token Usage: {usage_data}")

    for _, usage in usage_data.items():
        total_tokens_used["prompt_tokens"] += usage.get("input_tokens", 0)
        total_tokens_used["completion_tokens"] += usage.get("output_tokens", 0)
        total_tokens_used["total_tokens"] += usage.get("total_tokens", 0)

    # Get search results from our cache (populated by the tool calls)
    all_search_results = search_results_cache.copy()

    # Track tool calls and extract cart/checkout URLs
    agent_messages = result.get("messages", [])
    tool_calls_log = []
    cart_checkout_urls = set()

    # Add search history to tool calls log
    for search_record in search_history:
        tool_calls_log.append({
            "tool_name": "search_products",
            "tool_args": {
                "query": search_record["query"],
                "match_count": search_record["match_count"],
                "use_hybrid": search_record["use_hybrid"]
            },
            "tool_output": {
                "results_found": search_record["results_found"],
                "status": "success"
            },
            "timestamp": search_record["timestamp"],
            "tool_type": "search"
        })

    # Process all messages to extract tool calls
    for msg in agent_messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_output = None
                tool_output_parsed = None

                # Find corresponding tool result message
                for result_msg in agent_messages:
                    if hasattr(result_msg, 'tool_call_id') and result_msg.tool_call_id == tool_call.get("id"):
                        tool_output = result_msg.content
                        # Try to parse JSON output for structured data
                        try:
                            if tool_output and tool_output.strip().startswith('{'):
                                tool_output_parsed = json.loads(tool_output)
                        except json.JSONDecodeError:
                            pass
                        break

                # Determine tool type
                tool_type = "unknown"
                if tool_name == "search_products":
                    tool_type = "search"
                elif tool_name == "get_product_details":
                    tool_type = "details"
                elif tool_name.startswith("add_to_cart_"):
                    tool_type = "cart"
                elif tool_name.startswith("checkout_"):
                    tool_type = "checkout"

                # Log tool call with enhanced information
                tool_call_entry = {
                    "tool_name": tool_name,
                    "tool_type": tool_type,
                    "tool_args": tool_call.get("args", {}),
                    "tool_output_raw": tool_output,
                    "timestamp": datetime.now().isoformat()
                }

                # Add parsed output if available
                if tool_output_parsed:
                    tool_call_entry["tool_output_parsed"] = tool_output_parsed

                    # Add specific metrics for different tool types
                    if tool_type == "cart" and "cart" in tool_output_parsed:
                        tool_call_entry["items_in_cart"] = len(
                            tool_output_parsed.get("cart", []))
                        tool_call_entry["total_quantity"] = tool_output_parsed.get(
                            "total_items", 0)
                    elif tool_type == "checkout" and "items" in tool_output_parsed:
                        tool_call_entry["items_purchased"] = len(
                            tool_output_parsed.get("items", []))
                        tool_call_entry["order_id"] = tool_output_parsed.get(
                            "order_id", "")
                        tool_call_entry["total_amount"] = tool_output_parsed.get(
                            "total", "0.00")

                # Skip search_products as they're already added above
                if tool_name != "search_products":
                    tool_calls_log.append(tool_call_entry)

                # Extract URLs from cart/checkout tools
                if tool_name.startswith(("add_to_cart_", "checkout_")) and tool_output:
                    urls = extract_urls_from_cart_tool_output(
                        tool_output, tool_name)
                    cart_checkout_urls.update(urls)
                    print(
                        f"🛒 Extracted {len(urls)} URLs from {tool_name}: {urls}")

    # Get final answer from the agent's last message
    final_message = agent_messages[-1] if agent_messages else None
    answer = final_message.content if final_message and hasattr(
        final_message, 'content') else "No answer provided"

    # Aggregate all search results
    aggregated_results, url_ranks = aggregate_search_results(
        all_search_results, expected_flat)

    # Parse the agent's final answer directly
    parsed_urls = parse_model_answer(answer)

    print(f"\n📊 TOOL EXECUTION SUMMARY:")
    search_tools = [t for t in tool_calls_log if t.get(
        "tool_type") == "search"]
    details_tools = [t for t in tool_calls_log if t.get(
        "tool_type") == "details"]
    cart_tools = [t for t in tool_calls_log if t.get("tool_type") == "cart"]
    checkout_tools = [t for t in tool_calls_log if t.get(
        "tool_type") == "checkout"]

    print(f"  - Total tool calls: {len(tool_calls_log)}")
    print(f"  - Search tools: {len(search_tools)}")
    print(f"  - Details tools: {len(details_tools)}")
    print(f"  - Cart tools: {len(cart_tools)}")
    print(f"  - Checkout tools: {len(checkout_tools)}")
    print(f"  - Total unique products found: {len(aggregated_results)}")
    print(f"  - Agent returned {len(parsed_urls)} URLs")

    if search_tools:
        print(
            f"  - Search queries: {[t['tool_args'].get('query') for t in search_tools]}")
    if cart_checkout_urls:
        print(f"  - Cart/Checkout URLs: {len(cart_checkout_urls)} URLs")

    # Check coverage
    found_urls = [r['url'] for r in aggregated_results]
    found_normalized = [normalize_url(url) for url in found_urls]

    exact_url_matches = []
    for expected_url in expected_flat:
        if normalize_url(expected_url) in found_normalized:
            exact_url_matches.append(expected_url)

    rag_coverage = len(exact_url_matches) / \
        len(expected_flat) if expected_flat else 0

    # Display coverage and ranking info
    if url_ranks:
        best_rank = min(url_ranks.values())
        avg_rank = sum(url_ranks.values()) / len(url_ranks)

    # Calculate execution time
    execution_time = time.time() - task_start_time
    print(f"⏱️  Execution time: {execution_time:.2f} seconds")

    # Calculate retrieval metrics
    db_coverage = len(urls_in_db) / len(expected_flat) if expected_flat else 0

    # Return comprehensive results
    return {
        "parsed_urls": parsed_urls,
        "answer": answer,
        "search_history": search_history,
        "total_searches": len(all_search_results),
        "aggregated_results": aggregated_results,
        "rag_exact_url_matches": exact_url_matches,
        "rag_total_matches": len(exact_url_matches),
        "rag_coverage": rag_coverage,
        "url_ranks": url_ranks,
        "best_rank": min(url_ranks.values()) if url_ranks else None,
        "avg_rank": sum(url_ranks.values()) / len(url_ranks) if url_ranks else None,
        "db_coverage": db_coverage,
        "search_type": "multi_search",
        "prompt_tokens": total_tokens_used["prompt_tokens"],
        "completion_tokens": total_tokens_used["completion_tokens"],
        "total_tokens": total_tokens_used["total_tokens"],
        "execution_time_seconds": execution_time,
        "tool_calls_log": tool_calls_log,
        "cart_checkout_urls": list(cart_checkout_urls)
    }


# Load benchmark JSON file
BENCHMARK_JSON_PATH = "task_sets.json"

with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
    benchmark = json.load(f)


async def process_benchmark(model_name: str, chat_model: Any):
    """Process benchmark tasks using the LangGraph RAG workflow."""
    print("\n" + "=" * 60)
    print("BENCHMARK - RAG SYSTEM")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)

    reasoning_effort = extract_reasoning_effort(chat_model)
    results_output_dir = interface_results_dir(
        __file__, "rag", model_name, reasoning_effort)
    # Create a run-unique timestamp early so stream files are consistent
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare incremental output files (streaming)
    incremental_csv_file = results_output_dir / \
        f"benchmark_metrics_{current_timestamp}_stream.csv"
    incremental_jsonl_file = results_output_dir / \
        f"benchmark_results_{current_timestamp}.jsonl"

    # Initialize the streaming CSV with header
    with incremental_csv_file.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "category",
            "task_id",
            "task_completion_rate",
            "avg_precision",
            "avg_recall",
            "f1_score",
            "prompt_tokens",
            "completion_tokens",
            "execution_duration"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Reset all carts before starting benchmark
    reset_all_carts()
    print("Reset all shopping carts before starting benchmark")

    # Initialize results list
    results = []

    # Initialize counters for overall statistics
    total_urls_not_in_db = 0
    total_expected_urls = 0
    total_searches_performed = 0
    total_execution_time = 0.0
    total_tool_calls = 0
    total_cart_tools = 0
    total_checkout_tools = 0

    # Error tracking
    failed_tasks = 0
    recursion_errors = 0
    other_errors = 0

    # Ranking metrics
    total_rank_sum = 0
    total_ranked_results = 0
    top3_count = 0
    top10_count = 0

    # Initialize token tracking
    total_tokens_used = {
        "embedding_tokens": 0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0
    }

    # Process tasks
    for task_set in benchmark:
        for task in task_set["tasks"]:
            # if task["id"] != "Webmall_Add_To_Cart_Task4":
            #    continue

            print(f"\n=== TASK {task['id']} ===")
            # Reset carts before each task to ensure clean state
            reset_all_carts()
            # Track the start time for the whole task (in case preprocessing fails)
            task_preprocess_start = time.time()

            # Defaults to ensure we can still log on failure
            expected_flat = []
            urls_in_db = []
            urls_not_in_db = []
            user_task = ""
            task_category = task.get("category", "Search")
            preprocessing_failed = False

            try:
                # Check if correct answers exist in the database
                correct_answer = task.get(
                    "correct_answer", {}).get("answers", [])
                expected_flat = [fill_urls(x, URLS) for x in correct_answer]

                # Check database for exact URL matches
                for expected_url in expected_flat:
                    if not expected_url.endswith("/"):
                        expected_url += "/"
                    if await es_client.check_url_exists(expected_url):
                        urls_in_db.append(expected_url)
                    else:
                        urls_not_in_db.append(expected_url)

                total_urls_not_in_db += len(urls_not_in_db)

                if len(urls_not_in_db) > 0:
                    print(
                        f"  ⚠️  WARNING: {len(urls_not_in_db)} correct answer(s) not found in database!")

                total_expected_urls += len(expected_flat)

                # Get task category for evaluation logic
                task_category = task.get("category", "Search")
                print(f"  📝 Task Category: {task_category}")

                # Extract user task
                user_task = task["task"] if "task" in task else None
                if not user_task:
                    start = task["instruction"].find("<task>")
                    end = task["instruction"].find("</task>") + len("</task>")
                    user_task = task["instruction"][start:end]

                user_task = fill_urls(user_task, URLS)
                user_task = user_task.replace(
                    "<task>", "").replace("</task>", "")

                if "{{product_url}}" in user_task:
                    user_task = user_task.replace(
                        "{{product_url}}", str(expected_flat))

                if "{{email}}" in user_task:
                    user_details = task["user_details"]
                    user_task = user_task.replace(
                        "{{name}}", user_details["name"])
                    user_task = user_task.replace(
                        "{{email}}", user_details["email"])
                    user_task = user_task.replace(
                        "{{street}}", user_details["street"])
                    user_task = user_task.replace(
                        "{{house_number}}", user_details["house_number"])
                    user_task = user_task.replace(
                        "{{zip}}", user_details["zip"])
                    user_task = user_task.replace(
                        "{{city}}", user_details["city"])
                    user_task = user_task.replace(
                        "{{state}}", user_details["state"])
                    user_task = user_task.replace(
                        "{{country}}", user_details["country"])

                    # Replace payment info placeholders
                    payment_info = task["payment_info"]
                    user_task = user_task.replace(
                        "{{card}}", payment_info["card"])
                    user_task = user_task.replace(
                        "{{cvv}}", payment_info["cvv"])
                    user_task = user_task.replace(
                        "{{expiry_date}}", payment_info["expiry_date"])

                print(f"User task: {user_task}")
            except Exception as e:
                # If preprocessing fails, ensure we still log a result for this task
                preprocessing_failed = True
                error_msg = f"TaskProcessingError: {str(e)}"
                print(
                    f"❌ PREPROCESSING ERROR in task {task['id']}: {error_msg}")
                execution_time = time.time() - task_preprocess_start
                # Create a fallback result so downstream logging works
                model_result = create_fallback_result(
                    user_task or "",
                    urls_in_db,
                    expected_flat,
                    total_tokens_used,
                    error_msg,
                    execution_time,
                )
                failed_tasks += 1
                other_errors += 1

            # Capture token usage before this task
            tokens_before_task = {
                "prompt_tokens": total_tokens_used["prompt_tokens"],
                "completion_tokens": total_tokens_used["completion_tokens"],
                "total_tokens": total_tokens_used["total_tokens"]
            }

            # Get model answer using LangGraph system with error handling
            if not preprocessing_failed:
                try:
                    model_result = await get_model_answer(
                        user_task,
                        urls_in_db,
                        expected_flat,
                        total_tokens_used,
                        chat_model=chat_model,
                        model_name=model_name
                    )

                    # Check if this was a failed result
                    if model_result.get("error_occurred", False):
                        failed_tasks += 1
                        error_type = model_result.get("error_type", "unknown")
                        if error_type == "GraphRecursionError":
                            recursion_errors += 1
                        else:
                            other_errors += 1
                        print(
                            f"⚠️  Task {task['id']} failed with error: {model_result.get('error_message', 'Unknown error')}")

                except Exception as e:
                    # Fallback error handling if even the error handling fails
                    execution_time = time.time() - task_preprocess_start
                    error_msg = f"Critical benchmark error: {str(e)}"
                    print(
                        f"🔥 CRITICAL ERROR in task {task['id']}: {error_msg}")
                    model_result = create_fallback_result(
                        user_task, urls_in_db, expected_flat, total_tokens_used, error_msg, execution_time)
                    failed_tasks += 1
                    other_errors += 1

            # Calculate per-task token usage
            task_tokens = {
                "prompt_tokens": total_tokens_used["prompt_tokens"] - tokens_before_task["prompt_tokens"],
                "completion_tokens": total_tokens_used["completion_tokens"] - tokens_before_task["completion_tokens"],
                "total_tokens": total_tokens_used["total_tokens"] - tokens_before_task["total_tokens"]
            }

            # Track total searches and execution time
            total_searches_performed += model_result["total_searches"]
            total_execution_time += model_result["execution_time_seconds"]

            # Track tool call statistics
            tool_history = model_result.get("tool_calls_log", [])
            total_tool_calls += len(tool_history)
            total_cart_tools += len(
                [t for t in tool_history if t.get("tool_type") == "cart"])
            total_checkout_tools += len(
                [t for t in tool_history if t.get("tool_type") == "checkout"])

            # Extract values from the result
            parsed_urls = model_result["parsed_urls"]
            url_ranks = model_result["url_ranks"]
            best_rank = model_result["best_rank"]
            avg_rank = model_result["avg_rank"]
            cart_checkout_urls = model_result.get("cart_checkout_urls", [])

            # Determine which URLs to use for evaluation based on task category
            if task_category in ["Add_To_Cart", "Checkout", "FindAndOrder"]:
                # For cart/checkout tasks, use URLs from cart/checkout operations
                evaluation_urls = [normalize_url(url)
                                   for url in cart_checkout_urls]
                print(
                    f"🛒 Using cart/checkout URLs for evaluation: {cart_checkout_urls}")
            else:
                # For search tasks, use the final answer
                evaluation_urls = [normalize_url(url.strip())
                                   for url in parsed_urls if url.strip().lower() != "done"]
                print(
                    f"🔍 Using final answer URLs for evaluation: {parsed_urls}")

            # Ranking metrics
            if best_rank is not None:
                total_rank_sum += best_rank
                total_ranked_results += 1

                if best_rank <= 3:
                    top3_count += 1
                if best_rank <= 10:
                    top10_count += 1

            # Calculate accuracy metrics using evaluation URLs
            expected_normalized = [normalize_url(url) for url in expected_flat]

            # Handle failed tasks with empty metrics
            if model_result.get("error_occurred", False):
                # For failed tasks, set all metrics to 0
                correct_model_answers = []
                additional_urls = []
                missing_urls = expected_flat  # All expected URLs are missing
                metrics = {
                    "task_completion_rate": 0,
                    "avg_precision": 0.0,
                    "avg_recall": 0.0,
                    "f1_score": 0.0
                }
                print(
                    f"📊 FAILED TASK METRICS: All metrics set to 0 for task {task['id']}")
            else:
                correct_model_answers = [
                    url for url in expected_flat if normalize_url(url) in evaluation_urls]
                additional_urls = [
                    url for url in evaluation_urls if url not in expected_normalized]
                missing_urls = [
                    url for url in expected_normalized if url not in evaluation_urls]

                metrics = calculation_results(expected_flat, evaluation_urls)

            print("Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.2f}" if isinstance(
                    v, float) else f"  {k}: {v}")

            # Save results with collected metrics
            task_result = {
                "task_id": task["id"],
                "user_task": user_task,
                "metrics": metrics,
                "parsed_urls": parsed_urls,
                "db_urls_found": urls_in_db,
                "db_urls_missing": urls_not_in_db,
                "db_coverage": model_result["db_coverage"],
                "tool_history": model_result["tool_calls_log"],
                "total_searches": model_result["total_searches"],
                "rag_exact_url_matches": model_result["rag_exact_url_matches"],
                "rag_total_matches": model_result["rag_total_matches"],
                "rag_coverage": model_result["rag_coverage"],

                # Ranking details
                "search_type": model_result["search_type"],
                "url_rank_details": url_ranks,
                "best_rank": best_rank,
                "avg_rank": avg_rank,
                "multi_search": True,

                # Execution time
                "execution_time_seconds": model_result["execution_time_seconds"],

                "correct_answers": expected_normalized,
                "correct_model_answers": correct_model_answers,
                "additional_urls": additional_urls,
                "missing_urls": missing_urls,
                "parsed_model_response": parsed_urls,
                "model_response": model_result["answer"],
                "task_category": task["id"].split("_Task")[0],
                "evaluation_urls": evaluation_urls,
                "cart_checkout_urls": cart_checkout_urls,
                "prompt_tokens": task_tokens["prompt_tokens"],
                "completion_tokens": task_tokens["completion_tokens"],
                "total_tokens": task_tokens["total_tokens"],
                "error_occurred": model_result.get("error_occurred", False),
                "error_message": model_result.get("error_message"),
                "error_type": model_result.get("error_type")
            }

            results.append(task_result)

            # Append per-task result to JSONL for crash-safe logging
            try:
                with incremental_jsonl_file.open("a", encoding="utf-8") as jf:
                    jf.write(json.dumps(task_result) + "\n")
            except Exception as e:
                print(f"⚠️  Failed to append JSONL for task {task['id']}: {e}")

            # Append a row to the streaming CSV
            try:
                row = {
                    "category": task.get("category", "Unknown"),
                    "task_id": task.get("id", ""),
                    "task_completion_rate": metrics.get("task_completion_rate", 0),
                    "avg_precision": metrics.get("avg_precision", 0.0),
                    "avg_recall": metrics.get("avg_recall", 0.0),
                    "f1_score": metrics.get("f1_score", 0.0),
                    "prompt_tokens": 0 if model_result.get("error_occurred", False) else task_tokens.get("prompt_tokens", 0),
                    "completion_tokens": 0 if model_result.get("error_occurred", False) else task_tokens.get("completion_tokens", 0),
                    "execution_duration": model_result.get("execution_time_seconds", 0)
                }
                with incremental_csv_file.open("a", newline="", encoding="utf-8") as csvfile:
                    fieldnames = [
                        "category",
                        "task_id",
                        "task_completion_rate",
                        "avg_precision",
                        "avg_recall",
                        "f1_score",
                        "prompt_tokens",
                        "completion_tokens",
                        "execution_duration"
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(row)
            except Exception as e:
                print(f"⚠️  Failed to append CSV for task {task['id']}: {e}")
            # break
    # Generate results file

    # Calculate execution time statistics
    avg_execution_time = total_execution_time / len(results) if results else 0
    min_execution_time = min(r["execution_time_seconds"]
                             for r in results) if results else 0
    max_execution_time = max(r["execution_time_seconds"]
                             for r in results) if results else 0

    # Enhanced benchmark summary with detailed metrics
    benchmark_summary = {
        "benchmark_metadata": {
            "timestamp": current_timestamp,
            "version": "langgraph",
            "model": model_name,
            "reasoning_effort": reasoning_effort,
            "results_directory": str(results_output_dir),
            "total_tasks": len(results),
            "total_searches_performed": total_searches_performed,
            "avg_searches_per_task": total_searches_performed / len(results) if results else 0,
            "total_tool_calls": total_tool_calls,
            "total_cart_tools": total_cart_tools,
            "total_checkout_tools": total_checkout_tools,
            "avg_tools_per_task": total_tool_calls / len(results) if results else 0,
            "token_usage": total_tokens_used,
            "execution_time_stats": {
                "total_seconds": total_execution_time,
                "average_seconds": avg_execution_time,
                "min_seconds": min_execution_time,
                "max_seconds": max_execution_time
            }
        },
        "performance_summary": {
            "ranking_metrics": {
                "total_tasks_with_results": total_ranked_results,
                "avg_best_rank": total_rank_sum / total_ranked_results if total_ranked_results > 0 else None,
                "top3_success_rate": top3_count / len(results) if results else 0,
                "top10_success_rate": top10_count / len(results) if results else 0,
                "top3_count": top3_count,
                "top10_count": top10_count
            }
        },
        "results_summary": {
            "total_tasks": len(results),
            "total_urls_not_in_db": total_urls_not_in_db,
            "total_expected_urls": total_expected_urls,
            "total_searches": total_searches_performed
        },
        "error_summary": {
            "failed_tasks": failed_tasks,
            "successful_tasks": len(results) - failed_tasks,
            "recursion_errors": recursion_errors,
            "other_errors": other_errors,
            "success_rate": (len(results) - failed_tasks) / len(results) if results else 0
        },
        "results": results
    }

    results_file = results_output_dir / \
        f"benchmark_results_{current_timestamp}.json"

    # Save results
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Generate compact CSV metrics using external calculation function
    csv_data = []

    # Calculate metrics for each individual task
    for result in results:
        # Get benchmark solution (expected URLs) for this task
        benchmark_solution = result.get("correct_answers", [])

        # Get model solution (evaluation URLs) for this task
        model_solution = result.get("evaluation_urls", [])
        error_occurred = result.get("error_occurred", False)

        if not error_occurred and benchmark_solution and model_solution:
            metrics = calculation_results(benchmark_solution, model_solution)
        else:
            metrics_source = result.get("metrics", {})
            metrics = {
                "task_completion_rate": metrics_source.get("task_completion_rate", 0),
                "avg_precision": metrics_source.get("avg_precision", 0.0),
                "avg_recall": metrics_source.get("avg_recall", 0.0),
                "f1_score": metrics_source.get("f1_score", 0.0)
            }

        csv_data.append({
            "category": result.get("task_category", "Unknown"),
            "task_id": result.get("task_id", ""),
            "task_completion_rate": metrics["task_completion_rate"],
            "avg_precision": metrics["avg_precision"],
            "avg_recall": metrics["avg_recall"],
            "f1_score": metrics["f1_score"],
            "prompt_tokens": 0 if error_occurred else result.get("prompt_tokens", 0),
            "completion_tokens": 0 if error_occurred else result.get("completion_tokens", 0),
            "execution_duration": result.get("execution_time_seconds", 0)
        })

    csv_file = results_output_dir / \
        f"benchmark_metrics_{current_timestamp}.csv"

    # Write CSV file
    with csv_file.open("w", newline="", encoding="utf-8") as csvfile:
        if csv_data:
            fieldnames = [
                "category",
                "task_id",
                "task_completion_rate",
                "avg_precision",
                "avg_recall",
                "f1_score",
                "prompt_tokens",
                "completion_tokens",
                "execution_duration"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

    print(f"CSV metrics saved to {csv_file}")

    # Print statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)
    print(f"🔍 Total searches performed: {total_searches_performed}")
    print(
        f"📊 Average searches per task: {total_searches_performed/len(results):.1f}")
    print(f"🛠️  Total tool calls: {total_tool_calls}")
    print(f"🛒 Total cart operations: {total_cart_tools}")
    print(f"💳 Total checkout operations: {total_checkout_tools}")
    print(f"⚙️  Average tools per task: {total_tool_calls/len(results):.1f}")

    # Error statistics
    success_rate = (len(results) - failed_tasks) / \
        len(results) if results else 0
    print(f"\n🚨 ERROR STATISTICS:")
    print(f"  - Total tasks: {len(results)}")
    print(f"  - Successful tasks: {len(results) - failed_tasks}")
    print(f"  - Failed tasks: {failed_tasks}")
    print(f"  - Success rate: {success_rate:.1%}")
    if failed_tasks > 0:
        print(f"  - Recursion errors: {recursion_errors}")
        print(f"  - Other errors: {other_errors}")


    # Print execution time statistics
    print(f"\n⏱️  EXECUTION TIME METRICS:")
    print(f"  - Total: {total_execution_time:.2f} seconds")
    print(f"  - Average per task: {avg_execution_time:.2f} seconds")
    print(f"  - Min: {min_execution_time:.2f} seconds")
    print(f"  - Max: {max_execution_time:.2f} seconds")

    # Print overall performance metrics
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE METRICS")
    print("=" * 60)

    # Token usage and cost summary
    print("\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY")
    print("=" * 60)
    print(f"📊 Embedding Tokens: {total_tokens_used['embedding_tokens']:,}")
    print(f"📊 Prompt Tokens: {total_tokens_used['prompt_tokens']:,}")
    print(f"📊 Completion Tokens: {total_tokens_used['completion_tokens']:,}")
    print(f"📊 Total Tokens Used: {total_tokens_used['total_tokens']:,}")


# Main execution
async def main():
    """Main function with proper cleanup"""
    try:
        # You can easily switch models here
        # Examples: "gpt-4", "gpt-3.5-turbo", "claude-3-opus-20240229", "claude-3-sonnet-20240229"

        model_name = "gpt-5-mini"

        chat_model = ChatOpenAI(model=model_name,  reasoning_effort="medium")

        await process_benchmark(model_name=model_name, chat_model=chat_model)
    finally:
        # Clean up Elasticsearch client
        await es_client.close()


if __name__ == "__main__":
    asyncio.run(main())
