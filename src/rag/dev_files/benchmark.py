import os
import json
from openai import OpenAI, AsyncOpenAI
from typing import Dict, List
# from supabase import create_client, Client  # Commented out for Elasticsearch migration
from elasticsearch_client import ElasticsearchRAGClient
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# 1) Configuration: your real shop URLs and solution page URL
URLS = {
    "URL_1": "https://webmall-1.informatik.uni-mannheim.de",
    "URL_2": "https://webmall-2.informatik.uni-mannheim.de",
    "URL_3": "https://webmall-3.informatik.uni-mannheim.de",
    "URL_4": "https://webmall-4.informatik.uni-mannheim.de",
    "URL_5": "https://webmall-solution.informatik.uni-mannheim.de"
}

# 2) Initialize OpenAI and Supabase clients
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

client = OpenAI(api_key=api_key)
async_client = AsyncOpenAI(api_key=api_key)

# Initialize Elasticsearch for RAG
es_client = ElasticsearchRAGClient()


def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing trailing slashes and converting to lowercase."""
    return url.rstrip('/').lower()


def fill_urls(text: str, urls: Dict[str, str]) -> str:
    for key, val in urls.items():
        text = text.replace("{{" + key + "}}", val)
    return text


async def get_embedding(text: str) -> tuple[List[float], int]:
    """Get embedding vector from OpenAI and return tokens used."""
    try:
        response = await async_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        tokens_used = response.usage.total_tokens
        return response.data[0].embedding, tokens_used
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536, 0  # Return zero vector and zero tokens on error


async def retrieve_relevant_documentation(user_query: str, match_count: int = 10, token_tracker: Dict = None) -> List[Dict]:
    """
    Enhanced RAG retrieval with hybrid search and reranking.
    """
    try:
        print(f"🔍 ENHANCED RAG RETRIEVAL for query: {user_query[:100]}...")

        # Let the LLM generate a query
        llm_query = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": """You are a RAG-capable agent that can browse four webshops and find product offers. Use the provided knowledge base information to answer accurately. Only output the final answer. Given the user query, generate a query that will be used to search the knowledge base. Its a semantic search, so the query should be a single sentence that captures the user's intent. Try to be as specific as possible and dont add details that are not in the user query. Optimise your query in such a way that it works best with elasticsearch and openai embeddings."""},
                {"role": "user", "content": user_query}
            ]
        )

        # Track LLM query generation tokens
        if token_tracker and hasattr(llm_query, 'usage'):
            token_tracker["prompt_tokens"] += llm_query.usage.prompt_tokens
            token_tracker["completion_tokens"] += llm_query.usage.completion_tokens
            token_tracker["total_tokens"] += llm_query.usage.total_tokens

        print(f"🔍 LLM QUERY: {llm_query.choices[0].message.content}")

        semantic_task = semantic_search(
            llm_query.choices[0].message.content, match_count, token_tracker)

        results = []
        semantic_results = await semantic_task
        for doc in semantic_results:
            doc['search_type'] = 'semantic'
            doc['boost'] = 1.0
        results.extend(semantic_results)

        deduplicated = deduplicate_results(results)

        final_results = deduplicated[:match_count]
        # print(
        #    f"✅ RAG results: {len(final_results)} after deduplication")

        return final_results, llm_query.choices[0].message.content

    except Exception as e:
        print(f"Error in enhanced RAG retrieval: {e}")
        return []


async def semantic_search(query: str, match_count: int, token_tracker: Dict = None) -> List[Dict]:
    """Standard semantic search using Elasticsearch."""

    print(f"🔍 SEMANTIC SEARCH for query: {query}...")
    try:
        query_embedding, tokens_used = await get_embedding(query)

        # Track embedding tokens
        if token_tracker:
            token_tracker["embedding_tokens"] += tokens_used

        results = await es_client.semantic_search(query_embedding, match_count)

        print(f"The Sematic search found {len(results)} results")

        return results
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []


def deduplicate_results(results: List[Dict]) -> List[Dict]:
    """Remove duplicates and merge scores."""
    seen_urls = {}

    for doc in results:
        url_chunk_key = f"{doc['url']}_{doc.get('chunk_number', 0)}"

        if url_chunk_key in seen_urls:
            existing = seen_urls[url_chunk_key]
            existing['similarity'] = max(
                existing.get('similarity', 0),
                doc.get('similarity', 0)
            )
            existing['boost'] = max(
                existing.get('boost', 1.0),
                doc.get('boost', 1.0)
            )
        else:
            seen_urls[url_chunk_key] = doc

    return list(seen_urls.values())


def parse_model_answer(answer: str) -> str:
    """Parse the model answer and return the correct answer."""
    # Parse JSON response
    try:
        # Try to extract JSON array from the response
        if answer.startswith('[') and answer.endswith(']'):
            return json.loads(answer)
        else:
            # If response contains JSON within text, try to find it
            import re
            json_match = re.search(r'\[.*\]', answer, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: treat as single item or "Done"
                return [answer] if answer.lower() != "done" else [
                    "Done"]
    except json.JSONDecodeError:
        print("Warning: Could not parse JSON response, treating as plain text")
        # Fallback to original ### splitting method
        return [part.strip() for part in answer.split("###")]


async def get_model_answer(user_task: str, urls_in_db: List[str], expected_flat: List[str], total_tokens_used: Dict, n_docs: int = 10) -> str:
    print("🔍 RETRIEVING RAG INFORMATION...")
    rag_docs, rag_query = await retrieve_relevant_documentation(user_task, match_count=n_docs, token_tracker=total_tokens_used)

    print(f"The user is asking for: {user_task}")

    print(f"📋 RETRIEVED {len(rag_docs)} RELEVANT DOCUMENTS:")
    rag_context = ""
    if rag_docs:
        for i, doc in enumerate(rag_docs, 1):
            rag_context += f"\n--- Document {i} ---\n"
            rag_context += f"Title: {doc['title']}\n"
            rag_context += f"URL: {doc['url']}\n"
            rag_context += f"Content: {doc['content']}...\n"
    else:
        print("  No relevant documents found in RAG system")
        rag_context = "No relevant product information found in the knowledge base."

    db_coverage = len(urls_in_db) / \
        len(expected_flat) if expected_flat else 0

    # Now check if correct answers are in the retrieved documents
    rag_urls = [doc['url'] for doc in rag_docs]

    # Check for exact URL matches (with normalization)
    exact_url_matches = []
    rag_urls_normalized = [normalize_url(url) for url in rag_urls]
    expected_normalized = [normalize_url(url) for url in expected_flat]

    for expected_url in expected_flat:
        expected_norm = normalize_url(expected_url)
        if expected_norm in rag_urls_normalized:
            exact_url_matches.append(expected_url)

    total_rag_matches = len(exact_url_matches)
    rag_coverage = total_rag_matches / \
        len(expected_flat) if expected_flat else 0

    if rag_coverage < 1.0:
        print(
            f"  📊 RAG Coverage: {total_rag_matches}/{len(expected_flat)} ({rag_coverage:.1%}) of correct answers found in retrieved documents")

    # Only print detailed RAG results if coverage is not 100%
    """
    if rag_coverage < 1.0:
        # Print semantic search results for debugging
        print(
            f"🔍 SEMANTIC SEARCH RESULTS (showing URLs from retrieved documents):")
        for doc in rag_docs:
            print(f"🔍 SEMANTIC SEARCH RESULT: {doc['url']}")
    """

    if total_rag_matches == 0:
        print(
            "  ⚠️  WARNING: None of the correct answers were found in retrieved RAG documents!")
    elif rag_coverage < 1.0:
        print(
            f"  ⚠️  WARNING: Only {rag_coverage:.1%} of correct answers were found in RAG documents")

    # Get normalized versions of URLs found in DB
    urls_in_db_normalized = [
        normalize_url(url) for url in urls_in_db]

    # Get normalized versions of URLs found in RAG
    rag_found_normalized = []
    for url in exact_url_matches:
        rag_found_normalized.append(normalize_url(url))

    # Find intersection - URLs that are both in DB and retrieved by RAG
    db_and_rag_intersection = []
    db_not_retrieved = []

    for db_url in urls_in_db_normalized:
        if db_url in rag_found_normalized:
            db_and_rag_intersection.append(db_url)
        else:
            db_not_retrieved.append(db_url)

    # Calculate retrieval effectiveness
    retrieval_effectiveness = len(
        db_and_rag_intersection) / len(urls_in_db) if urls_in_db else 0

    prompt = f"\n\nRelevant Information from Knowledge Base:{'\n'*3}" + rag_context + "\n\n The user is asking for the following query: " + \
        user_task + \
        "\n\n Reply with a JSON array containing the EXACT URLs. Format: [\"url1\", \"url2\", ...] or [\"Done\"] if no products found."

    # call OpenAI chat
    response = client.chat.completions.create(
        model="gpt-4.1",  # or whichever model you prefer
        messages=[
            {"role": "system", "content": "You are a RAG-capable agent that can browse four webshops and find product offers. Use the provided knowledge base information to answer accurately. Only output the final answer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    # Track final answer generation tokens
    if hasattr(response, 'usage'):
        total_tokens_used["prompt_tokens"] += response.usage.prompt_tokens
        total_tokens_used["completion_tokens"] += response.usage.completion_tokens
        total_tokens_used["total_tokens"] += response.usage.total_tokens

    answer = response.choices[0].message.content.strip()
    parsed_urls = parse_model_answer(answer)

    # Return all the calculated metrics for JSON saving
    return {
        "parsed_urls": parsed_urls,
        "answer": answer,
        "rag_query": rag_query,
        "rag_docs": rag_docs,
        "rag_exact_url_matches": exact_url_matches,
        "rag_total_matches": total_rag_matches,
        "rag_coverage": rag_coverage,
        "db_and_rag_intersection": db_and_rag_intersection,
        "db_not_retrieved": db_not_retrieved,
        "retrieval_effectiveness": retrieval_effectiveness,
        "db_coverage": db_coverage,
        "total_urls_in_db_not_retrieved": len(db_not_retrieved),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }


# 4) Load your benchmark JSON file
BENCHMARK_JSON_PATH = "/Users/aaronsteiner/Documents/GitHub/webmall-alternative-interfaces/task_sets.json"  # adjust as needed

with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
    benchmark = json.load(f)


async def process_benchmark():
    """Process benchmark tasks with RAG integration"""
    # 5) Initialize results list
    results = []

    # Initialize counters for overall statistics
    total_urls_not_in_db = 0
    total_urls_in_db_not_retrieved = 0
    total_expected_urls = 0

    # Initialize token tracking
    total_tokens_used = {
        "embedding_tokens": 0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0
    }

    # 5) Iterate over tasks, build prompts, call the model, compare to correct answer
    for task_set in benchmark:
        for index, task in enumerate(task_set["tasks"]):
            # if task["id"] != "Webmall_Cheapest_Product_Search_Task9":
            #    continue
            print(f"\n=== TASK {task['id']} ===")

            # First, check if correct answers exist in the database at all
            correct_answer = task.get("correct_answer", {}).get("answers", [])
            expected_flat = [fill_urls(x, URLS) for x in correct_answer]

            # Check database for exact URL matches (with normalization)
            urls_in_db = []
            urls_not_in_db = []

            for expected_url in expected_flat:
                if not expected_url.endswith("/"):
                    expected_url += "/"
                if await es_client.check_url_exists(expected_url):
                    urls_in_db.append(expected_url)
                else:
                    urls_not_in_db.append(expected_url)

            total_urls_not_in_db = len(urls_not_in_db)

            if len(urls_not_in_db) > 0:
                print(
                    f"  ⚠️  WARNING: {len(urls_not_in_db)} correct answer(s) not found in database!")
                for url in urls_not_in_db:
                    print(f"    - {url}")

            # Update total counter
            total_expected_urls += len(expected_flat)

            # fill the instruction template
            instruction = fill_urls(task["instruction"], URLS)

            user_task = task["task"] if "task" in task else None
            if not user_task:
                # your JSON nests <task> inside the "instruction" string
                # rough extraction:
                start = task["instruction"].find("<task>")
                end = task["instruction"].find("</task>") + len("</task>")
                user_task = task["instruction"][start:end]

            # fill the url in the task
            user_task = fill_urls(user_task, URLS)
            user_task = user_task.replace("<task>", "").replace("</task>", "")

            rag_fetches = 0
            model_result = await get_model_answer(user_task, urls_in_db, expected_flat, total_tokens_used, n_docs=30)

            rag_fetches += 1
            """
            if model_result["parsed_urls"] != ["Done"] and len(model_result["parsed_urls"]) >= 2:
                print("Refetching with 30 docs")
                model_result = await get_model_answer(user_task, urls_in_db, expected_flat, total_tokens_used, n_docs=30)
                rag_fetches += 1
            if model_result["parsed_urls"] != ["Done"] and len(model_result["parsed_urls"]) > 5:
                print("Refetching with 50 docs")
                model_result = await get_model_answer(user_task, urls_in_db, expected_flat, total_tokens_used, n_docs=50)
                rag_fetches += 1
            
            # if the model did not find any urls, we need to refetch with 70 docs
            if model_result["parsed_urls"] == ["Done"] or len(model_result["parsed_urls"]) == 0:
                print("Refetching with 70 docs")
                model_result = await get_model_answer(user_task, urls_in_db, expected_flat, total_tokens_used, n_docs=70)
                rag_fetches += 1
            """

            total_urls_in_db_not_retrieved = model_result["total_urls_in_db_not_retrieved"]

            # Extract values from the result dictionary
            parsed_urls = model_result["parsed_urls"]
            answer = model_result["answer"]
            rag_docs = model_result["rag_docs"]
            exact_url_matches = model_result["rag_exact_url_matches"]
            total_rag_matches = model_result["rag_total_matches"]
            rag_coverage = model_result["rag_coverage"]
            db_and_rag_intersection = model_result["db_and_rag_intersection"]
            db_not_retrieved = model_result["db_not_retrieved"]
            retrieval_effectiveness = model_result["retrieval_effectiveness"]
            db_coverage = model_result["db_coverage"]

            # 7) Enhanced comparison check with URL normalization
            got = [url.strip()
                   for url in parsed_urls if url.strip().lower() != "done"]

            # Normalize both got and expected URLs
            got_normalized = [normalize_url(url) for url in got]
            expected_normalized = [normalize_url(url) for url in expected_flat]

            # correct answer urls
            correct_model_answers = [
                url for url in expected_flat if url in got_normalized]

            # additional urls that are not in the correct answers
            additional_urls = [
                url for url in got_normalized if url not in expected_normalized]

            # urls that are in the correct answers but not in the model response
            missing_urls = [
                url for url in expected_normalized if url not in got_normalized]

            # 8) Save the results to a JSON file
            results.append({
                "task_id": task["id"],
                "user_task": user_task,
                "parsed_urls": got,
                "db_urls_found": urls_in_db,
                "db_urls_missing": urls_not_in_db,
                "db_coverage": db_coverage,
                "rag_query": model_result["rag_query"],
                "rag_docs_retrieved": len(rag_docs),
                "rag_docs": [{"title": doc["title"], "url": doc["url"], "similarity": doc.get("similarity", 0)} for doc in rag_docs],
                "rag_exact_url_matches": exact_url_matches,
                "rag_total_matches": total_rag_matches,
                "rag_coverage": rag_coverage,
                "rag_fetches": rag_fetches,

                "db_and_rag_intersection": db_and_rag_intersection,
                "db_not_retrieved": db_not_retrieved,
                "retrieval_effectiveness": retrieval_effectiveness,
                "correct_answers": expected_normalized,
                "correct_model_answers": correct_model_answers,
                "additional_urls": additional_urls,
                "missing_urls": missing_urls,
                "parsed_model_response": parsed_urls,
                "model_response": answer,
                "accuracy": len(correct_model_answers) / max(len(expected_flat), 1),
                "recall": len(correct_model_answers) / max(len(expected_flat), 1),
                "precision": len(correct_model_answers) / max(len(got_normalized), 1) if got_normalized else 0,
                "prompt_tokens": model_result["prompt_tokens"],
                "completion_tokens": model_result["completion_tokens"],
                "total_tokens": model_result["total_tokens"]
            })

    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add overall token usage summary to results
    benchmark_summary = {
        "benchmark_metadata": {
            "timestamp": current_timestamp,
            "total_tasks": len(results),
            "token_usage": total_tokens_used
        },
        "results_summary": {
            "total_tasks": len(results),
            "total_urls_not_in_db": total_urls_not_in_db,
            "total_urls_in_db_not_retrieved": total_urls_in_db_not_retrieved,
            "total_expected_urls": total_expected_urls
        },
        "results": results
    }

    file_name = f"results/benchmark_results_gpt-4.1_{current_timestamp}.json"
    # 9) Save the results to a JSON file
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

        print(
            f"\nResults saved to {file_name}")

    # Print overall database and retrieval statistics
    print("\n" + "="*60)
    print("OVERALL DATABASE & RETRIEVAL STATISTICS")
    print("="*60)
    print(f"📊 Total Expected URLs across all tasks: {total_expected_urls}")
    print(
        f"❌ Total URLs NOT in Database: {total_urls_not_in_db} ({total_urls_not_in_db/total_expected_urls*100:.1f}% of expected)")
    print(
        f"🔄 Total URLs in DB but NOT Retrieved by RAG: {total_urls_in_db_not_retrieved}")

    # Calculate database coverage
    total_urls_in_db = total_expected_urls - total_urls_not_in_db
    if total_expected_urls > 0:
        db_coverage_overall = total_urls_in_db / total_expected_urls
        print(
            f"✅ Total URLs found in Database: {total_urls_in_db} ({db_coverage_overall:.1%} coverage)")

    if total_urls_in_db > 0:
        retrieval_success_rate = (
            total_urls_in_db - total_urls_in_db_not_retrieved) / total_urls_in_db
        print(
            f"🎯 RAG Retrieval Success Rate: {retrieval_success_rate:.1%} (of URLs in DB)")

    print("\n" + "="*60)
    print("OVERALL PERFORMANCE METRICS")
    print("="*60)

    # compute overall accuracy, recall, precision
    overall_accuracy = sum(result["accuracy"]
                           for result in results) / len(results)
    overall_recall = sum(result["recall"] for result in results) / len(results)
    overall_precision = sum(result["precision"]
                            for result in results) / len(results)
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Overall Recall: {overall_recall:.2%}")
    print(f"Overall Precision: {overall_precision:.2%}")

    # Token usage summary
    print("\n" + "="*60)
    print("TOKEN USAGE SUMMARY")
    print("="*60)
    print(f"📊 Embedding Tokens: {total_tokens_used['embedding_tokens']:,}")
    print(f"📊 Prompt Tokens: {total_tokens_used['prompt_tokens']:,}")
    print(f"📊 Completion Tokens: {total_tokens_used['completion_tokens']:,}")
    print(f"📊 Total Tokens Used: {total_tokens_used['total_tokens']:,}")

    # Calculate approximate costs (using OpenAI pricing as of 2024)
    # GPT-4.1-mini: $0.15/1M input tokens, $0.6/1M output tokens
    # text-embedding-3-small: $0.02/1M tokens
    embedding_cost = (total_tokens_used['embedding_tokens'] / 1_000_000) * 0.02
    prompt_cost = (total_tokens_used['prompt_tokens'] / 1_000_000) * 0.15
    completion_cost = (
        total_tokens_used['completion_tokens'] / 1_000_000) * 0.60
    total_cost = embedding_cost + prompt_cost + completion_cost

    print(f"\n💰 ESTIMATED COSTS (USD):")
    print(f"   Embeddings: ${embedding_cost:.4f}")
    print(f"   Prompts: ${prompt_cost:.4f}")
    print(f"   Completions: ${completion_cost:.4f}")
    print(f"   Total Cost: ${total_cost:.4f}")

    # Performance per task
    num_tasks = len(results)
    if num_tasks > 0:
        avg_tokens_per_task = total_tokens_used['total_tokens'] / num_tasks
        avg_cost_per_task = total_cost / num_tasks
        print(f"\n📈 PER TASK AVERAGES:")
        print(f"   Tokens per task: {avg_tokens_per_task:.0f}")
        print(f"   Cost per task: ${avg_cost_per_task:.4f}")


# Run the async benchmark
async def main():
    """Main function with proper cleanup"""
    try:
        await process_benchmark()
    finally:
        # Clean up Elasticsearch client
        await es_client.close()

if __name__ == "__main__":
    asyncio.run(main())
