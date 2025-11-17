from rag.elasticsearch_client import ElasticsearchRAGClient
import os
import sys
import json
from openai import OpenAI, AsyncOpenAI
from typing import Dict, List
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


load_dotenv()

# Configuration: webmall URLs
URLS = {
    "URL_1": "https://webmall-1.informatik.uni-mannheim.de",
    "URL_2": "https://webmall-2.informatik.uni-mannheim.de",
    "URL_3": "https://webmall-3.informatik.uni-mannheim.de",
    "URL_4": "https://webmall-4.informatik.uni-mannheim.de",
    "URL_5": "https://webmall-solution.informatik.uni-mannheim.de"
}

# Initialize OpenAI and Elasticsearch clients
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable")

client = OpenAI(api_key=api_key)
async_client = AsyncOpenAI(api_key=api_key)

# Initialize Elasticsearch V2 client
es_client = ElasticsearchRAGClient()


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


async def retrieve_relevant_documentation_v2(user_query: str, match_count: int = 10,
                                             token_tracker: Dict = None, use_hybrid: bool = True) -> List[Dict]:
    """
    Enhanced RAG retrieval with V2 weighted embeddings and hybrid search.
    """
    try:
        print(f"🔍 V2 SEARCH for query: {user_query[:100]}...")

        # Get query embedding
        query_embedding, tokens_used = await get_embedding(user_query)

        # Track embedding tokens
        if token_tracker:
            token_tracker["embedding_tokens"] += tokens_used

        # Choose search method
        if use_hybrid:
            print("🔄 Using hybrid search (semantic + text matching)")
            results = await es_client.hybrid_search(user_query, query_embedding, match_count)
        else:
            print("🧠 Using pure semantic search")
            results = await es_client.semantic_search(query_embedding, match_count)

        print(f"✅ V2 search found {len(results)} results")

        return results

    except Exception as e:
        print(f"Error in V2 RAG retrieval: {e}")
        return []


def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing trailing slashes and converting to lowercase."""
    return url.rstrip('/').lower()


def fill_urls(text: str, urls: Dict[str, str]) -> str:
    """Replace URL placeholders with actual URLs."""
    for key, val in urls.items():
        text = text.replace("{{" + key + "}}", val)
    return text


async def interactive_query_testing_v2():
    """Interactive query testing with V2 weighted embeddings system."""
    # Load benchmark JSON file
    BENCHMARK_JSON_PATH = "/Users/aaronsteiner/Documents/GitHub/webmall-alternative-interfaces/task_sets.json"

    try:
        with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
    except FileNotFoundError:
        print(f"❌ Benchmark file not found: {BENCHMARK_JSON_PATH}")
        print("Please ensure the task_sets.json file exists in the project root.")
        return

    print("\\n" + "=" * 60)
    print("INTERACTIVE QUERY TESTING FOR WEBMALL TASKS - V2 WEIGHTED EMBEDDINGS")
    print("=" * 60)
    print("\\n✨ Features:")
    print("  - Weighted embeddings (Title: 50%, Summary: 30%, Content: 20%)")
    print("  - Hybrid search with text boosting")
    print("  - Enhanced ranking for product searches")
    print("\\nCommands: 'skip' to skip a task, 'quit' to exit, 'hybrid'/'semantic' to switch modes\\n")

    # Initialize counters
    total_tasks = 0
    tasks_with_full_coverage = 0
    total_queries = 0
    search_mode = "hybrid"  # Default to hybrid search

    # Performance tracking
    rank_improvements = []

    # Iterate over tasks
    for task_set in benchmark:
        for task in task_set["tasks"]:
            if "Webmall_Find_Compatible_Products" not in task["id"]:
                continue
            total_tasks += 1
            print(f"\\n" + "=" * 60)
            print(f"TASK: {task['id']} (Mode: {search_mode.upper()})")
            print("=" * 60)

            # Get correct answers
            correct_answer = task.get("correct_answer", {}).get("answers", [])
            expected_urls = [fill_urls(x, URLS) for x in correct_answer]
            expected_normalized = [normalize_url(url) for url in expected_urls]

            # Display task description
            user_task = task["task"] if "task" in task else None
            if not user_task:
                start = task["instruction"].find("<task>")
                end = task["instruction"].find("</task>") + len("</task>")
                user_task = task["instruction"][start:end]

            user_task = fill_urls(user_task, URLS)
            user_task = user_task.replace("<task>", "").replace("</task>", "")

            print(f"\\nTask Description: {user_task}")
            print(f"Expected URLs ({len(expected_urls)}): ")
            for url in expected_urls:
                print(f"  - {url}")

            # Check which expected URLs exist in database
            print("\\nChecking V2 database coverage...")
            urls_in_db = []
            urls_not_in_db = []

            for expected_url in expected_urls:
                if not expected_url.endswith("/"):
                    expected_url += "/"
                if await es_client.check_url_exists(expected_url):
                    urls_in_db.append(expected_url)
                else:
                    urls_not_in_db.append(expected_url)

            db_coverage = len(urls_in_db) / \
                len(expected_urls) if expected_urls else 0
            print(
                f"\\n📊 V2 Database Coverage: {len(urls_in_db)}/{len(expected_urls)} ({db_coverage:.1%})")

            if urls_not_in_db:
                print("❌ URLs NOT in V2 database:")
                for url in urls_not_in_db:
                    print(f"   - {url}")

            # Get user query
            while True:
                user_query = input(
                    f"\\n🔍 Enter your search query (or 'skip'/'quit'/'hybrid'/'semantic'): ").strip()

                if user_query.lower() == 'quit':
                    print("\\nExiting...")
                    print_final_summary(
                        total_tasks, tasks_with_full_coverage, total_queries, rank_improvements)
                    return

                if user_query.lower() == 'skip':
                    print("Skipping task...")
                    break

                if user_query.lower() == 'hybrid':
                    search_mode = "hybrid"
                    print("🔄 Switched to hybrid search mode")
                    continue

                if user_query.lower() == 'semantic':
                    search_mode = "semantic"
                    print("🧠 Switched to semantic search mode")
                    continue

                if not user_query:
                    print("Please enter a query.")
                    continue

                total_queries += 1

                # Perform V2 RAG search
                print(f"\\nSearching for: {user_query}")

                # Track tokens
                token_tracker = {"embedding_tokens": 0}

                # Adjust match_count as needed
                match_count = 30
                use_hybrid = (search_mode == "hybrid")
                rag_results = await retrieve_relevant_documentation_v2(
                    user_query, match_count, token_tracker, use_hybrid
                )

                print(f"\\n📋 Retrieved {len(rag_results)} documents")

                # Check if correct answers are in RAG results
                rag_urls = [doc['url'] for doc in rag_results]
                rag_urls_normalized = [normalize_url(url) for url in rag_urls]

                # Find matches and their ranks
                exact_url_matches = []
                url_ranks = {}

                for expected_url in expected_urls:
                    expected_norm = normalize_url(expected_url)
                    # Find the rank (position) of this URL in the results
                    for rank, rag_url_norm in enumerate(rag_urls_normalized, 1):
                        if expected_norm == rag_url_norm:
                            exact_url_matches.append(expected_url)
                            url_ranks[expected_url] = rank
                            break

                rag_coverage = len(exact_url_matches) / \
                    len(expected_urls) if expected_urls else 0

                # Show top 10 retrieved documents first
                print(
                    f"\\n🔝 Top {min(10, len(rag_results))} Retrieved Documents:")
                for i, doc in enumerate(rag_results[:10], 1):
                    # Check if this document is a correct answer
                    is_correct = normalize_url(
                        doc['url']) in expected_normalized
                    marker = "✅" if is_correct else "  "

                    print(f"\\n{marker} {i}. {doc['title']}")
                    print(f"   URL: {doc['url']}")

                    # Show score or similarity depending on search mode
                    if 'score' in doc:
                        print(f"   Score: {doc.get('score', 0):.3f}")
                    else:
                        print(f"   Similarity: {doc.get('similarity', 0):.3f}")

                    print(f"   Content: {doc['content'][:200]}...")

                # Display coverage and rank results after the documents
                print(
                    f"\\n✅ V2 RAG Coverage: {len(exact_url_matches)}/{len(expected_urls)} ({rag_coverage:.1%})")

                if exact_url_matches:
                    print("\\n✅ Found correct URLs with ranks:")
                    # Sort by rank for better readability
                    sorted_matches = sorted(
                        exact_url_matches, key=lambda url: url_ranks[url])
                    for url in sorted_matches:
                        print(f"   - Rank #{url_ranks[url]}: {url}")

                    # Display rank statistics
                    ranks = list(url_ranks.values())
                    avg_rank = sum(ranks) / len(ranks)
                    best_rank = min(ranks)
                    worst_rank = max(ranks)

                    print(f"\\n📊 V2 Rank Statistics:")
                    print(f"   - Best rank: #{best_rank}")
                    print(f"   - Worst rank: #{worst_rank}")
                    print(f"   - Average rank: {avg_rank:.1f}")

                    # Track rank improvements (assuming V1 would be much worse)
                    rank_improvements.append({
                        'task_id': task['id'],
                        'query': user_query,
                        'best_rank': best_rank,
                        'avg_rank': avg_rank,
                        'coverage': rag_coverage
                    })

                    # Performance assessment
                    if best_rank <= 3:
                        print("   🎯 EXCELLENT: Top 3 ranking!")
                    elif best_rank <= 10:
                        print("   👍 GOOD: Top 10 ranking")
                    elif best_rank <= 20:
                        print("   ⚠️  FAIR: Needs improvement")
                    else:
                        print("   ❌ POOR: Significant ranking issues")

                missing_urls = [url for url in expected_normalized if normalize_url(
                    url) not in rag_urls_normalized]
                if missing_urls:
                    print("\\n❌ Missing URLs:")
                    for url in missing_urls:
                        print(f"   - {url}")

                if rag_coverage == 1.0:
                    print("\\n🎉 Perfect coverage! All expected URLs found.")
                    tasks_with_full_coverage += 1

                # Token usage info
                print(f"\\n🔋 Tokens used: {token_tracker['embedding_tokens']}")

                # Ask if user wants to try another query
                retry = input(
                    "\\nTry another query for this task? (y/n): ").strip().lower()
                if retry != 'y':
                    break

    print_final_summary(total_tasks, tasks_with_full_coverage,
                        total_queries, rank_improvements)


def print_final_summary(total_tasks, tasks_with_full_coverage, total_queries, rank_improvements):
    """Print comprehensive testing summary."""
    print("\\n" + "=" * 60)
    print("V2 TESTING SUMMARY")
    print("=" * 60)
    print(f"Total tasks tested: {total_tasks}")
    print(f"Total queries executed: {total_queries}")
    print(f"Tasks with full RAG coverage: {tasks_with_full_coverage}")
    print(
        f"Overall success rate: {tasks_with_full_coverage/total_tasks:.1%}" if total_tasks > 0 else "N/A")

    if rank_improvements:
        print("\\n📈 RANKING PERFORMANCE:")

        # Calculate statistics
        all_best_ranks = [item['best_rank'] for item in rank_improvements]
        all_avg_ranks = [item['avg_rank'] for item in rank_improvements]
        all_coverages = [item['coverage'] for item in rank_improvements]

        print(
            f"   - Queries with Top 3 results: {sum(1 for r in all_best_ranks if r <= 3)}/{len(all_best_ranks)} ({sum(1 for r in all_best_ranks if r <= 3)/len(all_best_ranks):.1%})")
        print(
            f"   - Queries with Top 10 results: {sum(1 for r in all_best_ranks if r <= 10)}/{len(all_best_ranks)} ({sum(1 for r in all_best_ranks if r <= 10)/len(all_best_ranks):.1%})")
        print(
            f"   - Average best rank: {sum(all_best_ranks)/len(all_best_ranks):.1f}")
        print(
            f"   - Average coverage: {sum(all_coverages)/len(all_coverages):.1%}")

        print("\\n🏆 BEST PERFORMING QUERIES:")
        # Sort by best rank and show top 5
        sorted_improvements = sorted(
            rank_improvements, key=lambda x: x['best_rank'])
        for i, item in enumerate(sorted_improvements[:5], 1):
            print(
                f"   {i}. Rank #{item['best_rank']}: \"{item['query']}\" ({item['task_id']})")


async def main():
    """Main function with proper cleanup"""
    try:
        await interactive_query_testing_v2()
    finally:
        # Clean up Elasticsearch client
        await es_client.close()


if __name__ == "__main__":
    asyncio.run(main())
