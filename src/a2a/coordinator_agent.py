import time
import requests
from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import datetime

# Load environment variables from .env file
load_dotenv()

# Verify the key is loaded
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY not found in .env file")

# Initialize LLM (you can set OPENAI_API_KEY in your environment)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def decide_task_with_llm(user_input):
    """
    Use LLM to reason about the user's intent,
    decide the task, and produce a clear search query.
    Returns a tuple: (task_type, refined_query, reasoning)
    """
    prompt = f"""
    You are an A2A coordinator reasoning module.

    For the given user message:
    1. Briefly explain your reasoning (1-2 sentences).
    2. Decide the task type (search / add_to_cart / checkout).
    3. If applicable, produce a concise query (2–6 words).

    Format your output as JSON:
    {{
      "reasoning": "<short explanation>",
      "task": "search",
      "query": "wireless mouse"
    }}

    User message: "{user_input}"
    """

    response = llm.invoke(prompt)
    import json, datetime
    try:
        parsed = json.loads(response.content.strip())
        reasoning = parsed.get("reasoning", "")
        task = parsed.get("task", "search").lower()
        refined_query = parsed.get("query", user_input)
        if not refined_query:
            refined_query = "N/A"
            
    except Exception:
        reasoning = "(no reasoning provided)"
        task = "search"
        refined_query = user_input

    # === Log decision ===
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_input": user_input,
        "reasoning": reasoning,
        "task": task,
        "refined_query": refined_query
    }

    os.makedirs("results", exist_ok=True)

    # Generate filename by date, e.g. results/log_2025-10-25.jsonl
    date_str = datetime.date.today().isoformat()
    log_path = os.path.join("results", f"log_{date_str}.jsonl")

    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return task, refined_query, reasoning


BUYER_URLS = [
    "http://localhost:10001/a2a/sendMessage",
    "http://localhost:10002/a2a/sendMessage",
    "http://localhost:10003/a2a/sendMessage",
    "http://localhost:10004/a2a/sendMessage"
]


def handle_search(query):
    print(f"Performing search for: {query}")

    best_price = float('inf')
    best_item = None
    attempt = 1

    print(f"\nQuery: {query}\n")

    while attempt <= 2:
        print(f"=== Round {attempt} ===")
        found_any = False

        for url in BUYER_URLS:
            start_time = time.time()
            response = requests.post(url, json={"input": {"text": query}})
            elapsed = time.time() - start_time
            data = response.json()
            artifacts = data["output"]["artifacts"]

            print(f"--- {url} ---")
            print(f"Response time: {elapsed:.2f} sec")

            for item in artifacts:
                price_str = str(item.get("price", "")).replace(",", ".")
                name = item.get("name", "Unknown product")
                url_link = item.get("url", "")
                print(f"  {name} | {price_str} | {url_link}")

                try:
                    price = float(price_str)
                    if price > 0:
                        found_any = True
                        if price < best_price:
                            best_price = price
                            best_item = item
                except ValueError:
                    continue
            print()

        if found_any or attempt == 2:
            break
        else:
            print("No valid items found. Trying again with relaxed condition...\n")
            query = "Find me an octopus under 100 EUR"
            attempt += 1

    if best_item:
        print(f"\nCheapest item: {best_item['name']} | {best_item['price']} EUR | {best_item['url']}")
        return best_item
    else:
        print("No matching items found.")
        return None
    

def handle_add_to_cart(query):
    print(f"Adding product to cart based on query: {query}")

    best_item = handle_search(query)
    if not best_item:
        print("No item available to add to cart.")
        return

    # Determine which Buyer handled this product
    selected_buyer = None
    for url in BUYER_URLS:
        item_url = best_item.get('url', '')
        if item_url and "://" in item_url:
            shop_domain = item_url.split("/")[2]
        else:
            shop_domain = ""  # fallback for missing URLs

        # --- log missing URL for analysis ---
        log_path = "results/missing_url_items.log"
        os.makedirs("results", exist_ok=True)
        with open(log_path, "a") as log_file:
            log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Missing URL for item: {best_item.get('name', 'Unknown')} | Price: {best_item.get('price', 'N/A')}\n")
        print(f"(Warning) Missing URL detected for: {best_item.get('name', 'Unknown')}")

        if shop_domain in url:
            selected_buyer = url
            break

    if not selected_buyer:
        selected_buyer = BUYER_URLS[0]

    print(f"\nSending add-to-cart request to: {selected_buyer}")
    add_request = {"input": {"text": f"Add the cheapest {query} to cart"}}
    add_response = requests.post(selected_buyer, json=add_request)
    print("Add-to-cart response:", add_response.json()["output"]["text"])


def handle_checkout():
    print("Proceeding to checkout...")
    url = "http://localhost:10001/a2a/sendMessage"
    response = requests.post(url, json={"input": {"text": "checkout"}})
    print("Checkout response:", response.json()["output"]["text"])


def main():
    user_input = input("User: ")
    task_type, refined_query, reasoning = decide_task_with_llm(user_input)
    print(f"Reasoning: {reasoning}")
    print(f"Task decided: {task_type}")
    print(f"Refined query: {refined_query}\n")

    if task_type == "search":
        handle_search(refined_query)
    elif task_type == "add_to_cart":
        handle_add_to_cart(refined_query)
    elif task_type == "checkout":
        handle_checkout()
    else:
        print("Unrecognized task type. Defaulting to search.")
        handle_search(refined_query)


def summarize_llm_decisions(results_dir="results"):
    import json, os, datetime
    from collections import Counter

    today_file = os.path.join(results_dir, f"log_{datetime.date.today().isoformat()}.jsonl")
    all_tasks = []
    today_tasks = []

    # --- 累積統計 (全部檔案) ---
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith(".jsonl"):
                with open(os.path.join(results_dir, file), "r") as f:
                    for line in f:
                        try:
                            task = json.loads(line).get("task", "unknown")
                            all_tasks.append(task)
                            if file == os.path.basename(today_file):
                                today_tasks.append(task)
                        except Exception:
                            continue

    def show_summary(title, tasks):
        if not tasks:
            print(f"\n(No {title.lower()} records found.)")
            return
        counts = Counter(tasks)
        total = sum(counts.values())
        print(f"\n=== {title} Decision Summary ===")
        for task, n in counts.items():
            print(f"{task:12s}: {n} ({n/total*100:.1f}%)")
        print("=" * 33)

    show_summary("Today's LLM", today_tasks)
    show_summary("Overall LLM", all_tasks)


if __name__ == "__main__":
    main()
    summarize_llm_decisions()
