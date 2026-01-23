import asyncio
import json
import time
import httpx
import os
import sys
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Import scoring utility
from utils import calculation_results

# Configuration Constants
REGISTRY_PATH = os.path.join(BASE_DIR, "src", "a2a", "registry.json")
TASK_FILE = os.path.join(BASE_DIR, "task_sets", "test_tasks_1.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "a2a", "scripted_buyer")
MODEL_NAME = "scripted-buyer-logic"

URLS = {
    "URL_1": "https://webmall-1.informatik.uni-mannheim.de",
    "URL_2": "https://webmall-2.informatik.uni-mannheim.de",
    "URL_3": "https://webmall-3.informatik.uni-mannheim.de",
    "URL_4": "https://webmall-4.informatik.uni-mannheim.de",
    "URL_5": "https://webmall-solution.informatik.uni-mannheim.de"
}

def fill_urls(text: str, urls: Dict[str, str]) -> str:
    """Replaces {{URL_X}} placeholders with actual webshop URLs."""
    for key, val in urls.items():
        text = text.replace("{{" + key + "}}", val)
    return text

def normalize_url(url: str) -> str:
    """Standardizes URLs."""
    if not url or not isinstance(url, str):
        return ""
    return url.rstrip('/').lower()

class A2ABenchmark:
    def __init__(self):
        """
        Initializes the Scripted Buyer without LLM or LangGraph.
        Focuses on A2A protocol compliance and efficient data aggregation.
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.endpoints = self._load_registry()
        self.results = []

    def _load_registry(self) -> List[str]:
        """Discovery: Load shop agent URLs from registry.json"""
        if not os.path.exists(REGISTRY_PATH):
            return [f"http://localhost:800{i+1}/messages" for i in range(4)]
        try:
            with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [shop['url'] for shop in data.get('shops', [])]
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return []

    def resolve_gt_urls(self, raw_urls: List[str]) -> List[str]:
        """Convert placeholder URLs to real URLs for scoring."""
        resolved = []
        for url in raw_urls:
            new_url = url
            for key, actual in URLS.items():
                new_url = new_url.replace("{{" + key + "}}", actual)
            resolved.append(new_url)
        return resolved

    async def _execute_shop_call(self, url: str, query: str) -> Dict:
        """Executes a JSON-RPC 2.0 call to a shop agent."""
        payload = {
            "jsonrpc": "2.0",
            "method": "ask_webmall",
            "params": {
                "query": query,
                "caller_identity": "Benchmark-Buyer-Script"
            },
            "id": str(int(time.time() * 1000))
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e)}

    def _extract_offers(self, shop_responses: List[Dict]) -> List[Dict]:
        """Parses Schema.org products and extracts prices for comparison."""
        all_offers = []
        for resp in shop_responses:
            res_data = resp.get("result", {})
            offers = res_data.get("offers", [])
            # In case the shop returns a list of URLs directly (old format fallback)
            if isinstance(offers, list):
                for item in offers:
                    if isinstance(item, dict):
                        all_offers.append(item)
                    elif isinstance(item, str):
                        # Handle case where only URL is returned (cannot compare price)
                        all_offers.append({"url": item, "offers": {"price": float('inf')}})
        return all_offers

    async def run_single_task(self, task: Dict):
        """Processes one task by broadcasting to all shops and selecting the cheapest."""
        task_id = task['id']
        query = fill_urls(task['task'], URLS)
        raw_gt_urls = self.resolve_gt_urls(task.get('correct_answer', {}).get('answers', []))
        gt_urls_normalized = [normalize_url(u) for u in raw_gt_urls]
        
        start_time = time.time()
        
        # 1. Broadcast: Call all shops concurrently
        tasks = [self._execute_shop_call(url, query) for url in self.endpoints]
        responses = await asyncio.gather(*tasks)
        
        # 2. Token & Offer Harvesting
        total_shop_prompt = 0
        total_shop_completion = 0
        all_offers = []
        
        for resp in responses:
            if "result" in resp:
                tokens = resp["result"].get("tokens", {})
                total_shop_prompt += tokens.get("prompt_tokens", 0)
                total_shop_completion += tokens.get("completion_tokens", 0)
                
                # Extract offers (supports both list of strings or list of dicts)
                offers = resp["result"].get("offers", [])
                for o in offers:
                    if isinstance(o, dict):
                        all_offers.append(o)
                    else:
                        # Fallback for simple URL list
                        all_offers.append({"url": o, "price": 999999})

        # 3. Decision Logic: Find cheapest (Min-selection)
        predicted_urls = []
        if all_offers:
            # Sort by price. Handle potential price type issues (string vs float)
            try:
                min_price = min(float(o.get('offers', {}).get('price', o.get('price', 999999))) for o in all_offers)
                predicted_urls = [
                    normalize_url(o['url']) 
                    for o in all_offers 
                    if float(o.get('offers', {}).get('price', o.get('price', 999999))) == min_price
                ]
            except (ValueError, TypeError):
                # If price is not a number, fallback to all returned URLs
                predicted_urls = [normalize_url(o['url']) for o in all_offers if 'url' in o]

        # 4. Metric Calculation
        metrics = calculation_results(gt_urls_normalized, predicted_urls)
        
        return {
            "summary": {
                "task_id": task_id,
                "task_completion_rate": metrics["task_completion_rate"],
                "f1_score": metrics["f1_score"],
                "shop_tokens": total_shop_prompt + total_shop_completion,
                "buyer_tokens": 0,  # Zero tokens used by the script
                "total_tokens": total_shop_prompt + total_shop_completion,
                "execution_time_seconds": time.time() - start_time
            },
            "detail": {
                "task_id": task_id,
                "wish": query,
                "ground_truth": raw_gt_urls,
                "predicted_urls": predicted_urls,
                "metrics": metrics
            }
        }

    async def run_benchmark(self):
        """
        Main execution loop for task sets. 
        Iterates through task sets, executes tasks, calculates metrics, and saves results.
        """
        if not os.path.exists(TASK_FILE):
            print(f"❌ Error: Task file not found at {TASK_FILE}")
            return

        with open(TASK_FILE, 'r', encoding='utf-8') as f:
            task_sets = json.load(f)

        total_tasks = sum(len(ts.get('tasks', [])) for ts in task_sets)
        print(f"📊 Starting A2A Benchmark for {total_tasks} tasks...")
        
        details_log = []
        self.results = []
        start_wall_time = time.time()

        # Iterate through each task set
        for task_set in task_sets:
            set_id = task_set.get('id', 'Unknown Set')
            print(f"\n📂 Processing Task Set: {set_id}")
            
            for task in task_set.get('tasks', []):
                # Execute single task and gather metrics
                res = await self.run_single_task(task)
                
                summary = res["summary"]
                self.results.append(summary)
                details_log.append(res["detail"])
                
                # Terminal output with CR, F1, and Token data
                print(f"Task: {summary['task_id']:<40} | "
                      f"CR: {summary['task_completion_rate']:.2f} | "
                      f"F1: {summary['f1_score']:.2f} | "
                      f"ShopTokens: {summary['shop_tokens']:<6}")

        # --- Calculate Overall Metrics ---
        total_processed = len(self.results)
        if total_processed == 0:
            return

        avg_cr = sum(r["task_completion_rate"] for r in self.results) / total_processed
        avg_f1 = sum(r["f1_score"] for r in self.results) / total_processed
        total_s_tokens = sum(r.get("shop_tokens", 0) for r in self.results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_summary_{timestamp}.json")
        debug_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_debug_{timestamp}.json")

        summary_payload = {
            "benchmark_metadata": {
                "timestamp": timestamp,
                "overall_metrics": {
                    "average_cr": avg_cr,
                    "average_f1": avg_f1,
                    "total_shop_tokens": total_s_tokens
                }
            },
            "results": self.results
        }
        
        # Save results to disk
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=4)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(details_log, f, indent=4)

        print("\n" + "="*60)
        print("🏆 FINAL BENCHMARK RESULTS")
        print(f"📈 Average CR:   {avg_cr:.4f}")
        print(f"🎯 Average F1:   {avg_f1:.4f}")
        print(f"💰 Shop Tokens:  {total_s_tokens}")
        print(f"⏱️ Total Time:   {time.time() - start_wall_time:.2f}s")
        print("="*60)

if __name__ == "__main__":
    asyncio.run(A2ABenchmark().run_benchmark())