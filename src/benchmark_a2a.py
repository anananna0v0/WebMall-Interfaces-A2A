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
    """Standardizes URLs for accurate comparison."""
    if not url or not isinstance(url, str):
        return ""
    return url.rstrip('/').lower()

class A2ABenchmark:
    def __init__(self):
        """
        Initializes the Scripted Buyer. No LLM used for decision making.
        Focuses on A2A protocol compliance and deterministic results.
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.endpoints = self._load_registry()
        self.results = []

    def _load_registry(self) -> List[str]:
        """Discovery: Load shop agent URLs from registry.json."""
        if not os.path.exists(REGISTRY_PATH):
            return [f"http://localhost:800{i+1}/messages" for i in range(4)]
        try:
            with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [shop['url'] for shop in data.get('shops', [])]
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return []

    def _parse_price(self, price_value: Any) -> float:
        """Robustly extracts a float from various price formats (e.g., '$120.00', 120)."""
        if isinstance(price_value, (int, float)):
            return float(price_value)
        if isinstance(price_value, str):
            clean_price = re.sub(r'[^\d.]', '', price_value)
            try:
                return float(clean_price)
            except ValueError:
                return float('inf')
        return float('inf')

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
        """Executes a JSON-RPC 2.0 call with debug logging for each shop."""
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
                res_json = response.json()
                
                # Immediate terminal debug output
                print(f"\n--- DEBUG: RAW RESPONSE FROM {url} ---")
                print(json.dumps(res_json, indent=2))
                print("-" * 50)
                
                return res_json
            except Exception as e:
                print(f"❌ Call failed to {url}: {e}")
                return {"error": str(e)}

    async def run_single_task(self, task: Dict):
        """Processes one task: Broadcase -> Harvest -> Min-Selection."""
        task_id = task['id']
        query = fill_urls(task['task'], URLS)
        raw_gt_urls = self.resolve_gt_urls(task.get('correct_answer', {}).get('answers', []))
        gt_urls_normalized = [normalize_url(u) for u in raw_gt_urls]
        
        start_time = time.time()
        
        # 1. Broadcast concurrently
        tasks = [self._execute_shop_call(url, query) for url in self.endpoints]
        responses = await asyncio.gather(*tasks)
        
        # 2. Token & Offer Harvesting
        total_shop_prompt = 0
        total_shop_completion = 0
        all_offers = []
        
        for resp in responses:
            if "result" in resp:
                res_obj = resp["result"]
                # Track token usage from the shop's LLM
                tokens = res_obj.get("tokens", {})
                total_shop_prompt += tokens.get("prompt_tokens", 0)
                total_shop_completion += tokens.get("completion_tokens", 0)
                
                # Handle 'offers' as a list of strings (URLs) or dicts (Products)
                offers = res_obj.get("offers", [])
                for o in offers:
                    if isinstance(o, dict):
                        all_offers.append(o)
                    else:
                        # Default high price if not specified
                        all_offers.append({"url": str(o), "price": 999999.0})

        # 3. Decision Logic: Select products matching the minimum price
        predicted_urls = []
        if all_offers:
            try:
                # Calculate minimum price across all collected offers
                min_price = min(self._parse_price(o.get('offers', {}).get('price', o.get('price', 999999))) 
                               for o in all_offers)
                
                # Filter for all offers that share this minimum price
                predicted_urls = [
                    normalize_url(o['url']) 
                    for o in all_offers 
                    if self._parse_price(o.get('offers', {}).get('price', o.get('price', 999999))) == min_price
                ]
            except Exception:
                predicted_urls = [normalize_url(o['url']) for o in all_offers if 'url' in o]

        # 4. Metric Calculation
        metrics = calculation_results(gt_urls_normalized, predicted_urls)
        
        return {
            "summary": {
                "task_id": task_id,
                "task_completion_rate": metrics["task_completion_rate"],
                "f1_score": metrics["f1_score"],
                "shop_tokens": total_shop_prompt + total_shop_completion,
                "buyer_tokens": 0,
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
        """Main execution loop for all task sets."""
        if not os.path.exists(TASK_FILE):
            print(f"❌ Error: Task file not found at {TASK_FILE}")
            return

        with open(TASK_FILE, 'r', encoding='utf-8') as f:
            task_sets = json.load(f)

        total_tasks = sum(len(ts.get('tasks', [])) for ts in task_sets)
        print(f"📊 Starting Scripted A2A Benchmark for {total_tasks} tasks...")
        
        details_log = []
        self.results = []
        start_wall_time = time.time()

        for task_set in task_sets:
            print(f"\n📂 Processing Task Set: {task_set.get('id', 'Unknown')}")
            for task in task_set.get('tasks', []):
                res = await self.run_single_task(task)
                summary = res["summary"]
                self.results.append(summary)
                details_log.append(res["detail"])
                
                print(f"Task: {summary['task_id']:<40} | "
                      f"CR: {summary['task_completion_rate']:.2f} | "
                      f"F1: {summary['f1_score']:.2f} | "
                      f"ShopTokens: {summary['shop_tokens']:<6}")

        # --- Final Aggregation and Storage ---
        total_processed = len(self.results)
        if total_processed == 0: return

        avg_cr = sum(r["task_completion_rate"] for r in self.results) / total_processed
        avg_f1 = sum(r["f1_score"] for r in self.results) / total_processed
        total_s_tokens = sum(r.get("shop_tokens", 0) for r in self.results)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_summary_{timestamp}.json")
        debug_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_debug_{timestamp}.json")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": {"timestamp": timestamp, "avg_cr": avg_cr, "avg_f1": avg_f1}, "results": self.results}, f, indent=4)
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