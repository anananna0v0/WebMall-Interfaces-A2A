import asyncio
import json
import time
import httpx
import tiktoken
import os
import sys
from typing import List, Dict
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Path setup and dependency loading
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
load_dotenv(os.path.join(BASE_DIR, ".env"))

from utils import calculation_results
from nlweb_mcp.config import WEBMALL_SHOPS

REGISTRY_PATH = os.path.join(BASE_DIR, "src", "a2a", "registry.json")
TASK_FILE = os.path.join(BASE_DIR, "task_sets", "experiment_tasks_4.json")
RESULTS_DIR = "/Users/yenanchen/Documents/Home/Mannheim/2025HWS/Seminar/WebMall-Interfaces-A2A-test/results/a2a/gpt-5-mini"
MODEL_NAME = "gpt-5-mini"

class A2ABenchmark:
    def __init__(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.endpoints = self._load_registry()
        self.client = AsyncOpenAI() 
        self.url_mapping = {f"{{{{URL_{i+1}}}}}" : WEBMALL_SHOPS[f"webmall_{i+1}"]["url"] for i in range(4)}
        self.results = []

    def _load_registry(self) -> List[str]:
        if not os.path.exists(REGISTRY_PATH):
            return [f"http://localhost:800{i+1}/messages" for i in range(4)]
        with open(REGISTRY_PATH, 'r') as f:
            data = json.load(f)
            return [shop['url'] for shop in data.get('shops', [])]

    def resolve_gt_urls(self, raw_urls: List[str]) -> List[str]:
        """Convert placeholder URLs in dataset to real shop URLs."""
        resolved = []
        for url in raw_urls:
            new_url = url
            for placeholder, actual in self.url_mapping.items():
                new_url = new_url.replace(placeholder, actual)
            resolved.append(new_url)
        return resolved

    async def call_shop_agent(self, client: httpx.AsyncClient, url: str, query: str) -> Dict:
        """Execute JSON-RPC 2.0 call to shop agent."""
        payload = {"jsonrpc": "2.0", "method": "ask_webmall", "params": {"query": query}, "id": "1"}
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            res_data = response.json().get("result", {})
            return {"status": "success", "data": res_data, "usage": res_data.get("usage", {})}
        except Exception:
            return {"status": "error", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}

    async def get_buyer_decision(self, query: str, context: str):
        """Final decision making with token usage tracking."""
        # System prompt for the Buyer Agent to perform cross-shop decision making
        system_prompt = """
            You are a strategic buyer agent. 
            You will receive a user's WISH and product data (JSON-LD) from multiple shops.

            Your task:
            1. Compare the offers based on the WISH (e.g., finding the absolute lowest price).
            2. Return ONLY the exact URL(s) of the matching products.
            3. Use ' ### ' to separate multiple URLs if they share the exact same lowest price.
            4. If NO product in the context matches the wish, return 'NONE'.

            Do not explain. Return ONLY the URL(s) or 'NONE'.
            """
        user_content = f"Wish: {query}\nContext: {context}"
        
        response = await self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        )
        output = response.choices[0].message.content.strip()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }
        return output, usage


    async def run_single_task(self, task: Dict):
        task_id = task['id']
        query = task['task']
        gt_urls = self.resolve_gt_urls(task.get('correct_answer', {}).get('answers', []))
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            # Broadcast wish to all shops
            shop_responses = await asyncio.gather(*[self.call_shop_agent(client, url, query) for url in self.endpoints])
            
            # --- NEW: Debug data collection ---
            debug_shops = []
            shop_context = ""
            total_shop_prompt = 0
            total_shop_completion = 0
            
            for i, res in enumerate(shop_responses):
                if res["status"] == "success":
                    data = res['data']
                    usage = res.get("usage", {})
                    
                    # Record exactly what each shop returned (JSON-LD offers)
                    debug_shops.append({
                        "shop_name": data.get("agent_name"),
                        "offers_returned": data.get("offers", []), # This is your JSON-LD data
                        "tokens": usage
                    })
                    
                    total_shop_prompt += usage.get("prompt_tokens", 0)
                    total_shop_completion += usage.get("completion_tokens", 0)
                    shop_context += f"\nShop {i+1}: {json.dumps(data.get('offers'))}\n"
            
            # Buyer decision phase
            decision_str, buyer_usage = await self.get_buyer_decision(query, shop_context)
            predicted_urls = [u.strip() for u in decision_str.split(' ### ') if u.strip() and u.strip().upper() != 'NONE']
            
            metrics = calculation_results(gt_urls, predicted_urls)
            
            # --- Store full details for the debug file ---
            task_detail = {
                "task_id": task_id,
                "wish": query,
                "ground_truth": gt_urls,
                "predicted_urls": predicted_urls,
                "metrics": metrics,
                "buyer_raw_output": decision_str,
                "shops_detail": debug_shops # Full transparency of shop outputs
            }
            
            return {
                "summary": {
                    "task_id": task_id,
                    "task_completion_rate": metrics["task_completion_rate"],
                    "f1_score": metrics["f1_score"],
                    "prompt_tokens": total_shop_prompt + buyer_usage["prompt_tokens"],
                    "completion_tokens": total_shop_completion + buyer_usage["completion_tokens"],
                    "execution_time_seconds": time.time() - start_time
                },
                "detail": task_detail
            }

    async def run_benchmark(self):
        """
        Main execution loop with overall metrics calculation.
        """
        if not os.path.exists(TASK_FILE):
            print(f"Error: Task file not found at {TASK_FILE}")
            return

        with open(TASK_FILE, 'r', encoding='utf-8') as f:
            tasks = json.load(f)

        print(f"📊 Starting A2A Benchmark for {len(tasks)} tasks...")
        
        details_log = []
        self.results = []

        for task in tasks:
            res = await self.run_single_task(task)
            summary = res["summary"]
            self.results.append(summary)
            details_log.append(res["detail"])
            
            task_tokens = summary.get('prompt_tokens', 0) + summary.get('completion_tokens', 0)
            print(f"Task: {summary['task_id']} | "
                  f"CR: {summary['task_completion_rate']} | "
                  f"F1: {summary['f1_score']:.2f} | "
                  f"Tokens: {task_tokens} | "
                  f"Time: {summary['execution_time_seconds']:.2f}s")

        # --- Calculate Overall Metrics ---
        total_tasks = len(self.results)
        avg_cr = sum(r["task_completion_rate"] for r in self.results) / total_tasks
        avg_f1 = sum(r["f1_score"] for r in self.results) / total_tasks
        total_p_tokens = sum(r.get("prompt_tokens", 0) for r in self.results)
        total_c_tokens = sum(r.get("completion_tokens", 0) for r in self.results)
        total_tokens = total_p_tokens + total_c_tokens

        # Generate unique timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Define storage paths
        summary_path = os.path.join(RESULTS_DIR, f"a2a_results_{timestamp}.json")
        debug_path = os.path.join(RESULTS_DIR, f"a2a_results_{timestamp}_DEBUG.json")

        summary_payload = {
            "benchmark_metadata": {
                "timestamp": timestamp,
                "version": "a2a_enhanced_v1",
                "total_tasks": total_tasks,
                "model": MODEL_NAME,
                "overall_metrics": {
                    "average_cr": avg_cr,
                    "average_f1": avg_f1,
                    "total_tokens": total_tokens
                }
            },
            "token_usage_summary": {
                "total_prompt_tokens": total_p_tokens,
                "total_completion_tokens": total_c_tokens,
                "total_tokens": total_tokens
            },
            "results": self.results
        }
        
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=4)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(details_log, f, indent=4)

        # --- Final Overall Output to Terminal ---
        print("\n" + "="*40)
        print("🏆 FINAL BENCHMARK RESULTS")
        print("="*40)
        print(f"📈 Average CR: {avg_cr:.4f}")
        print(f"🎯 Average F1: {avg_f1:.4f}")
        print(f"💰 Total Tokens Used: {total_tokens}")
        print(f"⏱️ Total Execution Time: {sum(r['execution_time_seconds'] for r in self.results):.2f}s")
        print("="*40)
        print(f"📁 Summary: {summary_path}")
        print(f"🔍 Debug: {debug_path}")

if __name__ == "__main__":
    asyncio.run(A2ABenchmark().run_benchmark())