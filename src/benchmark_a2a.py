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
from typing import Tuple, Dict, List, Any, Optional 

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

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

URLS = {
    "URL_1": "https://webmall-1.informatik.uni-mannheim.de",
    "URL_2": "https://webmall-2.informatik.uni-mannheim.de",
    "URL_3": "https://webmall-3.informatik.uni-mannheim.de",
    "URL_4": "https://webmall-4.informatik.uni-mannheim.de",
    "URL_5": "https://webmall-solution.informatik.uni-mannheim.de"
}

import re

def normalize_url(url: str) -> str:
    """
    Normalize URL for comparison by removing trailing slashes and converting to lowercase.
    Ported from nlweb_mcp benchmark logic.
    """
    if not url or not isinstance(url, str):
        return ""
    return url.rstrip('/').lower()

def extract_urls_from_response(response_text: str) -> set[str]:
    """
    Extracts URLs from the agent's response using a multi-layered fallback strategy.
    Ported from nlweb_mcp benchmark for high-resilience extraction.
    """
    if not isinstance(response_text, str):
        return set()
    
    # 1. Strategy: Parse the entire response as JSON
    try:
        data = json.loads(response_text.strip())
        if isinstance(data, dict) and "urls" in data:
            # Expected format: {"urls": ["url1", "url2"]}
            urls = data["urls"]
            if isinstance(urls, list):
                return set(u for u in urls if isinstance(u, str))
        elif isinstance(data, list):
            # Fallback format: ["url1", "url2"]
            return set(u for u in data if isinstance(u, str))
    except (json.JSONDecodeError, TypeError):
        pass
    
    # 2. Strategy: Find JSON patterns within mixed text content
    # Useful when LLM adds explanatory text around the JSON block
    json_pattern = r'\{"urls":\s*\[.*?\]\}'
    json_matches = re.findall(json_pattern, response_text, re.DOTALL)
    for match in json_matches:
        try:
            data = json.loads(match)
            if isinstance(data, dict) and "urls" in data and isinstance(data["urls"], list):
                return set(u for u in data["urls"] if isinstance(u, str))
        except (json.JSONDecodeError, TypeError):
            continue
            
    # 3. Strategy: Final fallback to raw regex search
    # Extracts anything that looks like a webmall URL
    urls_found = re.findall(r'https?://\S+', response_text)
    # Strip common trailing punctuation characters often included by LLMs
    return set([url.strip(')>."\',') for url in urls_found])

def fill_urls(text: str, urls: Dict[str, str]) -> str:
    """Replace URL placeholders like {{URL_3}} with actual Mannheim webmall URLs."""
    for key, val in urls.items():
        text = text.replace("{{" + key + "}}", val)
    return text

class A2ABenchmark:
    def __init__(self):
        """
        Initializes the benchmark with dynamic discovery and URL mapping.
        """
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Implement A2A Discovery Method 3: Load endpoints from registry.json
        self.endpoints = self._load_registry()
        
        # Initialize OpenAI client for the Buyer agent
        self.client = AsyncOpenAI() 
        
        # Mapping for Ground Truth resolution, compatible with the URLS dictionary
        self.url_mapping = URLS
        self.results = []
        self.buyer_llm = ChatOpenAI(model="gpt-5-mini")

    def _load_registry(self) -> List[str]:
        """
        A2A Discovery Method 3: Dynamically fetch shop agent URLs from the registry.
        Falls back to local ports if registry.json is missing.
        """
        if not os.path.exists(REGISTRY_PATH):
            print(f"⚠️ Registry not found at {REGISTRY_PATH}, using local defaults.")
            return [f"http://localhost:800{i+1}/messages" for i in range(4)]
            
        try:
            with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                endpoints = [shop['url'] for shop in data.get('shops', [])]
                print(f"✅ Discovered {len(endpoints)} shop agents via registry.")
                return endpoints
        except Exception as e:
            print(f"❌ Discovery failed: {e}")
            return []

    def resolve_gt_urls(self, raw_urls: List[str]) -> List[str]:
        """
        Convert placeholder URLs in the dataset (e.g., {{URL_3}}) to real shop URLs.
        Uses the shared URLS mapping for consistency.
        """
        resolved = []
        for url in raw_urls:
            new_url = url
            for key, actual in self.url_mapping.items():
                placeholder = "{{" + key + "}}"
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

    async def get_buyer_decision(self, wish: str, shop_context: str) -> Tuple[str, Dict]:
        """
        Refined Buyer Decision Phase with strict filtering and price comparison.
        Ported from the reference implementation's logic.
        """
        system_prompt = (
            "You are an expert E-commerce Decision Agent. Your goal is to help a user find "
            "the EXACT products they want from multiple shop results.\n\n"
            "EVALUATION RULES:\n"
            "1. **Strict Matching**: Only select products that match the specifications (e.g., storage, color, model) in the user's wish.\n"
            "2. **Cheapest Only**: If the user asks for the 'cheapest' or 'best price', you MUST only return the URL(s) of the product(s) with the absolute lowest price. Ignore more expensive alternatives.\n"
            "3. **Multiple Matches**: If multiple shops offer the same lowest price for the same item, include all of their URLs.\n"
            "4. **No Hallucinations**: Only use the URLs provided in the shop context. Never invent URLs.\n\n"
            "OUTPUT FORMAT:\n"
            "- Your final response must be a JSON object: {\"urls\": [\"url1\", \"url2\", ...]}\n"
            "- If no products fulfill the criteria, return: {\"urls\": []}\n"
            "- Do not add any explanatory text."
        )

        user_message = f"User Wish: {wish}\n\nShop Results:\n{shop_context}"

        with get_openai_callback() as cb:
            # Using GPT-4o-mini as the buyer for cost-effective but smart filtering
            response = await self.buyer_llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ])
            
            usage = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens
            }
            return response.content, usage


    async def run_single_task(self, task: Dict):
        """
        Executes a single benchmark task with robust URL processing and normalization.
        Pre-processes URLs, broadcasts to shop agents, and aggregates results for buyer decision.
        """
        task_id = task['id']
        
        # 1. URL Pre-processing: Resolve {{URL_X}} placeholders
        query = fill_urls(task['task'], URLS)
        
        # Resolve Ground Truth URLs and apply normalization for accurate comparison
        raw_gt_urls = self.resolve_gt_urls(task.get('correct_answer', {}).get('answers', []))
        gt_urls_normalized = [normalize_url(u) for u in raw_gt_urls]
        
        start_time = time.time()
        
        # Increased timeout to 60s for multi-step agent reasoning
        async with httpx.AsyncClient(timeout=60.0) as client:
            # 2. Broadcast the resolved wish to all shop agents
            shop_responses = await asyncio.gather(*[self.call_shop_agent(client, url, query) for url in self.endpoints])
            
            debug_shops = []
            shop_context = ""
            total_shop_prompt = 0
            total_shop_completion = 0
            
            # 3. Process Shop Responses (JSON-RPC 2.0)
            for i, res in enumerate(shop_responses):
                if res.get("jsonrpc") == "2.0" and "result" in res:
                    data = res['result']
                    usage = data.get("tokens", {})
                    # Normalize offers received from shops
                    offers = [normalize_url(u) for u in data.get("offers", [])]
                    
                    total_shop_prompt += usage.get("prompt_tokens", 0)
                    total_shop_completion += usage.get("completion_tokens", 0)
                    
                    debug_shops.append({
                        "shop_id": f"webmall_{i+1}",
                        "offers_returned": offers,
                        "tokens": usage
                    })
                    shop_context += f"\nShop {i+1} offers: {json.dumps(offers)}\n"
                else:
                    debug_shops.append({
                        "shop_id": f"webmall_{i+1}",
                        "error": res.get("error", "Invalid JSON-RPC response or timeout")
                    })
        
        # 4. Buyer Decision Phase
        decision_str, buyer_usage = await self.get_buyer_decision(query, shop_context)
        
        # Robust URL extraction from buyer output
        extracted_urls = extract_urls_from_response(decision_str)
        predicted_urls = [normalize_url(u) for u in extracted_urls]
        
        # 5. Metric Calculation using normalized lists
        metrics = calculation_results(gt_urls_normalized, predicted_urls)
        
        task_detail = {
            "task_id": task_id,
            "wish": query,
            "ground_truth": raw_gt_urls, # Keep raw for debug view
            "predicted_urls": predicted_urls,
            "metrics": metrics,
            "buyer_raw_output": decision_str,
            "shops_detail": debug_shops
        }
        
        return {
            "summary": {
                "task_id": task_id,
                "task_completion_rate": metrics["task_completion_rate"],
                "f1_score": metrics["f1_score"],
                "shop_tokens": total_shop_prompt + total_shop_completion,
                "buyer_tokens": buyer_usage["prompt_tokens"] + buyer_usage["completion_tokens"],
                "total_tokens": (total_shop_prompt + total_shop_completion + 
                                buyer_usage["prompt_tokens"] + buyer_usage["completion_tokens"]),
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
        task_count = len(self.results)
        summary_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_{task_count}tasks_summary_{timestamp}.json")
        debug_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_{task_count}tasks_debug_{timestamp}.json")

        total_s_tokens = sum(r.get("shop_tokens", 0) for r in self.results)
        total_b_tokens = sum(r.get("buyer_tokens", 0) for r in self.results)
        total_tokens = total_s_tokens + total_b_tokens

        summary_payload = {
            "benchmark_metadata": {
                "timestamp": timestamp,
                "overall_metrics": {
                    "average_cr": avg_cr,
                    "average_f1": avg_f1,
                    "total_tokens": total_tokens,
                    "shop_tokens": total_s_tokens,  
                    "buyer_tokens": total_b_tokens  
                }
            },
            "token_usage_summary": {
                "total_shop_tokens": total_s_tokens,     
                "total_buyer_tokens": total_b_tokens,    
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
        print(f"📈 Average CR:   {avg_cr:.4f}")
        print(f"🎯 Average F1:   {avg_f1:.4f}")
        print(f"🤖 Shop Tokens:  {total_s_tokens}")  
        print(f"🛒 Buyer Tokens: {total_b_tokens}")  
        print(f"💰 Total Tokens: {total_tokens}")
        print(f"⏱️ Total Time:   {sum(r['execution_time_seconds'] for r in self.results):.2f}s")
        print("="*40)

if __name__ == "__main__":
    asyncio.run(A2ABenchmark().run_benchmark())