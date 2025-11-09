import os
import json
import httpx
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# --- Configuration---
# Ensure V27 Coordinator and V27-V31 Buyers are RUNNING
COORDINATOR_URL = "http://localhost:11000/process_query"

# V4: URL normalization map (Output)
# (Converts real URLs back to placeholders for F1 comparison)
URL_MAPPINGS = {
    "https://webmall-1.informatik.uni-mannheim.de": "{{URL_1}}",
    "https://webmall-2.informatik.uni-mannheim.de": "{{URL_2}}",
    "https://webmall-3.informatik.uni-mannheim.de": "{{URL_3}}",
    "https://webmall-4.informatik.uni-mannheim.de": "{{URL_4}}",
    # Add http if needed
    "http://webmall-1.informatik.uni-mannheim.de": "{{URL_1}}",
    "http://webmall-2.informatik.uni-mannheim.de": "{{URL_2}}",
    "http://webmall-3.informatik.uni-mannheim.de": "{{URL_3}}",
    "http://webmall-4.informatik.uni-mannheim.de": "{{URL_4}}",
}

# V7: URL replacement map (Input)
# (Converts placeholders to real URLs for API calls)
URL_REPLACEMENTS = {
    "{{URL_1}}": "https://webmall-1.informatik.uni-mannheim.de",
    "{{URL_2}}": "https://webmall-2.informatik.uni-mannheim.de",
    "{{URL_3}}": "https://webmall-3.informatik.uni-mannheim.de",
    "{{URL_4}}": "https://webmall-4.informatik.uni-mannheim.de",
}

# --- Path Definitions ---
BASE_DIR = Path(__file__).resolve().parents[1]
DEV_SET_PATH = BASE_DIR / "dev_set3.json"
TEST_SET_PATH = BASE_DIR / "test_set3.json"

LOG_DIR = BASE_DIR / "results"
LOG_FILES = [
    LOG_DIR / "coordinator_reasoning_log.jsonl",
    LOG_DIR / "buyer_1_reasoning.jsonl",
    LOG_DIR / "buyer_2_reasoning.jsonl",
    LOG_DIR / "buyer_3_reasoning.jsonl",
    LOG_DIR / "buyer_4_reasoning.jsonl",
]

# --- 1. V4 F1-Score Helpers ---

def normalize_url(url: str) -> str:
    """
    (V4) Converts 'https://webmall-1...' to '{{URL_1}}...'
    and removes trailing '/' for fair F1 comparison.
    """
    url = url.strip().rstrip('/')
    # Replace known base URLs
    for real_url, placeholder in URL_MAPPINGS.items():
        if url.startswith(real_url):
            url = url.replace(real_url, placeholder, 1)
            break
    return url

def extract_predicted_urls(raw_data: List[Dict[str, Any]]) -> set:
    """
    (V4) Extracts all 'success' URLs from the Coordinator's 'raw_data'.
    This works for SEARCH, CHEAPEST, and (critically) CHECKOUT tasks,
    as it finds the results of the SEARCH step in the plan.
    """
    predicted_urls = set()
    for buyer_res in raw_data:
        # (V6 FIX) We only check for SEARCH results (content is list)
        if buyer_res.get("status") == "success" and isinstance(buyer_res.get("content"), list):
            for item in buyer_res["content"]:
                if isinstance(item, dict) and "url" in item:
                    predicted_urls.add(normalize_url(item["url"]))
    return predicted_urls

def calculate_f1_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """(V4) Calculates Precision, Recall, and F1-Score"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}

# --- 2. Load Task Set (V3) ---
def load_task_set(file_path: Path) -> List[Dict[str, Any]]:
    """
    (V3) Loads the dev_set or test_set JSON file, 
    which is a LIST of CATEGORIES, each containing a nested 'tasks' list.
    """
    print(f"[Benchmark] Loading tasks from: {file_path.name}")
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return []
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # (V3 FIX) Correctly parse the nested structure
    all_tasks = []
    if isinstance(data, list):
        for category_obj in data:
            category_name = category_obj.get("tasks", [{}])[0].get("category", "Unknown")
            if "tasks" in category_obj and isinstance(category_obj["tasks"], list):
                # (V4) Inject the category name into the task for grouping
                for task in category_obj["tasks"]:
                    task["benchmark_category"] = category_name # e.g., "Specific_Product"
                all_tasks.extend(category_obj["tasks"])
            else:
                print(f"[WARN] Category object {category_obj.get('id')} has no 'tasks' list.")
    
    if not all_tasks:
        print("[ERROR] Task file format not recognized or no tasks found. Expected a list of categories [{... 'tasks': [...] ...}].")

    return all_tasks

# --- 3. Run Single Task (V7) ---
async def run_single_task(client: httpx.AsyncClient, task: Dict[str, Any]) -> Dict[str, Any]:
    """(V7) Calls the Coordinator, UNIVERSALLY filling in ALL templates."""
    
    task_id = task.get("id", "unknown_task")
    user_query_raw = task.get("task", "")
    gt_urls_raw = task.get("correct_answer", {}).get("answers", [])
    gt_set = {normalize_url(url) for url in gt_urls_raw}

    if not user_query_raw:
        return {"id": task_id, "status": "error", "error": "Task has no 'task' field", "tp": 0, "fp": 0, "fn": len(gt_set)}

    # (V2 FIX) Clean the <task> tags
    match = re.search(r'<task>(.*?)</task>', user_query_raw, re.DOTALL | re.IGNORECASE)
    user_query = match.group(1).strip() if match else user_query_raw.strip()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!! --- V7 FIX: Universal Template Filler (START) ---              !!!
    # !!! This fixes the V6 bug. We build a dict of *all* templates    !!!
    # !!! ({{name}}, {{card}}, AND {{URL_X}}) and replace them all.     !!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # 1. Build a dictionary of *all* possible templates
    templates_to_fill = {}
    templates_to_fill.update(task.get("user_details", {}))
    templates_to_fill.update(task.get("payment_info", {}))
    
    # 2. Add URL replacement templates (e.g., "{{URL_1}}" -> "https://...")
    for placeholder, real_url in URL_REPLACEMENTS.items():
        # We store the *key* without braces (e.g., "URL_1")
        templates_to_fill[placeholder.strip("{}")] = real_url
        
    # 3. (V7 FIX 1.C) Accumulate replacements (don't reset the query string)
    for key, value in templates_to_fill.items():
        if value: # Only replace if value is not None
            user_query = user_query.replace(f"{{{{{key}}}}}", str(value))

    # This V7 logic is universal. 
    # - For SEARCH, templates_to_fill is empty, so nothing happens.
    # - For CHECKOUT, all templates ({{name}}, {{URL_3}}) are filled.
    # This fixes both the F1=0.0 and the 50% Completion bugs.
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # !!! --- V7 FIX: Universal Template Filler (END) ---                !!!
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    print(f"[Benchmark] Running Task [ {task_id} ] (Query: \"{user_query[:50]}...\")")
    
    payload = {"query": user_query}
    
    try:
        response = await client.post(COORDINATOR_URL, json=payload, timeout=120.0)
        
        if response.status_code >= 400:
             print(f"[Benchmark] Task [ {task_id} ] FAILED (HTTP Error {response.status_code}): {response.text[:200]}")
             response.raise_for_status()
        
        result = response.json()
        
        # (V4) F1-SCORE CALCULATION
        pred_set = extract_predicted_urls(result.get("raw_data", []))
        
        tp = len(gt_set.intersection(pred_set))
        fp = len(pred_set.difference(gt_set))
        fn = len(gt_set.difference(pred_set))
        
        has_buyer_error = any(b.get("status") == "error" for b in result.get("raw_data", []))
        status = "partial_success" if has_buyer_error else "success"
        
        if not has_buyer_error: print(f"[Benchmark] Task [ {task_id} ] FINISHED (Success)")
        else: print(f"[Benchmark] Task [ {task_id} ] FINISHED (with Buyer errors)")

        return {"id": task_id, "status": status, "tp": tp, "fp": fp, "fn": fn}

    except Exception as e:
        print(f"[Benchmark] Task [ {task_id} ] FAILED (Error): {e}")
        return {"id": task_id, "status": "error", "error": str(e), "tp": 0, "fp": 0, "fn": len(gt_set)}

# --- 4. Token Calculation (V4) ---
def clear_log_files():
    """(V4) Clears logs BEFORE a single task run."""
    for log_file in LOG_FILES:
        if log_file.exists():
            try:
                with open(log_file, 'w') as f:
                    pass
            except Exception as e:
                print(f"[WARN] Could not clear log file: {log_file}. Error: {e}")

def calculate_log_tokens() -> int:
    """(V4) Parses all 5 logs and sums token_usage."""
    total_tokens = 0
    
    for log_file in LOG_FILES:
        if not log_file.exists(): continue
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        if "token_usage" in log_entry:
                            total_tokens += log_entry.get("token_usage", 0)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[ERROR] Could not read log file: {log_file}. Error: {e}")
            
    return total_tokens

# --- 5. Main Benchmark Runner (V7) ---
async def main():
    
    tasks = load_task_set(TEST_SET_PATH) # (Switch to TEST_SET_PATH for final run)
    if not tasks:
        print("[ERROR] No tasks found. Exiting.")
        return

    total_tasks = len(tasks)
    
    # (V4) Metrics storage (per-category)
    category_metrics = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0,
        "tokens": 0, "completed": 0, "task_count": 0
    })
    
    overall_metrics = defaultdict(int)

    print(f"\n--- Starting A2A Benchmark Run ({total_tasks} tasks) ---")
    
    # (V5 FIX) Re-added the httpx.AsyncClient
    async with httpx.AsyncClient() as client:
        for task in tasks:
            # (V4) CRITICAL: Clear logs before *each* run to isolate token usage
            clear_log_files()
            
            result = await run_single_task(client, task) # (V7 logic)
            
            # (V4) CRITICAL: Calculate tokens *immediately* after the task
            task_tokens = calculate_log_tokens()
            
            # (V4) Get category and aggregate metrics
            category = task.get("benchmark_category", "Unknown")
            
            # Aggregate metrics into its category
            category_metrics[category]["tp"] += result.get("tp", 0)
            category_metrics[category]["fp"] += result.get("fp", 0)
            category_metrics[category]["fn"] += result.get("fn", 0)
            category_metrics[category]["tokens"] += task_tokens
            category_metrics[category]["task_count"] += 1
            
            if result["status"] == "success" or result["status"] == "partial_success":
                category_metrics[category]["completed"] += 1

            await asyncio.sleep(1) # Small delay

    print("--- A2A Benchmark Run Complete ---")

    # (STEP 5) V4/V5 Report
    print("\n--- Benchmark Report (V7 - Per Category) ---")

    for category, metrics in category_metrics.items():
        task_count = metrics["task_count"]
        comp_rate = (metrics["completed"] / task_count) * 100 if task_count > 0 else 0
        avg_tokens = (metrics["tokens"] / task_count) if task_count > 0 else 0
        f1_data = calculate_f1_metrics(metrics["tp"], metrics["fp"], metrics["fn"])
        
        print(f"\n[Category: {category}] ({task_count} tasks)")
        print(f"  Completion Rate: {comp_rate:.2f}% ({metrics['completed']}/{task_count})")
        print(f"  F1-Score:        {f1_data['f1']:.4f}")
        print(f"  Avg Token Usage: {avg_tokens:.0f} tokens")
        print(f"  (P: {f1_data['precision']:.4f}, R: {f1_data['recall']:.4f}, TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']})")
        
        # Aggregate to overall
        overall_metrics["tp"] += metrics["tp"]
        overall_metrics["fp"] += metrics["fp"]
        overall_metrics["fn"] += metrics["fn"]
        overall_metrics["tokens"] += metrics["tokens"]
        overall_metrics["completed"] += metrics["completed"]
        overall_metrics["task_count"] += metrics["task_count"]

    print("\n--- Benchmark Report (V7 - Overall) ---")
    
    total_task_count = overall_metrics["task_count"]
    
    overall_f1_data = calculate_f1_metrics(
        overall_metrics["tp"], 
        overall_metrics["fp"], 
        overall_metrics["fn"]
    )
    overall_comp_rate = (overall_metrics["completed"] / total_task_count) * 100 if total_task_count > 0 else 0
    avg_token_per_task = (overall_metrics["tokens"] / total_task_count) if total_task_count > 0 else 0

    print(f"Overall Completion Rate: {overall_comp_rate:.2f}% ({overall_metrics['completed']}/{total_task_count})")
    print(f"Micro-Avg F1-Score:    {overall_f1_data['f1']:.4f}")
    print(f"Total Token Usage:       {overall_metrics['tokens']} tokens")
    print(f"Average Token (Per Task):{avg_token_per_task:.0f} tokens")
    print(f"(Total TP: {overall_metrics['tp']}, Total FP: {overall_metrics['fp']}, Total FN: {overall_metrics['fn']})")


if __name__ == "__main__":
    # Ensure all 5 agents (V31) are running
    asyncio.run(main())