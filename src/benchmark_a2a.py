# src/benchmark_a2a.py

import os
import json
import time
import re
import csv
import argparse
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from utils import (
    calculation_results,
    interface_results_dir,
)

load_dotenv()

# ---------- WebMall URL placeholders ----------
URLS = {
    "URL_1": "https://webmall-1.informatik.uni-mannheim.de",
    "URL_2": "https://webmall-2.informatik.uni-mannheim.de",
    "URL_3": "https://webmall-3.informatik.uni-mannheim.de",
    "URL_4": "https://webmall-4.informatik.uni-mannheim.de",
    "URL_5": "https://webmall-solution.informatik.uni-mannheim.de",
}


# ---------- Helpers (kept compatible with benchmark_* scripts) ----------
def fill_urls(text: str, urls: Dict[str, str]) -> str:
    for key, val in urls.items():
        text = text.replace("{{" + key + "}}", val)
    return text


def normalize_url(url: str) -> str:
    return url.rstrip("/").lower()


def extract_urls_from_response(response_text: Any) -> Set[str]:
    """
    Try JSON {"urls":[...]} first, then JSON array, then fallback regex.
    Same spirit as benchmark_nlweb_mcp.py.
    """
    if not isinstance(response_text, str):
        return set()

    raw = response_text.strip()

    # 1) whole JSON
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "urls" in data and isinstance(data["urls"], list):
            return set(
                u for u in data["urls"]
                if isinstance(u, str) and u.strip() and u.strip().lower() != "done"
            )
        if isinstance(data, list):
            return set(
                u for u in data
                if isinstance(u, str) and u.strip() and u.strip().lower() != "done"
            )
    except Exception:
        pass

    # 2) embedded JSON object {"urls":[...]}
    obj_pat = r'\{"urls":\s*\[.*?\]\}'
    for m in re.findall(obj_pat, response_text, re.DOTALL):
        try:
            data = json.loads(m)
            if isinstance(data, dict) and "urls" in data and isinstance(data["urls"], list):
                return set(
                    u for u in data["urls"]
                    if isinstance(u, str) and u.strip() and u.strip().lower() != "done"
                )
        except Exception:
            continue

    # 3) embedded JSON array [...]
    arr_pat = r'\[(?:["\'][^"\']*["\'](?:\s*,\s*)?)+\]'
    for m in re.findall(arr_pat, response_text):
        try:
            data = json.loads(m)
            if isinstance(data, list):
                urls = set(
                    u for u in data
                    if isinstance(u, str) and u.strip() and u.strip().lower() != "done"
                )
                if urls:
                    return urls
        except Exception:
            continue

    # 4) regex fallback
    urls_found = re.findall(r"https?://\S+", response_text)
    return set([u.strip(')>."\',') for u in urls_found])


def safe_get(d: Any, path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_tasks(benchmark_path: str) -> List[Dict[str, Any]]:
    """
    task_sets_35.json format:
    [
      { "id": "...", "tasks": [ {...}, {...} ] },
      { "id": "...", "tasks": [ ... ] }
    ]
    We flatten groups into a single list of task dicts, while carrying the group_id.
    """
    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: already a flat list of tasks
    if isinstance(data, list) and data and isinstance(data[0], dict) and "correct_answer" in data[0]:
        return data

    # Case 2: list of groups (your file)
    if isinstance(data, list) and data and isinstance(data[0], dict) and "tasks" in data[0]:
        flat: List[Dict[str, Any]] = []
        for group in data:
            group_id = group.get("id")
            group_tasks = group.get("tasks", [])
            if not isinstance(group_tasks, list):
                continue
            for t in group_tasks:
                if isinstance(t, dict):
                    # keep group info for debugging/analysis if needed
                    t = dict(t)
                    t["_group_id"] = group_id
                    flat.append(t)
        return flat

    # Case 3: dict wrapper
    if isinstance(data, dict) and "tasks" in data and isinstance(data["tasks"], list):
        return data["tasks"]

    raise ValueError("Unsupported benchmark JSON format.")


def build_user_task(task: Dict[str, Any]) -> Tuple[str, List[str], str]:
    """
    Re-implements the task text filling logic from benchmark_nlweb_mcp.py:
    - Extract task text from "task" or from <task>...</task> inside "instruction"
    - Fill placeholders for checkout tasks (user details, payment)
    Returns: (user_task_text, expected_urls_flat, task_category)
    """
    user_task = task.get("task")
    if not user_task:
        instruction = task.get("instruction", "")
        if isinstance(instruction, str) and "<task>" in instruction and "</task>" in instruction:
            start = instruction.find("<task>")
            end = instruction.find("</task>") + len("</task>")
            user_task = instruction[start:end]
        else:
            user_task = str(instruction)

    user_task = re.sub(r"^\s*<task>\s*", "", user_task)
    user_task = re.sub(r"\s*</task>\s*$", "", user_task)

    task_category = task.get("category", "")

    correct_answer = task.get("correct_answer", {}).get("answers", [])
    expected_flat = [fill_urls(x, URLS) for x in correct_answer]

    # Checkout / FindAndOrder placeholder replacements
    if task_category in ("Checkout", "FindAndOrder"):
        # product url placeholder
        product_urls = [fill_urls(x, URLS) for x in correct_answer]
        user_task = user_task.replace("{{product_url}}", str(product_urls))

        user_details = task.get("user_details", {})
        for key in ["name", "email", "street", "house_number", "zip", "city", "state", "country"]:
            if key in user_details:
                user_task = user_task.replace("{{" + key + "}}", str(user_details[key]))

        payment_info = task.get("payment_info", {})
        for key in ["card", "cvv", "expiry_date"]:
            if key in payment_info:
                user_task = user_task.replace("{{" + key + "}}", str(payment_info[key]))

    user_task = fill_urls(user_task, URLS)
    if "{{product_url}}" in user_task:
        user_task = user_task.replace("{{product_url}}", str(expected_flat))

    return user_task, expected_flat, task_category


# ---------- Token aggregation (paper-aligned: LLM input+output tokens only) ----------
def aggregate_llm_tokens(a2a_result: Dict[str, Any]) -> Tuple[int, int]:
    """
    Returns (prompt_tokens, completion_tokens) for the whole task.
    Supports multiple possible response shapes:
    - top-level: prompt_tokens, completion_tokens
    - top-level: usage: {prompt_tokens, completion_tokens}
    - buyer_usage + shop_results[].usage
    """
    def read_usage(obj: Any) -> Tuple[int, int]:
        if not isinstance(obj, dict):
            return (0, 0)

        # direct
        if "prompt_tokens" in obj or "completion_tokens" in obj:
            return (int(obj.get("prompt_tokens", 0)), int(obj.get("completion_tokens", 0)))

        # nested usage
        u = obj.get("usage")
        if isinstance(u, dict):
            return (int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))

        return (0, 0)

    p, c = 0, 0

    # 1) top-level
    tp, tc = read_usage(a2a_result)
    p += tp
    c += tc

    # 2) buyer_usage
    bu = a2a_result.get("buyer_usage")
    bp, bc = read_usage(bu)
    p += bp
    c += bc

    # 3) shops
    shops = a2a_result.get("shop_results", [])
    if isinstance(shops, list):
        for s in shops:
            sp, sc = read_usage(s)
            p += sp
            c += sc

    return p, c


# ---------- A2A call ----------
def call_buyer(
    buyer_endpoint: str,
    task_id: str,
    user_task: str,
    task_category: str,
    expected_urls: List[str],
    clarify_policy: str,
    timeout_sec: int,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Buyer server contract (recommended minimal):
    POST {buyer_endpoint}/run_task
    {
      "task_id": "...",
      "task_category": "...",
      "task": "...",               # filled user_task text
      "expected_urls": [...],      # optional, can help buyer for checkout placeholders; not required
      "clarify_policy": "off" | "on_once",
      ...extra...
    }

    Response minimal:
    {
      "final_answer": "... or {\"urls\": [...]} ...",
      "final_urls": [...],                 # optional shortcut
      "buyer_usage": {...},                # optional
      "shop_results": [...],               # optional
    }
    """
    payload = {
        "task_id": task_id,
        "task_category": task_category,
        "task": user_task,
        "expected_urls": expected_urls,
        "clarify_policy": clarify_policy,
    }
    if extra_payload:
        payload.update(extra_payload)

    url = buyer_endpoint.rstrip("/") + "/run_task"
    r = requests.post(url, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError("Buyer response is not a JSON object.")
    return data


def pick_evaluation_urls(
    a2a_result: Dict[str, Any],
    task_category: str,
    final_answer_text: str
) -> Tuple[List[str], str]:
    """
    Evaluation strategy for A2A (Specific_Product only):

    - The buyer returns final_urls as the final answer.
    - Add_To_Cart, Checkout, and FindAndOrder tasks are not evaluated.
    - cart_only_urls and checkout_only_urls are ignored and kept only for schema compatibility.
    """

    # Prefer explicit urls fields if buyer returns them
    final_urls = a2a_result.get("final_urls")

    # Search / others
    if isinstance(final_urls, list):
        return [normalize_url(u) for u in final_urls if isinstance(u, str)], "final_urls"
    got = extract_urls_from_response(final_answer_text)
    return [normalize_url(u) for u in got], "final_answer"


def generate_csv_metrics(enhanced_summary: Dict[str, Any], output_dir: Path, prefix: str = "a2a"):
    results = enhanced_summary.get("results") or []
    if not results:
        print("No results to export to CSV")
        return None

    current_timestamp = safe_get(enhanced_summary, ["benchmark_metadata", "timestamp"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    csv_path = output_dir / f"{prefix}_metrics_{current_timestamp}.csv"

    rows = []
    for r in results:
        error = r.get("error_occurred", False)
        rows.append({
            "task_category": r.get("task_id", "").split("_Task")[0],
            "task_id": r.get("task_id", ""),
            "task_completion_rate": 0 if error else r.get("task_completion_rate", 0),
            "precision": 0.0 if error else r.get("precision", 0.0),
            "recall": 0.0 if error else r.get("recall", 0.0),
            "f1_score": 0.0 if error else r.get("f1_score", 0.0),
            "prompt_tokens": 0 if error else r.get("prompt_tokens", 0),
            "completion_tokens": 0 if error else r.get("completion_tokens", 0),
            "execution_time_seconds": r.get("execution_time_seconds", 0.0),
        })

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"📊 CSV metrics exported to: {csv_path}")
    return str(csv_path)


def run_benchmark_a2a(
    benchmark_path: str,
    buyer_endpoint: str,
    model_name: str,
    clarify_policy: str,
    timeout_sec: int,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tasks = load_tasks(benchmark_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = interface_results_dir(__file__, "a2a", model_name, reasoning_effort=None)

    execution_history: List[Dict[str, Any]] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_execution_time = 0.0
    successful_tasks = 0
    failed_tasks = 0

    for task in tasks:
        task_id = task.get("id", "")
        user_task, expected_flat, task_category = build_user_task(task)

        print("=" * 60)
        print(f"Task ID: {task_id}")
        print(f"Category: {task_category}")
        print(f"User task: {user_task}")
        print(f"Expected: {expected_flat}")

        expected_normalized = [normalize_url(u) for u in expected_flat]

        task_start = time.time()
        task_error = None
        buyer_result: Dict[str, Any] = {}
        final_answer_text = ""

        evaluation_urls = []
        try:
            buyer_result = call_buyer(
                buyer_endpoint=buyer_endpoint,
                task_id=task_id,
                user_task=user_task,
                task_category=task_category,
                expected_urls=expected_flat,
                clarify_policy=clarify_policy,
                timeout_sec=timeout_sec,
                extra_payload=extra_payload,
            )

            # === USE BUYER final_urls FOR EVALUATION ===
            evaluation_urls = []

            if isinstance(buyer_result, dict):
                urls = buyer_result.get("final_urls", [])
                if isinstance(urls, list):
                    evaluation_urls = [normalize_url(u) for u in urls]

            # final answer text (may be string or dict)
            fa = buyer_result.get("final_answer", "")
            if isinstance(fa, (dict, list)):
                final_answer_text = json.dumps(fa, ensure_ascii=False)
            else:
                final_answer_text = str(fa)

            successful_tasks += 1

        except Exception as e:
            task_error = str(e)
            failed_tasks += 1
            buyer_result = {}
            final_answer_text = f"Error: {task_error}"
            print(f"❌ Task failed: {task_error}")
            print(traceback.format_exc())

        exec_time = time.time() - task_start
        total_execution_time += exec_time

        # token aggregation (LLM inference only)
        prompt_tokens, completion_tokens = aggregate_llm_tokens(buyer_result) if not task_error else (0, 0)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        # evaluation urls selection (A2A: always use buyer final_urls)
        evaluation_strategy = "buyer_final_urls"

        if task_error:
            evaluation_urls = []


        # metrics
        task_metrics = calculation_results(
            benchmark_solutions=expected_normalized,
            model_solution=evaluation_urls
        )

        correct_model_answers = [
            u for u in expected_flat if normalize_url(u) in evaluation_urls
        ]
        additional_urls = [
            u for u in evaluation_urls if u not in expected_normalized
        ]
        missing_urls = [
            u for u in expected_normalized if u not in evaluation_urls
        ]

        history_entry = {
            "task_id": task_id,
            "task_category": task_category,
            "task": user_task,
            "task_completion_rate": task_metrics["task_completion_rate"],
            "precision": task_metrics["avg_precision"],
            "recall": task_metrics["avg_recall"],
            "f1_score": task_metrics["f1_score"],
            "raw_response": final_answer_text,
            "parsed_response": evaluation_urls,
            "correct_model_answers": correct_model_answers,
            "additional_urls": additional_urls,
            "missing_urls": missing_urls,
            "metrics": task_metrics,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "execution_time_seconds": exec_time,
            "expected_urls": expected_flat,
            "evaluation_strategy": evaluation_strategy,
            "buyer_result": buyer_result,  # keep for debugging
        }

        if task_error:
            history_entry["error"] = task_error
            history_entry["error_occurred"] = True
        else:
            history_entry["error_occurred"] = False

        execution_history.append(history_entry)

        # write per-task history entry (overwrite like other scripts)
        with (results_dir / "history_entry.json").open("w", encoding="utf-8") as f:
            json.dump(history_entry, f, indent=4, ensure_ascii=False)

        print(f"Final Answer: {final_answer_text}")
        print(f"Evaluation URLs ({evaluation_strategy}): {evaluation_urls}")
        print(f"Metrics: {task_metrics}")
        print(f"Tokens: {prompt_tokens + completion_tokens} (prompt {prompt_tokens}, completion {completion_tokens})")

    # summary
    avg_completion = sum(r["task_completion_rate"] for r in execution_history) / len(execution_history) if execution_history else 0
    avg_precision = sum(r["precision"] for r in execution_history) / len(execution_history) if execution_history else 0
    avg_recall = sum(r["recall"] for r in execution_history) / len(execution_history) if execution_history else 0
    avg_f1 = sum(r["f1_score"] for r in execution_history) / len(execution_history) if execution_history else 0

    enhanced_summary = {
        "benchmark_metadata": {
            "timestamp": timestamp,
            "version": "a2a_benchmark",
            "benchmark_path": benchmark_path,
            "buyer_endpoint": buyer_endpoint,
            "model_name": model_name,
            "clarify_policy": clarify_policy,
            "results_directory": str(results_dir),
            "task_count": len(tasks),
        },
        "summary": {
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "task_completion_rate": avg_completion,
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "f1_score": avg_f1,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "total_execution_time_seconds": total_execution_time,
        },
        "results": execution_history,
    }

    return enhanced_summary


def main():
    parser = argparse.ArgumentParser(description="Run A2A benchmark for WebMall task set.")
    parser.add_argument("--benchmark", default="task_sets/task_sets_35.json", help="Path to task set JSON.")
    parser.add_argument("--buyer", default=os.getenv("A2A_BUYER_ENDPOINT", "http://localhost:8005"), help="Buyer server base URL.")
    parser.add_argument("--model", default=os.getenv("A2A_MODEL", "gpt-5-mini"), help="Model name label for results dir.")
    parser.add_argument("--clarify", default=os.getenv("A2A_CLARIFY_POLICY", "on_once"), choices=["off", "on_once"], help="Clarification policy.")
    parser.add_argument("--timeout", type=int, default=int(os.getenv("A2A_TIMEOUT_SEC", "180")), help="HTTP timeout seconds per task.")
    args = parser.parse_args()

    summary = run_benchmark_a2a(
        benchmark_path=args.benchmark,
        buyer_endpoint=args.buyer,
        model_name=args.model,
        clarify_policy=args.clarify,
        timeout_sec=args.timeout,
        extra_payload=None,
    )

    # output files (match other scripts style)
    results_dir_str = safe_get(summary, ["benchmark_metadata", "results_directory"])
    results_dir = Path(results_dir_str) if results_dir_str else interface_results_dir(__file__, "a2a", args.model)

    ts = safe_get(summary, ["benchmark_metadata", "timestamp"], datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_json = results_dir / f"a2a_results_{ts}.json"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"📁 Results saved to: {out_json}")

    if summary.get("results"):
        generate_csv_metrics(summary, results_dir, prefix="a2a")


if __name__ == "__main__":
    main()
