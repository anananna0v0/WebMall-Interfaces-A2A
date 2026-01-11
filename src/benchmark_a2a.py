import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from utils import calculation_results, interface_results_dir
from a2a.main import initialize_system
from a2a.config import TASK_SET_PATH, WEBMALL_SHOPS

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("a2a_benchmark")

TARGET_MODEL = "gpt-5-mini"

def count_tokens(text: str) -> int:
    """Character-based token estimation to avoid tiktoken issues."""
    return len(text) // 4

class A2ABenchmark:
    def __init__(self, interface_name: str = "a2a-interface"):
        with open(TASK_SET_PATH, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        
        # URL mapping to resolve placeholders (e.g., {{URL_1}})
        url_map = {
            "{{URL_1}}": WEBMALL_SHOPS["webmall_1"]["url"],
            "{{URL_2}}": WEBMALL_SHOPS["webmall_2"]["url"],
            "{{URL_3}}": WEBMALL_SHOPS["webmall_3"]["url"],
            "{{URL_4}}": WEBMALL_SHOPS["webmall_4"]["url"]
        }

        self.all_tasks = []
        for cat in categories:
            for t in cat.get("tasks", []):
                # Resolve ground truth URLs
                raw_answers = t.get("correct_answer", {}).get("answers", [])
                resolved_answers = []
                for ans in raw_answers:
                    new_ans = ans
                    for placeholder, actual_url in url_map.items():
                        new_ans = new_ans.replace(placeholder, actual_url)
                    resolved_answers.append(new_ans)
                
                t["resolved_answers"] = list(set(resolved_answers))
                self.all_tasks.append(t)
        
        self.buyer_agent = initialize_system()
        self.results_dir = interface_results_dir(__file__, interface_name, TARGET_MODEL)
        self.metrics_summary = {"total_f1": 0.0, "total_cr": 0.0}
        self.task_logs = []

    def run_all_tasks(self):
        total_tasks = len(self.all_tasks)
        print(f"\n{'='*60}")
        print(f"Starting A2A Benchmark: {total_tasks} tasks | Model: {TARGET_MODEL}")
        print(f"{'='*60}\n")

        for i, task_entry in enumerate(self.all_tasks):
            task_id = task_entry.get("id")
            instruction = task_entry.get("task", "")
            ground_truth = task_entry.get("resolved_answers", [])

            print(f"--- Task [{i+1}/{total_tasks}]: {task_id} ---")
            start_time = time.time()
            
            try:
                # 1. Execute via BuyerAgent (which now performs LLM filtering)
                agent_output = self.buyer_agent.execute_procurement_task(instruction)
                results = agent_output.get("results", [])
                
                # 2. NORMALIZATION: Strip trailing slashes to ensure exact string matching
                predictions = [res["url"].rstrip('/') for res in results if "url" in res]
                normalized_gt = [gt.rstrip('/') for gt in ground_truth]
                
                # 3. CR Calculation: Binary Metric (1 if sets are identical, 0 otherwise)
                task_completion = 1 if set(predictions) == set(normalized_gt) and len(predictions) > 0 else 0
                
                # 4. F1 Calculation: Based on Precision and Recall
                metrics = calculation_results(normalized_gt, predictions)
                
                self.metrics_summary["total_f1"] += metrics['f1_score']
                self.metrics_summary["total_cr"] += task_completion
                
                # Real-time Terminal Feedback
                print(f"  > Search Results Found: {len(predictions)}")
                if predictions:
                    for idx, p_url in enumerate(predictions[:2]):
                        print(f"    - Pred {idx+1}: {p_url}")
                
                print(f"  > F1 Score: {metrics['f1_score']:.4f} | CR: {task_completion}")
                print(f"  > Latency: {time.time() - start_time:.2f}s\n")

                self.task_logs.append({
                    "task_id": task_id,
                    "metrics": metrics,
                    "prediction": predictions,
                    "ground_truth": normalized_gt,
                    "completion": task_completion
                })

            except Exception as e:
                print(f"  [X] Task Failed: {str(e)}\n")
                logger.error(f"Execution failed on {task_id}: {e}")

        self._finalize_report(total_tasks)

    def _finalize_report(self, total: int):
        avg_f1 = self.metrics_summary["total_f1"] / total if total > 0 else 0
        final_cr = (self.metrics_summary["total_cr"] / total * 100) if total > 0 else 0
        
        report = {
            "summary": {
                "model": TARGET_MODEL,
                "average_f1": avg_f1,
                "completion_rate_percentage": final_cr
            },
            "details": self.task_logs
        }

        output_file = self.results_dir / "a2a_benchmark_report.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        print(f"{'='*60}")
        print(f"FINAL RESULTS - {TARGET_MODEL}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Final CR:   {final_cr:.2f}%")
        print(f"Report:     {output_file}")
        print(f"{'='*60}")

if __name__ == "__main__":
    benchmark = A2ABenchmark()
    benchmark.run_all_tasks()