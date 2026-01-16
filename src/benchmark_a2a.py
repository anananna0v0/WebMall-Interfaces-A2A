import json
import logging
import time
import datetime
from pathlib import Path
from utils import calculation_results, interface_results_dir
from a2a.main import initialize_system
from a2a.config import TASK_SET_PATH, WEBMALL_SHOPS

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("a2a_benchmark")

TARGET_MODEL = "gpt-5-mini"

def count_tokens(text: str) -> int:
    """
    Estimate tokens based on character length (1 token approx 4 chars).
    """
    return len(text) // 4 if text else 0

class A2ABenchmark:
    def __init__(self, interface_name: str = "a2a-interface"):
        # Load task sets
        with open(TASK_SET_PATH, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        
        # Mapping for URL placeholders
        url_map = {"{{URL_"+str(i)+"}}": WEBMALL_SHOPS[f"webmall_{i}"]["url"] for i in range(1, 5)}
        self.all_tasks = []
        for cat in categories:
            for t in cat.get("tasks", []):
                raw_ans = t.get("correct_answer", {}).get("answers", [])
                resolved = [ans.replace(k, v) for ans in raw_ans for k, v in url_map.items() if k in ans]
                t["resolved_answers"] = list(set(resolved if resolved else raw_ans))
                self.all_tasks.append(t)
        
        # System initialization
        self.buyer_agent = initialize_system()
        self.results_dir = interface_results_dir(__file__, interface_name, TARGET_MODEL)
        
        # Metrics summary with corrected keys
        self.metrics_summary = {
            "total_f1": 0.0, 
            "total_cr": 0.0, 
            "total_input_tokens": 0, 
            "total_output_tokens": 0
        }
        self.task_logs = []

    def run_all_tasks(self):
        total_tasks = len(self.all_tasks)
        print(f"\n{'='*80}")
        print(f"STARTING BENCHMARK: {total_tasks} TASKS | MODEL: {TARGET_MODEL}")
        print(f"{'='*80}\n")

        for i, task in enumerate(self.all_tasks):
            task_id = task.get("id")
            instruction = task.get("task", "")
            gt_urls = [url.rstrip('/') for url in task["resolved_answers"]]
            
            # 1. External Input Token tracking
            ext_in = count_tokens(instruction)
            
            print(f"[{i+1}/{total_tasks}] Processing Task: {task_id}")
            start_time = time.time()
            
            try:
                # 2. Agent execution
                agent_output = self.buyer_agent.execute_procurement_task(instruction)
                results = agent_output.get("results", [])
                internal = agent_output.get("internal_usage", {"in": 0, "out": 0})
                
                # Normalize predictions
                preds = [res["url"].rstrip('/') for res in results if "url" in res]
                
                # 3. Aggregate all tokens
                t_in = ext_in + internal["in"]
                t_out = internal["out"] + count_tokens(json.dumps(results))
                
                # 4. Indicators calculation
                t_cr = 1 if set(preds) == set(gt_urls) and len(preds) > 0 else 0
                calc = calculation_results(gt_urls, preds)
                
                # Accumulate summary
                self.metrics_summary["total_f1"] += calc['f1_score']
                self.metrics_summary["total_cr"] += t_cr
                self.metrics_summary["total_input_tokens"] += t_in
                self.metrics_summary["total_output_tokens"] += t_out
                
                # 5. Terminal Feedback
                print(f"    - F1 Score: {calc['f1_score']:.4f}")
                print(f"    - CR:       {t_cr}")
                print(f"    - Tokens:   In {t_in} / Out {t_out}")
                print(f"    - Latency:  {time.time() - start_time:.2f}s\n")

                self.task_logs.append({
                    "task_id": task_id,
                    "f1": calc['f1_score'],
                    "cr": t_cr,
                    "tokens": {"input": t_in, "output": t_out},
                    "prediction": preds,
                    "ground_truth": gt_urls
                })
                
            except Exception as e:
                print(f"    [!] Error: {str(e)}\n")

        self._finalize_report(total_tasks)

    def _finalize_report(self, total: int):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"a2a_full_report_{timestamp}.json"
        
        avg_f1 = self.metrics_summary["total_f1"] / total
        cr_pct = (self.metrics_summary["total_cr"] / total) * 100
        
        final_results = {
            "summary": {
                "model": TARGET_MODEL,
                "avg_f1": avg_f1,
                "cr_percentage": cr_pct,
                "total_input_tokens": self.metrics_summary["total_input_tokens"],
                "total_output_tokens": self.metrics_summary["total_output_tokens"]
            },
            "details": self.task_logs
        }
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)
            
        print(f"{'='*80}")
        print(f"FINAL SUMMARY - {TARGET_MODEL}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Final CR:   {cr_pct:.2f}%")
        print(f"Total In Tokens:  {self.metrics_summary['total_input_tokens']}")
        print(f"Total Out Tokens: {self.metrics_summary['total_output_tokens']}")
        print(f"Report: {report_file}")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    A2ABenchmark().run_all_tasks()