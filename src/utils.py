
from pathlib import Path
import re
from typing import Optional, Any


def calculation_results(benchmark_solutions, model_solution):
    """
    Calculate task completion, precision, and recall metrics.

    Args:
        benchmark_solutions: List of strings containing benchmark solutions
        model_solution: List of strings containing model solution

    Returns:
        dict: Contains task_completion_rate, avg_precision, avg_recall, f1_score
    """
    # Convert lists to sets of complete strings, not individual characters
    if isinstance(benchmark_solutions, list):
        benchmark_set = set(benchmark_solutions)
    elif isinstance(benchmark_solutions, set):
        benchmark_set = benchmark_solutions
    else:
        benchmark_set = set([str(benchmark_solutions)])

    if isinstance(model_solution, list):
        model_set = set(model_solution)
    elif isinstance(model_solution, set):
        model_set = model_solution
    else:
        model_set = set([str(model_solution)])

    # Task completion: 1 if exact match, 0 otherwise
    task_completion = 1 if benchmark_set == model_set else 0

    # Precision: intersection / model_set size
    if len(model_set) > 0:
        precision = len(benchmark_set.intersection(model_set)) / len(model_set)
    else:
        precision = 0.0

    # Recall: intersection / benchmark_set size
    if len(benchmark_set) > 0:
        recall = len(benchmark_set.intersection(
            model_set)) / len(benchmark_set)
    else:
        recall = 0.0

    # Calculate F1 score with zero division protection
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return {
        'task_completion_rate': task_completion,
        'avg_precision': precision,
        'avg_recall': recall,
        'f1_score': f1_score
    }


def _find_repo_root(start_file: str) -> Path:
    """Return project root by locating the src directory."""
    current_path = Path(start_file).resolve()
    for parent in current_path.parents:
        if parent.name == "src":
            return parent.parent
    return current_path.parents[-1]


def _slugify_for_path(value: str) -> str:
    """Sanitize strings for safe filesystem paths."""
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '-', value.strip())
    cleaned = re.sub(r'-{2,}', '-', cleaned)
    cleaned = cleaned.strip('-')
    return cleaned or "model"


def interface_results_dir(
    start_file: str,
    interface_name: str,
    model_name: str,
    reasoning_effort: Optional[str] = None
) -> Path:
    """Build and create the results directory for interface benchmarks."""
    repo_root = _find_repo_root(start_file)

    interface_part = _slugify_for_path(interface_name) or "interface"
    base_dir = repo_root / "results" / interface_part

    model_part = _slugify_for_path(model_name)
    if reasoning_effort:
        reasoning_part = _slugify_for_path(str(reasoning_effort))
        if reasoning_part:
            model_part = f"{model_part}-{reasoning_part}"

    results_dir = base_dir / model_part
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def extract_reasoning_effort(chat_model: Any) -> Optional[str]:
    """Best-effort extraction of reasoning effort metadata from a chat model."""
    if chat_model is None:
        return None

    possible_keys = ("reasoning_effort", "reasoning", "reasoning_mode")

    for attr in possible_keys:
        value = getattr(chat_model, attr, None)
        if value:
            return str(value)

    model_kwargs = getattr(chat_model, "model_kwargs", None)
    if isinstance(model_kwargs, dict):
        for key in possible_keys:
            value = model_kwargs.get(key)
            if value:
                return str(value)

    return None


def resolve_model_name(chat_model: Any, default: Optional[str] = None) -> str:
    """Resolve a readable model name from a chat model or fallback to default."""
    if isinstance(chat_model, str):
        return chat_model

    for attr in ("model_name", "model", "name"):
        value = getattr(chat_model, attr, None)
        if value:
            return str(value)

    if default:
        return default

    if chat_model is None:
        return "unknown-model"

    return chat_model.__class__.__name__
