import json

def extract_tasks_hierarchical(input_file, output_name, tasks_per_category=5):
    """
    Extracts tasks while maintaining the Category -> Tasks hierarchy 
    required by benchmark_a2a.py.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    target_categories = [
        "WEBMALL_SUBSTITUTES_V1",
        "WEBMALL_CHEAPEST_PRODUCT_SEARCH_V1",
        "WEBMALL_CHEAPEST_BEST_FIT_SPECIFIC_V1",
        "WEBMALL_CHEAPEST_BEST_FIT_VAGUE_V1"
    ]

    new_dataset = []

    for category_block in data:
        if category_block["id"] in target_categories:
            # Create a copy of the category but only keep the first N tasks
            new_block = category_block.copy()
            original_tasks = category_block.get("tasks", [])
            new_block["tasks"] = original_tasks[:tasks_per_category]
            
            new_dataset.append(new_block)
            print(f"✅ Category {category_block['id']}: Kept {len(new_block['tasks'])} tasks.")

    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=2, ensure_ascii=False)

input_path = "task_sets.json"
extract_tasks_hierarchical(input_path, "experiment_tasks_5.json", 5)
extract_tasks_hierarchical(input_path, "test_tasks_1.json", 1)