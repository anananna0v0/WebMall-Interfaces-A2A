import json

def extract_a2a_tasks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Mapping requested categories to JSON indices
    # Index 8: Find Substitutes (6 tasks)
    # Index 1: Find Cheapest Offer (10 tasks)
    # Index 4: Cheapest Offer Specific Requirements (10 tasks)
    # Index 5: Cheapest Offer Vague Requirements (6 tasks)
    target_mapping = {
        "Find Substitutes": 8,
        "Find Cheapest Offer": 1,
        "Cheapest Offer Specific Requirements": 4,
        "Cheapest Offer Vague Requirements": 5
    }
    
    selected_tasks = []
    for category_name, index in target_mapping.items():
        tasks = data[index].get('tasks', [])
        # Add category tag to each task for later analysis
        for t in tasks:
            t['experiment_category'] = category_name
        selected_tasks.extend(tasks)
        print(f"Extracted {len(tasks)} tasks from: {category_name}")
        
    return selected_tasks

# Usage
tasks = extract_a2a_tasks('task_sets.json')
print(f"Total tasks for experiment: {len(tasks)}")

# Save to a new file for your experiment
with open('experiment_tasks_32.json', 'w', encoding='utf-8') as f:
    json.dump(tasks, f, indent=2, ensure_ascii=False)