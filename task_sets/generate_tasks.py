import json

# Define the 7 target categories aligned with the paper
target_categories = [
    "Specific_Product",
    "Cheapest_Product",
    "Best_Fit_Specific",
    "Best_Fit_Vague",
    "Cheapest_Best_Fit_Specific",
    "Cheapest_Best_Fit_Vague",
    "Compatible_Products",
]

def generate():
    try:
        # Read your local task_sets.json
        with open('task_sets.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        new_task_sets = []

        for category_name in target_categories:
            for category_node in data:
                # Check category consistency within the task set
                if category_node['tasks'] and category_node['tasks'][0].get('category') == category_name:
                    # Select exactly 5 tasks to reduce statistical noise
                    limited_tasks = category_node['tasks'][:5]
                    new_node = category_node.copy()
                    new_node['tasks'] = limited_tasks
                    new_task_sets.append(new_node)
                    print(f"Added {category_name}: {len(limited_tasks)} tasks")
                    break

        # Save the optimized 35-task set
        with open('task_sets_35.json', 'w', encoding='utf-8') as f:
            json.dump(new_task_sets, f, indent=2, ensure_ascii=False)
        
        print("\nSuccess: 'task_sets_35.json' created locally.")

    except FileNotFoundError:
        print("Error: 'task_sets.json' not found in current directory.")

if __name__ == "__main__":
    generate()