#!/usr/bin/env python3
import json
import csv
from pathlib import Path

def test_rag_fix():
    """Test if the RAG fix improves detection rates"""
    
    # Get several RAG GPT-4.1 missed URLs that are marked as NOT SEEN
    csv_file = 'error_analysis_20250821_144436.csv'
    not_seen_examples = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['agent_type'] == 'RAG' and 
                row['model'] == 'GPT-4.1' and 
                row['seen_in_history'] == 'No'):
                not_seen_examples.append(row)
                if len(not_seen_examples) >= 10:  # Test 10 examples
                    break
    
    print(f"Testing {len(not_seen_examples)} RAG GPT-4.1 'Not Seen' examples...")
    
    # Load RAG data
    rag_file = Path('results/v1/rag/gpt.41/benchmark_v2_improved_results_20250722_231047.json')
    with open(rag_file, 'r') as f:
        rag_data = json.load(f)
    
    # Test each example with the new function
    improved_count = 0
    
    for i, example in enumerate(not_seen_examples):
        task_id = example['task_id']
        url = example['url']
        
        # Find the task in RAG data
        for result in rag_data.get('results', []):
            if result.get('task_id') == task_id:
                # Test with fixed function
                from error_analysis import check_url_in_tool_history
                new_result = check_url_in_tool_history(url, result)
                
                if new_result:
                    improved_count += 1
                    print(f"✓ Example {i+1}: Now FOUND - {task_id}")
                else:
                    print(f"✗ Example {i+1}: Still not found - {task_id}")
                break
    
    print(f"\n=== Results ===")
    print(f"Before fix: {len(not_seen_examples)} URLs marked as 'Not Seen'")
    print(f"After fix: {improved_count} would now be marked as 'Seen'")
    print(f"Improvement rate: {improved_count}/{len(not_seen_examples)} = {improved_count/len(not_seen_examples)*100:.1f}%")
    
    if improved_count > 0:
        print(f"\nThe fix should improve RAG GPT-4.1 from ~52% to ~{52 + (improved_count/len(not_seen_examples)*48):.1f}%")

if __name__ == '__main__':
    test_rag_fix()