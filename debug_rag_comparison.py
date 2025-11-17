#!/usr/bin/env python3
import json
import csv
from pathlib import Path

def debug_rag_comparison():
    """Compare RAG URLs that were seen vs not seen"""
    
    # Get both seen and not seen examples
    csv_file = 'error_analysis_20250821_144436.csv'
    seen_example = None
    not_seen_example = None
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['agent_type'] == 'RAG' and 
                row['model'] == 'GPT-4.1'):
                if row['seen_in_history'] == 'Yes' and not seen_example:
                    seen_example = row
                elif row['seen_in_history'] == 'No' and not not_seen_example:
                    not_seen_example = row
                
                if seen_example and not_seen_example:
                    break
    
    print("=== Comparison: Seen vs Not Seen ===")
    print(f"SEEN example: {seen_example['task_id']} - {seen_example['url'][:80]}...")
    print(f"NOT SEEN example: {not_seen_example['task_id']} - {not_seen_example['url'][:80]}...")
    
    # Load RAG data
    rag_file = Path('results/v1/rag/gpt.41/benchmark_v2_improved_results_20250722_231047.json')
    with open(rag_file, 'r') as f:
        rag_data = json.load(f)
    
    # Check both examples
    examples = [
        ("SEEN", seen_example),
        ("NOT SEEN", not_seen_example)
    ]
    
    for label, example in examples:
        print(f"\n=== {label} Example ===")
        task_id = example['task_id'] 
        url = example['url']
        
        for result in rag_data.get('results', []):
            if result.get('task_id') == task_id:
                print(f"Task: {task_id}")
                
                # Test current function
                from error_analysis import check_url_in_tool_history
                current_result = check_url_in_tool_history(url, result)
                print(f"Current function: {current_result}")
                
                # Manual checks
                tool_history = result.get('tool_history', [])
                print(f"Tool history entries: {len(tool_history)}")
                
                found_in_tool_output = False
                found_in_tool_output_raw = False
                found_in_full_string = False
                
                for i, tool_call in enumerate(tool_history):
                    # Check tool_output
                    tool_output = str(tool_call.get('tool_output', ''))
                    if url.rstrip('/') in tool_output:
                        found_in_tool_output = True
                        print(f"  ✓ Found in tool_output of call {i}")
                    
                    # Check tool_output_raw  
                    tool_output_raw = str(tool_call.get('tool_output_raw', ''))
                    if url.rstrip('/') in tool_output_raw:
                        found_in_tool_output_raw = True
                        print(f"  ✓ Found in tool_output_raw of call {i}")
                    
                    # Check full string
                    full_str = str(tool_call)
                    if url.rstrip('/') in full_str:
                        found_in_full_string = True
                        print(f"  ✓ Found in full string of call {i}")
                
                print(f"Summary - tool_output: {found_in_tool_output}, tool_output_raw: {found_in_tool_output_raw}, full_string: {found_in_full_string}")
                break
    
    print(f"\n=== Analysis ===")
    print("If SEEN example shows URLs in tool_output_raw but NOT SEEN doesn't,")
    print("then we need to check tool_output_raw field in addition to current checks")

if __name__ == '__main__':
    debug_rag_comparison()