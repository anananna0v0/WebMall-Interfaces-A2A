#!/usr/bin/env python3
import json
import csv
from pathlib import Path

def debug_rag_detailed():
    """Examine RAG tool_output field vs our current checking"""
    
    # Get one RAG GPT-4.1 missed URL
    csv_file = 'error_analysis_20250821_144436.csv'
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['agent_type'] == 'RAG' and 
                row['model'] == 'GPT-4.1' and 
                row['type'] == 'missed' and 
                row['seen_in_history'] == 'No'):
                task_id = row['task_id']
                missed_url = row['url']
                break
    
    print(f"Debugging Task: {task_id}")
    print(f"Missed URL: {missed_url}")
    
    # Load RAG data
    rag_file = Path('results/v1/rag/gpt.41/benchmark_v2_improved_results_20250722_231047.json')
    with open(rag_file, 'r') as f:
        rag_data = json.load(f)
    
    # Find the task
    for result in rag_data.get('results', []):
        if result.get('task_id') == task_id:
            print(f"\n=== Task Data Structure ===")
            print(f"Keys: {list(result.keys())}")
            
            tool_history = result.get('tool_history', [])
            print(f"Tool history entries: {len(tool_history)}")
            
            for i, tool_call in enumerate(tool_history):
                print(f"\n--- Tool Call {i} ---")
                print(f"Keys: {list(tool_call.keys())}")
                print(f"Tool name: {tool_call.get('tool_name')}")
                print(f"Tool type: {tool_call.get('tool_type')}")
                
                # Check different fields for the URL
                tool_output = tool_call.get('tool_output', '')
                tool_args = str(tool_call.get('tool_args', ''))
                full_str = str(tool_call)
                
                url_in_output = missed_url.rstrip('/') in str(tool_output)
                url_in_args = missed_url.rstrip('/') in tool_args  
                url_in_full = missed_url.rstrip('/') in full_str
                
                print(f"URL in tool_output: {url_in_output}")
                print(f"URL in tool_args: {url_in_args}")
                print(f"URL in full string: {url_in_full}")
                
                if url_in_output:
                    print("✓ Found in tool_output!")
                    # Show snippet
                    output_str = str(tool_output)
                    idx = output_str.find(missed_url.rstrip('/'))
                    snippet = output_str[max(0, idx-100):idx+len(missed_url)+100]
                    print(f"Snippet: ...{snippet}...")
                
                if url_in_full and not url_in_output:
                    print("Found in full string but not tool_output - investigating...")
                    # Show where it was found
                    idx = full_str.find(missed_url.rstrip('/'))
                    snippet = full_str[max(0, idx-100):idx+len(missed_url)+100] 
                    print(f"Full string snippet: ...{snippet}...")
            
            break

if __name__ == '__main__':
    debug_rag_detailed()