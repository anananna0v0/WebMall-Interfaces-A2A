#!/usr/bin/env python3
import json
import csv
from pathlib import Path

def debug_rag_structure():
    """Debug RAG GPT-4.1 missed URLs to see if they're in tool_output_raw"""
    
    # Get a few RAG GPT-4.1 missed URLs from the current CSV
    csv_file = 'error_analysis_20250821_144436.csv'
    rag_missed = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row['agent_type'] == 'RAG' and 
                row['model'] == 'GPT-4.1' and 
                row['type'] == 'missed' and 
                row['seen_in_history'] == 'No'):
                rag_missed.append(row)
                if len(rag_missed) >= 3:  # Just need a few examples
                    break
    
    print(f"Found {len(rag_missed)} RAG GPT-4.1 missed URLs marked as 'No'")
    
    if not rag_missed:
        print("No RAG GPT-4.1 missed URLs found")
        return
    
    # Load the RAG GPT-4.1 results file
    rag_file = Path('results/v1/rag/gpt.41/benchmark_v2_improved_results_20250722_231047.json')
    with open(rag_file, 'r') as f:
        rag_data = json.load(f)
    
    # Check each missed URL
    for i, missed_entry in enumerate(rag_missed):
        print(f"\n=== Missed URL {i+1} ===")
        task_id = missed_entry['task_id']
        missed_url = missed_entry['url']
        print(f"Task: {task_id}")
        print(f"URL: {missed_url}")
        
        # Find the task in the results
        task_found = False
        for result in rag_data.get('results', []):
            if result.get('task_id') == task_id:
                task_found = True
                print(f"✓ Found task in RAG results")
                
                # Check current function result
                from error_analysis import check_url_in_tool_history
                current_result = check_url_in_tool_history(missed_url, result)
                print(f"Current function result: {current_result}")
                
                # Manual check in tool_history
                tool_history = result.get('tool_history', [])
                manual_found_in_history = False
                
                for tool_call in tool_history:
                    tool_output_raw = str(tool_call.get('tool_output_raw', ''))
                    if missed_url.rstrip('/') in tool_output_raw:
                        manual_found_in_history = True
                        print(f"✓ URL found in tool_output_raw!")
                        # Show a snippet
                        idx = tool_output_raw.find(missed_url.rstrip('/'))
                        snippet = tool_output_raw[max(0, idx-50):idx+len(missed_url)+50]
                        print(f"  Snippet: ...{snippet}...")
                        break
                
                if not manual_found_in_history:
                    print("✗ URL not found in any tool_output_raw")
                    print(f"  Tool history has {len(tool_history)} entries")
                    
                    # Check what fields are available in tool_history
                    if tool_history:
                        print(f"  First tool call keys: {list(tool_history[0].keys())}")
                
                break
        
        if not task_found:
            print("✗ Task not found in RAG results")

if __name__ == '__main__':
    debug_rag_structure()