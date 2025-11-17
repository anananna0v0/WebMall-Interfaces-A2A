#!/usr/bin/env python3
import json
from pathlib import Path

def analyze_nlweb_sonnet():
    # Load the NLWeb Sonnet file
    nlweb_file = Path('results/v1/nlweb/sonnet/nlweb_mcp_enhanced_results_20250724_133510.json')
    
    with open(nlweb_file, 'r') as f:
        data = json.load(f)
    
    print(f'Analyzing: {nlweb_file.name}')
    print(f'Total entries: {len(data)}')
    print()
    
    # Find actual tasks (skip metadata)
    task_count = 0
    tasks_with_urls = 0
    total_urls = 0
    
    for task_id, task_data in data.items():
        if not task_id.startswith('task_'):
            continue
            
        task_count += 1
        
        if task_count <= 3:  # Show first 3 tasks in detail
            print(f'\n=== {task_id} ===')
            print(f'Keys: {list(task_data.keys())[:10]}...')  # Show first 10 keys
            
        if 'tool_calls' in task_data:
            task_urls = set()
            for tc in task_data['tool_calls']:
                # Check urls_extracted field
                if 'urls_extracted' in tc and tc['urls_extracted']:
                    urls = tc['urls_extracted']
                    task_urls.update(urls)
                    if task_count <= 3:
                        print(f'  Found {len(urls)} URLs in urls_extracted')
                        print(f'    Sample: {urls[0] if urls else "None"}')
                
                # Also check tool_output_raw
                if 'tool_output_raw' in tc and tc['tool_output_raw']:
                    output = str(tc['tool_output_raw'])
                    # Look for webmall URLs in the output
                    if 'webmall' in output.lower():
                        if task_count <= 3:
                            print(f'  Found "webmall" in tool_output_raw')
            
            if task_urls:
                tasks_with_urls += 1
                total_urls += len(task_urls)
    
    print(f'\n=== Summary ===')
    print(f'Total tasks: {task_count}')
    print(f'Tasks with URLs extracted: {tasks_with_urls}')
    print(f'Total unique URLs found: {total_urls}')
    
    # Now check the CSV to see what we're comparing
    csv_file = Path('error_analysis_20250820_091820.csv')
    if csv_file.exists():
        import csv
        print(f'\n=== Checking CSV errors for NLWeb Sonnet ===')
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            nlweb_sonnet_errors = []
            for row in reader:
                if row['agent_type'] == 'nlweb' and row['model'] == 'sonnet':
                    nlweb_sonnet_errors.append(row)
        
        print(f'Found {len(nlweb_sonnet_errors)} NLWeb Sonnet errors in CSV')
        
        # Check first few errors
        for i, error in enumerate(nlweb_sonnet_errors[:5]):
            print(f'\nError {i+1}:')
            print(f'  Task: {error["task_id"]}')
            print(f'  Type: {error["type"]}')
            print(f'  URL: {error["url"]}')
            print(f'  Seen in history: {error["seen_in_history"]}')

if __name__ == '__main__':
    analyze_nlweb_sonnet()