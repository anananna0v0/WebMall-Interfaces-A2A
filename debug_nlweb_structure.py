#!/usr/bin/env python3
import json
from pathlib import Path

def debug_nlweb_structure():
    """Debug the NLWeb Sonnet file structure to understand URL storage"""
    
    # Load the shortened example which should have full structure
    nlweb_file = Path('results/v1/shortened_logs/NLWeb_Agent_Claude_Sonnet_4_example.json')
    
    with open(nlweb_file, 'r') as f:
        data = json.load(f)
    
    print(f'Keys in main JSON: {list(data.keys())[:10]}')
    
    # Find a task with actual data
    sample_task = None
    for task_id, task_data in data.items():
        if task_id.startswith('task_') or 'task' in task_id.lower():
            sample_task = (task_id, task_data)
            break
    
    if sample_task:
        task_id, task_data = sample_task
        print(f'\n=== Sample Task: {task_id} ===')
        print(f'Keys in task_data: {list(task_data.keys())}')
        
        # Check tool_calls
        tool_calls = task_data.get('tool_calls', [])
        print(f'tool_calls length: {len(tool_calls)}')
        
        # Check response field 
        response = task_data.get('response', '')
        if response:
            response_str = str(response)
            print(f'Response field exists, length: {len(response_str)} chars')
            
            # Look for webmall URLs in the response
            webmall_count = response_str.lower().count('webmall')
            print(f'Found "webmall" {webmall_count} times in response')
            
            # Look for specific URL patterns
            if 'webmall-1.informatik.uni-mannheim.de' in response_str:
                print('Found webmall-1 URLs in response')
            if 'webmall-2.informatik.uni-mannheim.de' in response_str:
                print('Found webmall-2 URLs in response')
            
            # Show a sample of the response to understand structure
            print(f'\nFirst 500 chars of response:')
            print(response_str[:500])
    else:
        print('No task data found')

    # Now test our URL checking function
    print(f'\n=== Testing URL Checking Function ===')
    
    # Import our current function
    import sys
    sys.path.append('.')
    from error_analysis import check_url_in_tool_history
    
    if sample_task:
        task_id, task_data = sample_task
        test_url = 'https://webmall-1.informatik.uni-mannheim.de/product/amd-ryzen-9-5900x-3-7-ghz-12-cores-24-threads'
        
        print(f'Testing URL: {test_url}')
        result = check_url_in_tool_history(test_url, task_data)
        print(f'Current function result: {result}')
        
        # Manual check in response field
        response_str = str(task_data.get('response', ''))
        manual_result = test_url.rstrip('/') in response_str
        print(f'Manual check in response: {manual_result}')

if __name__ == '__main__':
    debug_nlweb_structure()