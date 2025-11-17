#!/usr/bin/env python3
import json
import csv
from pathlib import Path
from datetime import datetime

def extract_task_info(task_data):
    """Extract task information from a result entry."""
    task_id = task_data.get('task_id', 'unknown')
    task_category = task_data.get('task_category', 'unknown')
    task_description = task_data.get('task', task_data.get('user_task', '')).replace('\n', ' ').strip()
    
    return task_id, task_category, task_description

def check_url_in_tool_history(url, task_data):
    """Check if a URL appears anywhere in the tool history."""
    # Normalize URL by removing trailing slashes for comparison
    normalized_url = url.rstrip('/')
    
    # Check tool_history field (used by RAG agents)
    tool_history = task_data.get('tool_history', [])
    for tool_call in tool_history:
        # Check tool_output field
        tool_output = str(tool_call.get('tool_output', ''))
        if normalized_url in tool_output:
            return True
        
        # Check tool_output_raw field (RAG agents store URLs here)
        tool_output_raw = str(tool_call.get('tool_output_raw', ''))
        if normalized_url in tool_output_raw:
            return True
        
        # Fallback: check full tool_call string
        tool_output_str = str(tool_call)
        if normalized_url in tool_output_str:
            return True
    
    # Check tool_calls field (used by MCP and NLWeb agents)
    tool_calls = task_data.get('tool_calls', [])
    for tool_call in tool_calls:
        # Check in response content (MCP style)
        response_str = str(tool_call.get('response', ''))
        if normalized_url in response_str:
            return True
        
        # Check in urls_extracted field (NLWeb style)
        urls_extracted = tool_call.get('urls_extracted', [])
        for extracted_url in urls_extracted:
            if normalized_url == extracted_url.rstrip('/'):
                return True
        
        # Also check tool_output_raw (NLWeb may store URLs here)
        tool_output = str(tool_call.get('tool_output_raw', ''))
        if normalized_url in tool_output:
            return True
    
    # Check hybrid_correct_retrieved and hybrid_additional_retrieved (MCP specific)
    hybrid_correct = task_data.get('hybrid_correct_retrieved', [])
    for h_url in hybrid_correct:
        if normalized_url == h_url.rstrip('/'):
            return True
    
    hybrid_additional = task_data.get('hybrid_additional_retrieved', [])
    for h_url in hybrid_additional:
        if normalized_url == h_url.rstrip('/'):
            return True
    
    # Check rag_exact_url_matches (RAG specific)
    rag_matches = task_data.get('rag_exact_url_matches', [])
    for r_url in rag_matches:
        if normalized_url == r_url.rstrip('/'):
            return True
    
    # Check v2_url_ranks (RAG specific - contains URLs that were ranked)
    v2_ranks = task_data.get('v2_url_ranks', {})
    for r_url in v2_ranks.keys():
        if normalized_url == r_url.rstrip('/'):
            return True
    
    # Check response field (NLWeb agents store full interaction here)
    response = task_data.get('response', '')
    if response:
        response_str = str(response)
        if normalized_url in response_str:
            return True
    
    return False

def process_json_file(filepath, agent_type, model):
    """Process a single JSON result file and extract error information."""
    errors = []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        for task_data in results:
            task_id, task_category, task_description = extract_task_info(task_data)
            
            # Get missing URLs (ground truth URLs not found by model)
            missing_urls = task_data.get('missing_urls', [])
            for url in missing_urls:
                # Check if URL was seen in tool history
                was_seen = check_url_in_tool_history(url, task_data)
                
                errors.append({
                    'task_id': task_id,
                    'task_category': task_category,
                    'task_description': task_description,
                    'type': 'missed',
                    'url': url,
                    'seen_in_history': 'Yes' if was_seen else 'No',
                    'model': model,
                    'agent_type': agent_type
                })
            
            # Get additional URLs (model found but not in ground truth)
            additional_urls = task_data.get('additional_urls', [])
            for url in additional_urls:
                # Check if additional URL was seen in tool history
                was_seen = check_url_in_tool_history(url, task_data)
                
                errors.append({
                    'task_id': task_id,
                    'task_category': task_category,
                    'task_description': task_description,
                    'type': 'additional',
                    'url': url,
                    'seen_in_history': 'Yes' if was_seen else 'No',
                    'model': model,
                    'agent_type': agent_type
                })
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
    
    return errors

def main():
    results_dir = Path('/Users/aaronsteiner/Documents/GitHub/WebMall-Interfaces/results/v1')
    all_errors = []
    
    # Define the file patterns to process
    files_to_process = [
        # RAG Agent
        ('rag/gpt.41/benchmark_v2_improved_results_20250722_231047.json', 'RAG', 'GPT-4.1'),
        ('rag/sonnet/benchmark_v2_improved_results_20250723_173336.json', 'RAG', 'Sonnet'),
        
        # NLWeb Agent
        ('nlweb/gpt.41/nlweb_mcp_enhanced_results_20250724_143245.json', 'NLWeb', 'GPT-4.1'),
        ('nlweb/sonnet/nlweb_mcp_enhanced_results_20250724_133510.json', 'NLWeb', 'Sonnet'),
        ('nlweb/sonnet/nlweb_mcp_enhanced_results_20250724_135630.json', 'NLWeb', 'Sonnet'),
        
        # MCP Agent
        ('mcp/gpt4.1/hybrid_execution_history_20250725_104451.json', 'MCP', 'GPT-4.1'),
        ('mcp/sonnet/hybrid_execution_history_20250725_144242.json', 'MCP', 'Sonnet'),
    ]
    
    # Process each file
    for relative_path, agent_type, model in files_to_process:
        filepath = results_dir / relative_path
        if filepath.exists():
            print(f"Processing {relative_path}...")
            errors = process_json_file(filepath, agent_type, model)
            all_errors.extend(errors)
            print(f"  Found {len(errors)} errors")
        else:
            print(f"Warning: File not found - {filepath}")
    
    # Sort errors by task_id and then by type (missed first, then additional)
    all_errors.sort(key=lambda x: (x['task_id'], x['type'] == 'additional', x['url']))
    
    # Generate timestamp for output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'error_analysis_{timestamp}.csv'
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['task_id', 'task_category', 'type', 'url', 'seen_in_history', 'model', 'agent_type', 'task_description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for error in all_errors:
            writer.writerow(error)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ERROR ANALYSIS SUMMARY")
    print("="*60)
    
    # Count errors by type
    missed_count = sum(1 for e in all_errors if e['type'] == 'missed')
    additional_count = sum(1 for e in all_errors if e['type'] == 'additional')
    
    # Count missed URLs that were seen vs not seen
    missed_seen = sum(1 for e in all_errors if e['type'] == 'missed' and e['seen_in_history'] == 'Yes')
    missed_not_seen = sum(1 for e in all_errors if e['type'] == 'missed' and e['seen_in_history'] == 'No')
    
    # Count additional URLs that were seen vs not seen
    additional_seen = sum(1 for e in all_errors if e['type'] == 'additional' and e['seen_in_history'] == 'Yes')
    additional_not_seen = sum(1 for e in all_errors if e['type'] == 'additional' and e['seen_in_history'] == 'No')
    
    print(f"Total errors found: {len(all_errors)}")
    print(f"  - Missed URLs: {missed_count}")
    print(f"    • Seen in tool history: {missed_seen}")
    print(f"    • Never seen: {missed_not_seen}")
    print(f"  - Additional URLs: {additional_count}")
    print(f"    • Seen in tool history: {additional_seen}")
    print(f"    • Never seen: {additional_not_seen}")
    
    # Count errors by agent type
    print("\nErrors by Agent Type:")
    for agent in ['RAG', 'NLWeb', 'MCP']:
        agent_errors = [e for e in all_errors if e['agent_type'] == agent]
        if agent_errors:
            missed = sum(1 for e in agent_errors if e['type'] == 'missed')
            missed_seen_agent = sum(1 for e in agent_errors if e['type'] == 'missed' and e['seen_in_history'] == 'Yes')
            additional = sum(1 for e in agent_errors if e['type'] == 'additional')
            additional_seen_agent = sum(1 for e in agent_errors if e['type'] == 'additional' and e['seen_in_history'] == 'Yes')
            print(f"  {agent}: {len(agent_errors)} total")
            print(f"    - Missed: {missed} (seen: {missed_seen_agent}, not seen: {missed - missed_seen_agent})")
            print(f"    - Additional: {additional} (seen: {additional_seen_agent}, not seen: {additional - additional_seen_agent})")
    
    # Count errors by model
    print("\nErrors by Model:")
    for model in ['GPT-4.1', 'Sonnet']:
        model_errors = [e for e in all_errors if e['model'] == model]
        if model_errors:
            missed = sum(1 for e in model_errors if e['type'] == 'missed')
            additional = sum(1 for e in model_errors if e['type'] == 'additional')
            print(f"  {model}: {len(model_errors)} total (missed: {missed}, additional: {additional})")
    
    print(f"\nResults saved to: {output_file}")
    print(f"You can now review the CSV file to identify error patterns.")

if __name__ == '__main__':
    main()