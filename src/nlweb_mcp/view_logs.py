#!/usr/bin/env python3
"""
Simple log viewer for NLWeb MCP servers
"""

import os
import sys
import time
import glob
import argparse
from datetime import datetime

def tail_file(filepath, lines=20):
    """Read the last N lines from a file"""
    try:
        with open(filepath, 'r') as f:
            return f.readlines()[-lines:]
    except Exception as e:
        return [f"Error reading {filepath}: {e}\n"]

def follow_file(filepath):
    """Follow a file like tail -f"""
    try:
        with open(filepath, 'r') as f:
            # Go to end of file
            f.seek(0, os.SEEK_END)
            
            while True:
                line = f.readline()
                if line:
                    print(line.rstrip())
                    sys.stdout.flush()
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\nStopped following {filepath}")
    except Exception as e:
        print(f"Error following {filepath}: {e}")

def view_all_logs(tail_lines=20, follow=False):
    """View logs from all MCP servers"""
    log_pattern = "/tmp/nlweb_mcp_*.log"
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print("No log files found. Make sure servers are running with --debug flag.")
        print(f"Looking for files matching: {log_pattern}")
        return
    
    log_files.sort()
    
    if follow:
        if len(log_files) == 1:
            print(f"Following {log_files[0]}...")
            follow_file(log_files[0])
        else:
            print("Cannot follow multiple files. Please specify a single server with --server option.")
            print(f"Available log files: {', '.join(log_files)}")
    else:
        for log_file in log_files:
            server_name = os.path.basename(log_file).replace('nlweb_mcp_', '').replace('.log', '')
            print(f"\n{'='*60}")
            print(f"LOGS FOR {server_name.upper()} (last {tail_lines} lines)")
            print(f"{'='*60}")
            
            lines = tail_file(log_file, tail_lines)
            for line in lines:
                print(line.rstrip())

def main():
    parser = argparse.ArgumentParser(description='View NLWeb MCP server logs')
    parser.add_argument('--server', help='View logs for specific server (webmall_1, webmall_2, etc.)')
    parser.add_argument('--lines', '-n', type=int, default=20, help='Number of lines to show (default: 20)')
    parser.add_argument('--follow', '-f', action='store_true', help='Follow log output (like tail -f)')
    parser.add_argument('--all', action='store_true', help='Show logs from all servers')
    
    args = parser.parse_args()
    
    if args.server:
        log_file = f"/tmp/nlweb_mcp_{args.server}.log"
        if not os.path.exists(log_file):
            print(f"Log file not found: {log_file}")
            print("Make sure the server is running with --debug flag.")
            sys.exit(1)
        
        if args.follow:
            print(f"Following {log_file}...")
            follow_file(log_file)
        else:
            print(f"Last {args.lines} lines from {args.server}:")
            print("-" * 40)
            lines = tail_file(log_file, args.lines)
            for line in lines:
                print(line.rstrip())
    else:
        view_all_logs(args.lines, args.follow)

if __name__ == "__main__":
    main()