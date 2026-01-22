#!/bin/bash

# 1. Cleanup old processes
pkill -f shop_agent.py
sleep 2

# Set absolute PYTHONPATH to the 'src' directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "Starting 4 Shop Agents..."
for i in {1..4}
do
    # Run shops with explicitly set PYTHONPATH
    python3 src/a2a/shop_agent.py --shop_id webmall_$i --port 800$i & 
done

echo "Waiting 20s for all shops to be ready..."
sleep 20

echo "🚀 Starting 4-task Benchmark..."
python3 src/benchmark_a2a.py

echo "Cleaning up processes..."
pkill -f shop_agent.py
echo "✅ Done."