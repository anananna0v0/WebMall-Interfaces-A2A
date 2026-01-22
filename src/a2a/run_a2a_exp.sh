#!/bin/bash

# 1. Force cleanup old processes to avoid duplicates
pkill -f shop_agent.py
sleep 2

echo "Starting 4 Shop Agents..."
for i in {1..4}
do
    # Use '&' to run shops in background
    python3 src/a2a/shop_agent.py --shop_id webmall_$i --port 800$i & 
done

# 2. Crucial: Wait for all shops to fully initialize before starting benchmark
echo "Waiting 20s for all shops to be ready..."
sleep 20

# 3. RUN BENCHMARK ONLY ONCE (Must be outside the for-loop)
echo "🚀 Starting 4-task Benchmark..."
python3 src/benchmark_a2a.py

# 4. Final Cleanup
echo "Cleaning up processes..."
pkill -f shop_agent.py
echo "✅ Done."