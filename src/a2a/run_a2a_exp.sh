#!/bin/bash

# --- Path Configuration ---
# Set paths relative to script location [cite: 1]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"

# Ensure PYTHONPATH points to the 'src' directory [cite: 1]
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/src

# --- Pre-flight Checks ---
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "❌ Error: .env file missing in $PROJECT_ROOT"
    exit 1
fi

# 1. Cleanup old processes [cite: 1]
echo "🧹 Cleaning up old shop_agent processes..."
pkill -f shop_agent.py
sleep 2

# 2. Start Shop Agents [cite: 1]
echo "🚀 Starting 4 Shop Agents..."
for i in {1..4}
do
    # Direct logs to specific files in project root [cite: 1]
    python3 "$PROJECT_ROOT/src/a2a/shop_agent.py" --shop_id webmall_$i --port 800$i > "$PROJECT_ROOT/shop_$i.log" 2>&1 & 
    echo "   -> webmall_$i starting on port 800$i"
done

# 3. Buffer for initialization [cite: 1]
echo "⏳ Waiting 20s for LangGraph and ES initialization..."
sleep 20

# 4. Connectivity Pre-check [cite: 1]
echo "🔍 Running Connectivity Pre-check..."
CHECK_OUTPUT=$(python3 "$PROJECT_ROOT/src/a2a/test_connection.py")
echo "$CHECK_OUTPUT"

# Stop if any shop failed the connectivity check [cite: 1]
if echo "$CHECK_OUTPUT" | grep -qE "❌|Unreachable"; then
    echo "❌ CRITICAL: Connectivity check failed!"
    echo "👉 Check shop_*.log in the project root for details."
    pkill -f shop_agent.py
    exit 1
fi

echo "✅ All shops verified. Commencing A2A Benchmark..."

# 5. Execute A2A Benchmark [cite: 1]
python3 "$PROJECT_ROOT/src/benchmark_a2a.py"

# 6. Final Cleanup [cite: 1]
echo "🛑 Shutting down Shop Agents..."
pkill -f shop_agent.py
echo "✅ Experiment sequence completed."