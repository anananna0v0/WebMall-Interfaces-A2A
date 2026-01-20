#!/usr/bin/env bash
set -e

# ====== Basic configuration ======
# Repo root
ROOT_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"

# Source directory
SRC_DIR="$ROOT_DIR/src"
A2A_DIR="$SRC_DIR/a2a"

# Ports
BUYER_PORT=8005
SHOP_PORTS=(8011 8012 8013 8014)
MCP_PORTS=(8001 8002 8003 8004)

# Shops
SHOP_IDS=(webmall_1 webmall_2 webmall_3 webmall_4)

# LLM
export SHOP_LLM_MODEL=gpt-5-mini

# ====== Start NLWeb MCP servers (tool layer) ======
echo "▶ Starting NLWeb MCP servers..."
(
  cd "$ROOT_DIR"
  PYTHONPATH=src python src/nlweb_mcp/start_all_servers.py
) &
MCP_PID=$!

sleep 3

# ====== Start Shop agents ======
echo "▶ Starting Shop agents..."
for i in {0..3}; do
  SHOP_ID="${SHOP_IDS[$i]}"
  SHOP_PORT="${SHOP_PORTS[$i]}"
  MCP_PORT="${MCP_PORTS[$i]}"

  echo "  - $SHOP_ID on port $SHOP_PORT (MCP :$MCP_PORT)"
  (
    export SHOP_ID="$SHOP_ID"
    export SHOP_PORT="$SHOP_PORT"
    export SHOP_BACKEND=nlweb
    export SHOP_MCP_URL="http://localhost:${MCP_PORT}/sse"
    PYTHONPATH=src python "$A2A_DIR/shop_server.py"
  ) &
done

sleep 2

# ====== Start Buyer agent ======
echo "▶ Starting Buyer agent on port $BUYER_PORT"
(
  export BUYER_PORT="$BUYER_PORT"
  PYTHONPATH=src python "$A2A_DIR/buyer_server.py"
) &

echo ""
echo "✅ All servers started."
echo "   Buyer : http://localhost:$BUYER_PORT"
echo "   Shops : ${SHOP_PORTS[*]}"
echo "   MCP   : ${MCP_PORTS[*]}"
echo ""
echo "▶ Press Ctrl+C to stop all servers."

wait
