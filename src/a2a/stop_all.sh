#!/usr/bin/env bash

echo "Stopping all A2A and NLWeb MCP servers..."

# -------- Stop Buyer agent --------
BUYER_PORT=8005
BUYER_PID=$(lsof -ti tcp:${BUYER_PORT})
if [ -n "$BUYER_PID" ]; then
  kill $BUYER_PID
  echo "Stopped Buyer agent (port ${BUYER_PORT})"
else
  echo "Buyer agent not running"
fi

# -------- Stop Shop agents --------
SHOP_PORTS=(8011 8012 8013 8014)
for PORT in "${SHOP_PORTS[@]}"; do
  PID=$(lsof -ti tcp:${PORT})
  if [ -n "$PID" ]; then
    kill $PID
    echo "Stopped Shop agent on port ${PORT}"
  else
    echo "Shop agent on port ${PORT} not running"
  fi
done

# -------- Stop NLWeb MCP servers --------
MCP_PORTS=(8001 8002 8003 8004)
for PORT in "${MCP_PORTS[@]}"; do
  PID=$(lsof -ti tcp:${PORT})
  if [ -n "$PID" ]; then
    kill $PID
    echo "Stopped NLWeb MCP server on port ${PORT}"
  else
    echo "NLWeb MCP server on port ${PORT} not running"
  fi
done

echo "All servers stopped."
