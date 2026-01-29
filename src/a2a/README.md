# A2A Module

This folder contains a small proof-of-concept implementation of the Agent - to - Agent protocol for the WebMall benchmark.  The aim is to showcase how the emerging A2A standard can be used to connect multiple shop agents and a buyer agent in a multi-agent workflow.

## Overview

- **`shop_agent.py`** – a FastAPI server implementing the A2A JSON-RPC endpoint.  It wraps a LangGraph ReAct agent that queries the existing NLWeb search index and returns structured product offers in Schema.org format while recording token usage.
- **`run_a2a_exp.sh`** – a convenience script to launch four shop agents, perform a connectivity pre-check and run the A2A benchmark.
- **`benchmark_a2a.py`** – a scripted buyer that broadcasts each task to all shop agents, collects offers and selects the cheapest product.
- **`registry.json`** – defines the shop endpoints used by the benchmark.
- **`test_connection.py`** – checks whether all registered shop agents are reachable.

## Prerequisites

- Python 3.8+
- Elasticsearch 8.x running at `http://localhost:9200`
- An OpenAI API key set in `.env` (copy `.env.example` and fill in your credentials)
- Data indexed with `python src/nlweb_mcp/ingest_data.py --shop all --force-recreate`

## Running

Start the shop agents and benchmark with:

```bash
bash run_a2a_exp.sh
