# A2A Module

This folder contains a proof-of-concept implementation of the **Agent-to-Agent (A2A) Protocol** within the WebMall benchmark. It demonstrates a complete multi-agent commerce workflow: Discovery, Connection, and Message Exchange.

## Overview

The system consists of independent Shop Agents and a Buyer Agent interacting via JSON-RPC over HTTP:

- **`shop_agent.py`** – A FastAPI server acting as an A2A-compliant Shop Agent. It wraps a LangGraph ReAct agent that queries the local Elasticsearch index and returns structured product offers.
- **`benchmark_a2a.py`** – The **Buyer Agent**. It implements **Capability Discovery** by inspecting Agent Cards to find suitable shops before broadcasting tasks. It aggregates responses and selects the cheapest product.
- **`registry.json`** – A **static centralized registry** used for discovery simulation. It hosts **Agent Cards** containing metadata and capability lists (e.g., `["product_search"]`), enabling the buyer to look up and filter agents before connection.
- **`run_a2a_exp.sh`** – An automated orchestration script that launches four shop agents, performs connectivity checks, and runs the full benchmark suite.
- **`test_connection.py`** – A utility to verify that all registered shop agents are reachable and responding correctly.

## Prerequisites

- **Python 3.8+**
- **Elasticsearch 8.11.0** (Strictly required to match the Python client version)
  - Must be running at `http://localhost:9200`.
- **OpenAI API Key**: Set in `.env` (copy `.env.example` and fill in your credentials).
- **Data Ingestion**: Ensure the shop data is indexed:
  ```bash
  python src/nlweb_mcp/ingest_data.py --shop all --force-recreate