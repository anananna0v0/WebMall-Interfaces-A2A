# WebMall-Interfaces A2A Extension

This repository is a fork of the upstream [WebMall-Interfaces](https://github.com/wbsg-uni-mannheim/WebMall-Interfaces) project developed at the University of Mannheim.

The original project provides an experimental framework for evaluating different agent interfaces for web-based e-commerce tasks, including HTML-based browsing, RAG, MCP, and NLWeb.

This fork extends the benchmark with a fully functional **Agent-to-Agent (A2A) Protocol implementation**. It introduces a multi-agent environment where Shop Agents expose structured Agent Cards and capabilities, allowing a Buyer Agent to perform discovery, selection, and negotiation via standard JSON-RPC 2.0 messaging.

## Key Features in this Fork

- **A2A Protocol Compliance**: Implements the core A2A specifications, including JSON-RPC 2.0 communication and standard error handling.
- **Capability-based Discovery**: A simulation of the A2A Discovery phase where the Buyer Agent inspects **Agent Cards** (via a registry) to filter shops based on supported capabilities (e.g., `product_search`).
- **Scripted Buyer Logic**: A deterministic buyer that queries multiple capable agents, aggregates standardized offers (Schema.org), and selects the best deal.
- **Benchmark Integration**: Seamlessly integrates A2A metrics (Task Completion Rate, Token Usage, Latency) into the existing WebMall evaluation framework.

The A2A-related code is located under `src/a2a/`. This implementation serves as a concrete proof-of-concept for deploying autonomous commerce agents using the A2A standard.

## Getting Started

For environment setup, data ingestion, and baseline benchmarks, please follow the instructions in the upstream WebMall-Interfaces repository.

To run the A2A experiment, see the detailed README in `src/a2a/`.