# WebMall-Interfaces A2A Extension

This repository is a fork of the upstream [WebMall-Interfaces](https://github.com/wbsg-uni-mannheim/WebMall-Interfaces) project developed at the University of Mannheim.

The original project provides an experimental framework for evaluating different agent interfaces for web-based e-commerce tasks, including HTML-based browsing, RAG, MCP, and NLWeb.

This fork adds a lightweight, experimental **Agent-to-Agent (A2A)-style** interaction setup on top of the existing WebMall benchmark. The goal is to explore how multiple shop agents and a buyer agent can interact via a structured JSON-RPC interface in a multi-agent setting, while keeping the benchmark and evaluation logic unchanged.

## What is added in this fork

- A simple A2A-style shop agent implementation exposing a JSON-RPC endpoint
- A scripted buyer that queries multiple shop agents and aggregates their responses
- A registry-based setup for static agent discovery
- Benchmark scripts to compare agent responses within the existing WebMall task framework

The A2A-related code is located under `src/a2a/` and is intended as a proof-of-concept rather than a full implementation of the A2A specification.

## Getting started

For environment setup, data ingestion, and baseline benchmarks, please follow the instructions in the upstream WebMall-Interfaces repository.

To run the A2A experiment, see the README in `src/a2a/`.
