# NLWeb MCP Implementation

This directory contains a complete implementation of NLWeb using MCP (Model Context Protocol) servers for the WebMall benchmark. Each WebMall shop has its own dedicated MCP server with semantic search capabilities.

## Architecture

- **4 MCP Servers**: One for each WebMall shop (webmall_1, webmall_2, webmall_3, webmall_4)
- **4 Elasticsearch Indices**: `webmall_1_nlweb`, `webmall_2_nlweb`, `webmall_3_nlweb`, `webmall_4_nlweb`
- **Semantic Search**: Using OpenAI text-embedding-3-small embeddings with cosine similarity
- **Schema.org Compliance**: All products stored and returned in schema.org JSON-LD format

## Components

### Core Components

- `config.py` - Configuration settings for all components
- `elasticsearch_client.py` - Elasticsearch integration with semantic search
- `embedding_service.py` - OpenAI embedding generation and batch processing
- `woocommerce_client.py` - WooCommerce API integration for data extraction
- `search_engine.py` - Semantic search engine with NLWeb-compatible responses
- `data_ingestion.py` - Complete data ingestion pipeline

### MCP Servers

- `mcp_servers/base_server.py` - Base MCP server implementation
- `mcp_servers/webmall_1_server.py` - MCP server for E-Store Athletes
- `mcp_servers/webmall_2_server.py` - MCP server for TechTalk  
- `mcp_servers/webmall_3_server.py` - MCP server for CamelCases
- `mcp_servers/webmall_4_server.py` - MCP server for Hardware Cafe

### Utilities

- `ingest_data.py` - Data ingestion script
- `start_all_servers.py` - Server management script

## Setup

### Prerequisites

1. **Elasticsearch**: Running on `http://localhost:9200`
2. **OpenAI API Key**: Set in environment variable `OPENAI_API_KEY`
3. **WooCommerce API Access**: Consumer keys and secrets for each WebMall shop (optional)
4. **Python Dependencies**: Install required packages

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file with:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
ELASTICSEARCH_HOST=http://localhost:9200

# Optional - WooCommerce API credentials for authenticated access
WOO_CONSUMER_KEY_1=your_webmall_1_consumer_key
WOO_CONSUMER_SECRET_1=your_webmall_1_consumer_secret
WOO_CONSUMER_KEY_2=your_webmall_2_consumer_key
WOO_CONSUMER_SECRET_2=your_webmall_2_consumer_secret
WOO_CONSUMER_KEY_3=your_webmall_3_consumer_key
WOO_CONSUMER_SECRET_3=your_webmall_3_consumer_secret
WOO_CONSUMER_KEY_4=your_webmall_4_consumer_key
WOO_CONSUMER_SECRET_4=your_webmall_4_consumer_secret
```

**Note**: WooCommerce credentials are optional. The system will attempt to access public endpoints if credentials are not provided.

## Usage

### 1. Data Ingestion

First, ingest product data from all WebMall shops:

```bash
cd src/nlweb_mcp
python ingest_data.py --shop all --force-recreate
```

### 2. Start MCP Servers

Start all MCP servers:

```bash
# OR from nlweb_mcp directory:
cd src/nlweb_mcp
python start_all_servers.py
```

Or start individual servers:

```bash
python src/nlweb_mcp/mcp_servers/webmall_1_server.py
python src/nlweb_mcp/mcp_servers/webmall_2_server.py
python src/nlweb_mcp/mcp_servers/webmall_3_server.py
python src/nlweb_mcp/mcp_servers/webmall_4_server.py
```

### 3. Run Benchmark

Execute the benchmark using MCP servers:

```bash
python src/benchmark_nlweb_mcp.py
```

## Data Structure

### Product Storage Format

Each product is stored in Elasticsearch with:

```json
{
  "product_id": "1234",
  "url": "https://webmall-1.informatik.uni-mannheim.de/product/example/",
  "title": "Product Name",
  "price": 99.99,
  "description": "Product description",
  "related_products": ["1235", "1236"],
  "category": "Electronics",
  "string_representation": "1234 - https://... - Product Name - $99.99 - Description - 1235,1236 - Electronics",
  "embedding": [0.1, 0.2, ...],  // 1536-dimensional vector
  "schema_org": { ... },  // Complete schema.org Product JSON-LD
  "created_at": "2025-01-01T00:00:00",
  "updated_at": "2025-01-01T00:00:00"
}
```

### MCP Tool Interface

Each MCP server provides three tools:

1. **ask** - Natural language product search
   - Input: `question` (string), `top_k` (integer, optional)
   - Output: NLWeb-compatible JSON response

2. **get_product** - Get specific product by ID
   - Input: `product_id` (string)
   - Output: Single product schema.org JSON

3. **health_check** - Check server health
   - Input: None
   - Output: Health status information

## Configuration

### Shop Configuration

```python
WEBMALL_SHOPS = {
    "webmall_1": {
        "url": "https://webmall-1.informatik.uni-mannheim.de",
        "index_name": "webmall_1_nlweb",
        "mcp_port": 8001
    },
    # ... other shops
}
```

### Search Configuration

- **Default top_k**: 10 results
- **Maximum top_k**: 50 results
- **Embedding model**: text-embedding-3-small (1536 dimensions)
- **Similarity**: Cosine similarity

## Performance

### Semantic Search
- Uses dense vector search with cosine similarity
- Configurable result count (k=1 to 50)
- Real-time embedding generation for queries

### Batch Processing
- Embedding generation in batches of 100
- Bulk Elasticsearch indexing (100 documents per batch)
- Rate limiting for OpenAI API calls

## Monitoring

### Logs
- Ingestion logs saved to `ingestion_YYYYMMDD_HHMMSS.log`
- Server logs output to console
- JSON results saved to timestamped files

### Health Checks
- Elasticsearch connection status
- OpenAI API availability
- Index document counts and sizes
- Server process status

## Troubleshooting

### Common Issues

1. **Elasticsearch Connection Failed**
   - Ensure Elasticsearch is running on `localhost:9200`
   - Check firewall settings

2. **OpenAI API Errors**
   - Verify `OPENAI_API_KEY` environment variable
   - Check API rate limits and usage

3. **Empty Search Results**
   - Ensure data ingestion completed successfully
   - Check index document counts with health check

4. **MCP Server Connection Issues**
   - Verify servers are running with `--check-only`
   - Check for port conflicts
   - Review server logs for errors

### Debug Mode

Enable debug logging for detailed information:

```bash
python src/nlweb_mcp/ingest_data.py --debug
python src/nlweb_mcp/start_all_servers.py --debug
```

## Comparison with Original NLWeb

This implementation provides:

- ✅ **Local Control**: No external API dependencies
- ✅ **Semantic Search**: Advanced embedding-based search
- ✅ **Schema.org Compliance**: Full structured data support
- ✅ **Individual Servers**: Dedicated server per shop
- ✅ **Real-time Search**: No pre-computed results
- ✅ **Configurable Results**: Adjustable top-k values
- ✅ **Health Monitoring**: Built-in status checks

## Future Enhancements

Potential improvements:
- Hybrid search (semantic + keyword)
- Query expansion and synonyms
- Result ranking optimization
- Caching layer for frequent queries
- Multi-language support
- Product image search