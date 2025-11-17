# Website Crawler for Elasticsearch RAG

`crawl_website.py` is an async web crawler that indexes website content into Elasticsearch for RAG (Retrieval-Augmented Generation) applications.

## Features

- Crawls product pages and extracts content using unstructured library
- Generates embeddings and summaries using OpenAI API
- Stores chunked content in Elasticsearch with metadata
- Supports multiple crawling modes: single URL, sitemap, RSS feed, or link discovery
- Skips duplicate URLs and non-product pages
- Parallel processing with configurable concurrency
- Index reset functionality

## Requirements

- OpenAI API key in `.env` file
- Elasticsearch instance running
- Python dependencies: `crawl4ai`, `openai`, `elasticsearch`, `unstructured`, `beautifulsoup4`

## Usage

### Single URL
```bash
python crawl_website.py https://example.com/product/item
```

### Sitemap crawling
```bash
python crawl_website.py https://example.com/sitemap.xml --sitemap
```

### RSS feed crawling
```bash
python crawl_website.py https://example.com/?feed=products --rss
```

### Multiple RSS feeds from JSON file
```bash
python crawl_website.py --rss-list rss_feeds.json
```

### Link discovery
```bash
python crawl_website.py https://example.com --discover --max-depth 3
```

### Reset index before crawling
```bash
python crawl_website.py --rss-list rss_feeds.json --reset-index
```

## Options

- `--sitemap`: Treat URL as sitemap XML
- `--rss`: Treat URL as RSS feed  
- `--rss-list FILE`: JSON file containing list of RSS feed URLs
- `--discover`: Follow links to discover URLs
- `--reset-index`: Delete and recreate Elasticsearch index before crawling
- `--max-depth N`: Maximum depth for link discovery (default: 2)
- `--max-concurrent N`: Concurrent requests limit (default: 5)
- `--allow-domain-hopping`: Follow external domain links during discovery

## RSS Feed Support

The crawler can extract product URLs from RSS feeds using the Google Shopping namespace (`g:link` elements). Example RSS structure:

```xml
<rss xmlns:g="http://base.google.com/ns/1.0" version="2.0">
  <channel>
    <item>
      <g:link>https://example.com/product/item1</g:link>
    </item>
  </channel>
</rss>
```

## Filtering

- Only processes URLs containing `/product/`
- Skips cart, checkout, and add-to-cart pages
- Ignores images, videos, and anchor links
- Prevents duplicate URL processing

## Example RSS Feeds File

`rss_feeds.json`:
```json
[
    "https://webmall-1.informatik.uni-mannheim.de/?feed=products",
    "https://webmall-2.informatik.uni-mannheim.de/?feed=products",
    "https://webmall-3.informatik.uni-mannheim.de/?feed=products",
    "https://webmall-4.informatik.uni-mannheim.de/?feed=products"
]
```