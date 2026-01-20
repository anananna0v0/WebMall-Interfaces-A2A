import asyncio

async def main():
    # Pick one server first (E-Store Athletes / webmall_1)
    server_url = "http://localhost:8060/sse"

    from mcp import ClientSession
    from mcp.client.sse import sse_client

    print("Connecting to:", server_url)

    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]

            print("\n=== TOOLS ===")
            for name in tool_names:
                print("-", name)

            # Find a search-like tool
            candidates = [n for n in tool_names if "search" in n.lower() or "find" in n.lower() or "keyword" in n.lower() or "items" in n.lower()]
            if not candidates:
                print("\nNo search-like tools found.")
                return

            # Prefer known tool names if present
            preferred = ["search_products", "find_items_techtalk", "query_stock", "get_items_by_keyword"]
            tool_name = next((p for p in preferred if p in tool_names), candidates[0])

            query = "AMD Ryzen 9 5900X"
            print("\nUsing tool:", tool_name)
            print("Query:", query)

            # Try common arg shapes
            arg_variants = [{"query": query}, {"keyword": query}, {"q": query}, {"text": query}]
            for args in arg_variants:
                try:
                    print("\nCalling with args:", args)
                    result = await session.call_tool(tool_name, args)

                    print("\n=== RAW TOOL RESULT ===")
                    print(result)

                    if hasattr(result, "content"):
                        print("\n=== RESULT.content (text) ===")
                        for c in result.content:
                            if hasattr(c, "text"):
                                print(c.text)
                    return
                except Exception as e:
                    print("Call failed:", repr(e))

            print("\nAll arg variants failed.")

if __name__ == "__main__":
    asyncio.run(main())
