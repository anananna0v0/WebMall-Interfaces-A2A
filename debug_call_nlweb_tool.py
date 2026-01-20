import asyncio

async def main():
    server_url = "http://localhost:8001/sse"

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

            # For nlweb_mcp, search is typically behind ask_webmall_1
            tool_name = "ask_webmall_1"
            if tool_name not in tool_names:
                print(f"\nTool {tool_name} not found. Exiting cleanly.")
                return

            query = "Find all offers for the AMD Ryzen 9 5900X."
            print("\nUsing tool:", tool_name)
            print("Query:", query)

            # Try common argument names
            arg_variants = [
                #{"question": query},
                #{"task": query},
                #{"input": query},
                {"query": query},
                #{"message": query},
                #{"text": query},
            ]

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

            print("\nAll arg variants failed (but exited cleanly).")

if __name__ == "__main__":
    asyncio.run(main())
