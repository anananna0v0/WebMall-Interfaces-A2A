import httpx
import asyncio
import json
import os
import sys

# Define path to registry.json
REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "registry.json")

async def verify_shop_connectivity():
    """
    Checks if all shop agents listed in registry.json are reachable.
    Sends a dummy JSON-RPC request to verify service status.
    """
    if not os.path.exists(REGISTRY_PATH):
        print(f"❌ Error: registry.json not found at {REGISTRY_PATH}")
        return

    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            shops = data.get("shops", [])
        except json.JSONDecodeError:
            print("❌ Error: Failed to parse registry.json")
            return

    print(f"🔍 Testing connectivity for {len(shops)} shops...\n")

    async with httpx.AsyncClient(timeout=5.0) as client:
        for shop in shops:
            url = shop.get('url')
            shop_id = shop.get('id')
            
            # Construct a minimal JSON-RPC 2.0 payload
            payload = {
                "jsonrpc": "2.0",
                "method": "ping",
                "params": {"query": "test"},
                "id": "ping-1"
            }
            
            try:
                response = await client.post(url, json=payload)
                if response.status_code == 200:
                    print(f"✅ {shop_id:<12} | Status: Online  | URL: {url}")
                else:
                    print(f"⚠️ {shop_id:<12} | Status: Error {response.status_code} | URL: {url}")
            except httpx.ConnectError:
                print(f"❌ {shop_id:<12} | Status: Unreachable | URL: {url}")
            except Exception as e:
                print(f"❌ {shop_id:<12} | Status: Failed | Error: {type(e).__name__}")

if __name__ == "__main__":
    # Ensure PYTHONPATH includes src to handle any internal imports if needed
    asyncio.run(verify_shop_connectivity())