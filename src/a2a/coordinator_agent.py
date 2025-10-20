import time
import requests

BUYER_URLS = [
    "http://localhost:10001/a2a/sendMessage",
    "http://localhost:10002/a2a/sendMessage",
    "http://localhost:10003/a2a/sendMessage",
    "http://localhost:10004/a2a/sendMessage"
]

def main():
    query = "Find me a wireless mouse"
    best_price = float('inf')
    best_item = None

    print(f"Query: {query}\n")

    for url in BUYER_URLS:
        start_time = time.time()  # Record the start time
        response = requests.post(url, json={"input": {"text": query}})
        elapsed = time.time() - start_time  # Calculate the duration

        data = response.json()
        artifacts = data["output"]["artifacts"]

        print(f"--- {url} ---")
        print(f"Response time: {elapsed:.2f} sec")

        for item in artifacts:
            price_str = str(item.get("price", "0")).replace(",", ".")
            name = item.get("name", "Unknown product")
            url_link = item.get("url", "")
            print(f"  {name} | {price_str} | {url_link}")

            try:
                price = float(price_str)
                if price < best_price:
                    best_price = price
                    best_item = item
            except:
                continue
        print()

    if best_item:
        print(f"\nCheapest item: {best_item['name']} | {best_item['price']} EUR | {best_item['url']}")
    else:
        print("No matching items found.")



if __name__ == "__main__":
    main()
