import time
import requests

BUYER_URLS = [
    "http://localhost:10001/a2a/sendMessage",
    "http://localhost:10002/a2a/sendMessage",
    "http://localhost:10003/a2a/sendMessage",
    "http://localhost:10004/a2a/sendMessage"
]

def main():
    query = "Find me a keyboard"
    best_price = float('inf')
    best_item = None
    attempt = 1

    print(f"Query: {query}\n")

    while attempt <= 2:
        print(f"=== Round {attempt} ===")
        found_any = False

        for url in BUYER_URLS:
            start_time = time.time()
            response = requests.post(url, json={"input": {"text": query}})
            elapsed = time.time() - start_time
            data = response.json()
            artifacts = data["output"]["artifacts"]

            print(f"--- {url} ---")
            print(f"Response time: {elapsed:.2f} sec")

            for item in artifacts:
                price_str = str(item.get("price", "")).replace(",", ".")
                name = item.get("name", "Unknown product")
                url_link = item.get("url", "")
                print(f"  {name} | {price_str} | {url_link}")

                try:
                    price = float(price_str)
                    if price > 0:
                        found_any = True
                        if price < best_price:
                            best_price = price
                            best_item = item
                except ValueError:
                    continue
            print()

        if found_any or attempt == 2:
            break
        else:
            print("No valid items found. Trying again with relaxed condition...\n")
            query = "Find me an octopus under 100 EUR"
            attempt += 1

    if best_item:
        print(f"\nCheapest item: {best_item['name']} | {best_item['price']} EUR | {best_item['url']}")

        # Determine which Buyer handled this product
        selected_buyer = None
        for url in BUYER_URLS:
            shop_domain = best_item['url'].split("/")[2]
            # Match webmall domain with Buyer port
            if shop_domain in url:
                selected_buyer = url
                break

        if not selected_buyer:
            selected_buyer = BUYER_URLS[0]

        print(f"\nSending add-to-cart request to: {selected_buyer}")
        add_request = {"input": {"text": f"Add the cheapest {query} to cart"}}
        add_response = requests.post(selected_buyer, json=add_request)
        print("Add-to-cart response:", add_response.json()["output"]["text"])

        print(f"\nProceeding to checkout at: {selected_buyer}")
        checkout_request = {"input": {"text": "checkout"}}
        checkout_response = requests.post(selected_buyer, json=checkout_request)
        print("Checkout response:", checkout_response.json()["output"]["text"])

    else:
        print("No matching items found.")

if __name__ == "__main__":
    main()
