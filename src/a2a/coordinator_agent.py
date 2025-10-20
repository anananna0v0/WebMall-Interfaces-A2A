import requests

BUYER_URL = "http://localhost:10001/a2a/sendMessage"

def main():
    # This simulates an A2A client sending a task to the Buyer
    payload = {
        "input": {
            "text": "Find me a 13-inch laptop under 1000 EUR"
        }
    }
    response = requests.post(BUYER_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        print("\n=== Coordinator received response from Buyer ===")
        print("Text:", data["output"]["text"])
        for item in data["output"]["artifacts"]:
            print(f"- {item['name']} | {item['price']} EUR | {item['url']}")
    else:
        print("Request failed:", response.status_code, response.text)

if __name__ == "__main__":
    main()
