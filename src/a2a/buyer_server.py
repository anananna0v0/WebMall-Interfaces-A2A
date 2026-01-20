# src/a2a/buyer_server.py

import os
import json
import asyncio
from typing import Dict, Any, List

import httpx
from fastapi import FastAPI, HTTPException

app = FastAPI(title="A2A Buyer Agent")

# ---------------- Config ----------------
SHOP_ENDPOINTS = [
    os.getenv("SHOP_1", "http://localhost:8011"),
    os.getenv("SHOP_2", "http://localhost:8012"),
    os.getenv("SHOP_3", "http://localhost:8013"),
    os.getenv("SHOP_4", "http://localhost:8014"),
]

TIMEOUT = float(os.getenv("SHOP_TIMEOUT", "120"))
BUYER_PORT = int(os.getenv("BUYER_PORT", "8005"))

# ---------------- Helpers ----------------
async def call_shop(
    client: httpx.AsyncClient,
    shop_url: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        r = await client.post(
            shop_url.rstrip("/") + "/run_task",
            json=payload,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError("Shop response is not JSON object")
        return data
    except Exception as e:
        return {
            "shop_id": shop_url,
            "candidates": [],
            "final_urls": [],
            "cart_only_urls": [],
            "checkout_only_urls": [],
            "checkout_successful": False,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "error": str(e),
        }


def select_best_shop(shop_results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """
    Return the FULL shop result that has candidates.
    """
    for r in shop_results:
        cands = r.get("candidates")
        if isinstance(cands, list) and len(cands) > 0:
            return r
    return None


def extract_urls_from_candidate(candidate: Dict[str, Any]) -> List[str]:
    if isinstance(candidate, dict) and isinstance(candidate.get("url"), str):
        return [candidate["url"]]
    return []


# ---------------- API ----------------
@app.post("/run_task")
async def run_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input:
    {
      "task_id": "...",
      "task": "...",
      "task_category": "...",
      "clarify_policy": "off" | "on_once"
    }
    """
    try:
        task_id = payload.get("task_id")
        task_text = payload.get("task")
        task_category = payload.get("task_category", "")

        if not task_id or not isinstance(task_text, str) or not task_text.strip():
            raise HTTPException(status_code=400, detail="Missing task_id or non-empty task")

        shop_payload = {
            "task_id": task_id,
            "task": task_text,
            "task_category": task_category,
        }

        # ---------- SAFE PARALLEL SHOP CALLS ----------
        async with httpx.AsyncClient() as client:
            tasks = [
                asyncio.wait_for(
                    call_shop(client, shop_url, shop_payload),
                    timeout=TIMEOUT,
                )
                for shop_url in SHOP_ENDPOINTS
            ]

            shop_results: List[Dict[str, Any]] = []
            for coro in asyncio.as_completed(tasks):
                try:
                    r = await coro
                    shop_results.append(r)
                except asyncio.TimeoutError:
                    shop_results.append({
                        "shop_id": "unknown",
                        "candidates": [],
                        "final_urls": [],
                        "cart_only_urls": [],
                        "checkout_only_urls": [],
                        "checkout_successful": False,
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                        "error": "shop_timeout",
                    })

        # ---------- UNION ALL SHOP final_urls ----------
        # Keep best_shop only for debug / checkout metadata
        best_shop = select_best_shop(shop_results)

        # Union and deduplicate URLs across all shop agents
        union_urls: List[str] = []
        seen: set[str] = set()

        for shop_result in shop_results:
            urls = shop_result.get("final_urls", []) or []
            if not isinstance(urls, list):
                continue

            for url in urls:
                if isinstance(url, str) and url not in seen:
                    seen.add(url)
                    union_urls.append(url)

        final_urls: List[str] = union_urls

        # Provide a traceable answer payload (not used for evaluation)
        if final_urls:
            final_answer_text = json.dumps(
                {
                    "aggregation": "union",
                    "n_union": len(final_urls),
                    "final_urls": final_urls,
                },
                ensure_ascii=False,
            )
        else:
            final_answer_text = "No suitable product found."

        # Preserve existing checkout-related metadata (unchanged behavior)
        cart_only_urls: List[str] = []
        checkout_only_urls: List[str] = []
        checkout_successful = False

        if best_shop:
            cart_only_urls = best_shop.get("cart_only_urls", []) or []
            checkout_only_urls = best_shop.get("checkout_only_urls", []) or []
            checkout_successful = bool(best_shop.get("checkout_successful", False))


        buyer_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

        return {
            "task_id": task_id,
            "final_answer": final_answer_text,
            "final_urls": final_urls,
            "cart_only_urls": cart_only_urls,
            "checkout_only_urls": checkout_only_urls,
            "checkout_successful": checkout_successful,
            "buyer_usage": buyer_usage,
            "shop_results": shop_results,
        }

    except Exception as e:
        import traceback
        print("🔥 BUYER SERVER ERROR")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=BUYER_PORT)
