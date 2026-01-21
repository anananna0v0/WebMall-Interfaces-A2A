# src/a2a/buyer_server.py

import os
import json
import asyncio
from typing import Dict, Any, List
from openai import OpenAI

import httpx
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

load_dotenv()
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

@app.post("/run_task")
async def run_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Buyer agent /run_task.

    Notes:
    - Always calls all shop agents in parallel and unions candidate URLs.
    - Applies buyer-side decisions by task_category.
    - Never crashes the server: decision logic is guarded with try/except.
    """
    try:
        task_id = payload.get("task_id")
        task_text = payload.get("task")

        # Accept both keys to avoid schema mismatch
        task_category = (payload.get("task_category") or payload.get("category") or "").strip()

        if not task_id or not isinstance(task_text, str) or not task_text.strip():
            raise HTTPException(status_code=400, detail="Missing task_id or non-empty task")

        shop_payload = {
            "task_id": task_id,
            "task": task_text,
            "task_category": task_category,
            "category": task_category,  # backward/compat
        }

        buyer_debug = {
            "BUYER_USE_LLM": os.getenv("BUYER_USE_LLM"),
            "BUYER_LLM_MODEL": os.getenv("BUYER_LLM_MODEL"),
            "cheapest": {"entered": False, "use_llm": None, "called": False, "error": None},
        }

        # Initialize usage BEFORE any decision logic (prevents NameError)
        buyer_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
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
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                        "error": "shop_timeout",
                    })

        # ---------- UNION ALL SHOP final_urls ----------
        best_shop = select_best_shop(shop_results)

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

        # =========================================================
        # Buyer-side decision logic (guarded; never crash server)
        # =========================================================

        # --- Specific_Product: exact-match post-filter (buyer decision) ---
        if task_category == "Specific_Product" and final_urls:
            try:
                import re
                q = task_text.lower()
                token = None

                m = re.search(r"\b(\d{4}x)\b", q)  # e.g., 5900x
                if m:
                    token = m.group(1)

                if token:
                    kept = [
                        u for u in final_urls
                        if isinstance(u, str) and "/product/" in u and token in u.lower()
                    ]
                    if kept:
                        final_urls = kept
            except Exception as e:
                print(f"[buyer] Specific_Product exact filter failed: {type(e).__name__}: {e}")

        # --- Cheapest_Product: LLM decision stub (safe) ---
        # IMPORTANT: This does NOT fetch prices yet.
        # It only demonstrates how to plug in LLM without crashing.
        if task_category == "Cheapest_Product" and final_urls:
            buyer_debug["cheapest"]["entered"] = True
            try:
                use_llm = os.getenv("BUYER_USE_LLM", "0").strip() == "1"
                buyer_debug["cheapest"]["use_llm"] = use_llm

                if use_llm:
                    buyer_debug["cheapest"]["called"] = True
                    try:
                        from openai import OpenAI
                        client = OpenAI()
                        model = os.getenv("BUYER_LLM_MODEL", "gpt-5-mini")

                        prompt_obj = {
                            "task": task_text,
                            "candidates": final_urls,
                            "instruction": (
                                "Choose the cheapest offer URL(s) that satisfy the task. "
                                "Return ONLY JSON: {\"urls\": [\"<url>\", ...]} "
                                "Only include URLs from candidates."
                            ),
                        }

                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": "Return strict JSON only."},
                                {"role": "user", "content": json.dumps(prompt_obj, ensure_ascii=False)},
                            ],
                        )

                        content = resp.choices[0].message.content if resp.choices else ""
                        obj = json.loads(content) if isinstance(content, str) else {}
                        llm_urls = obj.get("urls", [])

                        if isinstance(llm_urls, list) and llm_urls:
                            s = set(final_urls)
                            picked = [u for u in llm_urls if isinstance(u, str) and u in s]
                            if picked:
                                final_urls = picked

                        try:
                            buyer_usage["prompt_tokens"] += int(resp.usage.prompt_tokens or 0)
                            buyer_usage["completion_tokens"] += int(resp.usage.completion_tokens or 0)
                        except Exception:
                            pass

                    except Exception as inner:
                        buyer_debug["cheapest"]["error"] = f"{type(inner).__name__}: {inner}"

            except Exception as e:
                buyer_debug["cheapest"]["error"] = f"{type(e).__name__}: {e}"

        # =========================================================

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

        return {
            "task_id": task_id,
            "final_answer": final_answer_text,
            "final_urls": final_urls,
            "buyer_usage": buyer_usage,
            "buyer_debug": buyer_debug,
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
