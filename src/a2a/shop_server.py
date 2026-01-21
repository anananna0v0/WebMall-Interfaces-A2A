# src/a2a/shop_server.py

import os
import json
import re
import asyncio
from typing import Dict, Any, List, Tuple

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

app = FastAPI(title="A2A Shop Agent (wraps MCP)")

SHOP_ID = os.getenv("SHOP_ID", "webmall_1")

# Choose backend: nlweb or api (hybrid)
BACKEND = os.getenv("SHOP_BACKEND", "nlweb").strip().lower()

# Point this shop to EXACTLY ONE MCP server (per shop process)
# nlweb_mcp benchmark uses 8001..8004/sse
# api_mcp benchmark uses 8060..8063/sse (hybrid)
MCP_URL = os.getenv("SHOP_MCP_URL", "http://localhost:8001/sse")

MCP_NAME = os.getenv("SHOP_MCP_NAME", f"{SHOP_ID}-{BACKEND}")

MODEL_NAME = os.getenv("SHOP_LLM_MODEL", "gpt-5-mini")

TIMEOUT_SEC = int(os.getenv("SHOP_TASK_TIMEOUT_SEC", "180"))

TOP_K_DEFAULT = int(os.getenv("SHOP_TOP_K", "3"))


def _extract_json_obj(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object from the model output.
    Expected format (strict):
    {
      "urls": ["..."],
      "cart_only_urls": ["..."],
      "checkout_only_urls": ["..."],
      "checkout_successful": true/false
    }
    If parsing fails, fallback: extract URLs via regex into urls[].
    """
    text = text.strip()

    # Try full JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Try to find an embedded {...} JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Fallback: regex URLs
    urls_found = re.findall(r"https?://\S+", text)
    urls = [u.strip(')>."\',') for u in urls_found]
    return {"urls": urls}


def _to_candidates(urls: List[str], top_k: int) -> List[Dict[str, Any]]:
    cands = []
    for u in urls[:top_k]:
        if isinstance(u, str) and u.strip():
            cands.append({"url": u.strip()})
    return cands


async def _get_tools() -> Tuple[MultiServerMCPClient, List[Any]]:
    """
    Create a MCP client connected to ONE server and return tools.
    """
    servers = {
        MCP_NAME: {
            "url": MCP_URL,
            "transport": "sse",
            "shop_id": SHOP_ID,
        }
    }
    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()
    return client, tools


async def _run_agent(task_text: str, top_k: int) -> Tuple[str, Dict[str, int]]:
    """
    Run a ReAct agent that can call MCP tools and MUST return strict JSON.
    Returns: (final_text, usage_dict)
    """
    llm = ChatOpenAI(model=MODEL_NAME)

    system = SystemMessage(
        content=(
            f"You are the Shop agent for shop_id={SHOP_ID} using backend={BACKEND}.\n\n"
            "You are solving ONLY tasks of category: Specific_Product.\n\n"
            "Task definition:\n"
            "- The user query refers to a specific, concrete product or clearly identifiable item.\n"
            "- Your job is to find ALL product or offer pages in THIS shop that match the product exactly.\n\n"
            "Rules:\n"
            "- Use the available search tools to find matching offers.\n"
            "- Return ALL matching offer URLs from this shop.\n"
            "- Do NOT rank, filter by price, or choose a best option.\n"
            "- Do NOT infer alternatives, substitutes, or similar products.\n"
            "- If the product does not exist in this shop, return an empty list.\n"
            "- Do NOT add to cart.\n"
            "- Do NOT checkout.\n\n"
            "Output format (STRICT):\n"
            "Return ONLY a JSON object with exactly this schema:\n"
            '{ "urls": ["<full offer url>", "..."] }\n\n'
            "If no matching product is found, return:\n"
            '{ "urls": [] }\n\n'
            "Do not include any other text, explanations, or keys."
        )
    )


    human = HumanMessage(
        content=(
            f"Task:\n{task_text}\n\n"
            f"Return at most {top_k} URLs in 'urls'."
        )
    )

    client, tools = await _get_tools()
    try:
        agent = create_react_agent(llm, tools)

        with get_openai_callback() as cb:
            result = await agent.ainvoke({"messages": [system, human]})
            # result is a dict with "messages"
            final_text = result["messages"][-1].content if result and "messages" in result else ""
            usage = {
                "prompt_tokens": int(getattr(cb, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(cb, "completion_tokens", 0) or 0),
            }
            return str(final_text), usage
    finally:
        # close MCP client if possible
        try:
            if hasattr(client, "close"):
                if asyncio.iscoroutinefunction(client.close):
                    await client.close()
                else:
                    client.close()
        except Exception:
            pass

async def run_tool_only_search(task_text: str, top_k: int):
    """
    Tool-only MCP search (no LLM, no ReAct).
    Returns: (urls, usage) where usage is always zero.
    """
    import re
    from urllib.parse import urlparse

    debug = (os.getenv("SHOP_DEBUG_MCP", "0").strip() == "1")

    # ---- helper: parse target model token from task text ----
    # Example: "AMD Ryzen 9 5900X" -> target_token="5900x"
    m = re.search(r"\b(\d{4}x)\b", task_text.lower())
    target_token = m.group(1) if m else None  # e.g., "5900x"

    def _is_product_page(u: str) -> bool:
        try:
            p = urlparse(u)
            return "/product/" in p.path
        except Exception:
            return False

    def _is_image_or_asset(u: str) -> bool:
        u2 = u.lower()
        return any(u2.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif")) or "/wp-content/" in u2

    def _is_schema_org(u: str) -> bool:
        return "schema.org" in u.lower()

    def _match_target(u: str) -> bool:
        if not target_token:
            return True
        return target_token in u.lower()

    def _filter_urls(urls):
        """Keep only product pages; drop schema.org and assets; enforce target token if available."""
        kept = []
        seen = set()
        for u in urls:
            if not isinstance(u, str) or not u.startswith("http"):
                continue
            if _is_schema_org(u) or _is_image_or_asset(u):
                continue
            if not _is_product_page(u):
                continue
            if not _match_target(u):
                continue
            u = u.rstrip("/")
            if u not in seen:
                seen.add(u)
                kept.append(u)
        return kept

    # ---- tool call ----
    client, tools = await _get_tools()
    try:
        if not tools:
            if debug:
                print(f"[{SHOP_ID}] No MCP tools available.")
            return [], {"prompt_tokens": 0, "completion_tokens": 0}

        # Pick a likely search tool
        tool = None
        for t in tools:
            name = str(getattr(t, "name", "")).lower()
            if any(k in name for k in ("ask", "search", "query", "find", "lookup")):
                tool = t
                break
        if tool is None:
            tool = tools[0]

        attempts = [
            {"query": task_text, "top_k": top_k},
            {"query": task_text, "k": top_k},
            {"q": task_text, "top_k": top_k},
            {"q": task_text, "k": top_k},
            {"query": task_text},
            {"q": task_text},
            {"input": task_text},
            {"text": task_text},
        ]

        def _to_str(x):
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)

        def _extract_urls_any(obj):
            urls = []

            def add(u):
                if isinstance(u, str) and u.startswith("http"):
                    urls.append(u)

            if isinstance(obj, dict):
                for k in ("urls", "final_urls"):
                    v = obj.get(k)
                    if isinstance(v, list):
                        for it in v:
                            add(it)

                for k in ("results", "items", "products", "offers", "candidates", "data"):
                    v = obj.get(k)
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict):
                                add(it.get("url"))
                                add(it.get("product_url"))
                                add(it.get("offer_url"))

                # MCP content wrapper: {"content":[{"type":"text","text":"..."}]}
                content = obj.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            txt = part["text"]
                            # Try parse JSON embedded in text
                            try:
                                j = json.loads(txt)
                                urls.extend(_extract_urls_any(j))
                            except Exception:
                                # Fallback: regex
                                for u in re.findall(r"https?://\S+", txt):
                                    urls.append(u.strip(')>."\','))

                # Fallback: regex on whole dict string
                s = _to_str(obj)
                for u in re.findall(r"https?://\S+", s):
                    urls.append(u.strip(')>."\','))

            elif isinstance(obj, list):
                for it in obj:
                    urls.extend(_extract_urls_any(it))

            elif isinstance(obj, str):
                for u in re.findall(r"https?://\S+", obj):
                    urls.append(u.strip(')>."\','))

            return urls

        last_err = None

        for args in attempts:
            try:
                out = await tool.ainvoke(args) if hasattr(tool, "ainvoke") else tool.invoke(args)

                out_str = _to_str(out)
                if debug:
                    print(f"[{SHOP_ID}] tool={getattr(tool,'name','?')} args={args} out_chars={len(out_str)} target={target_token}")
                    print(out_str[:1200])

                raw_urls = _extract_urls_any(out)
                filtered = _filter_urls(raw_urls)

                # If filtering removes everything but we have raw product pages, relax target match as fallback
                if not filtered:
                    # keep product pages only, no target token constraint
                    def _filter_relaxed(urls2):
                        kept = []
                        seen2 = set()
                        for u in urls2:
                            if not isinstance(u, str) or not u.startswith("http"):
                                continue
                            if _is_schema_org(u) or _is_image_or_asset(u):
                                continue
                            if not _is_product_page(u):
                                continue
                            u = u.rstrip("/")
                            if u not in seen2:
                                seen2.add(u)
                                kept.append(u)
                        return kept

                    filtered = _filter_relaxed(raw_urls)

                return filtered[:top_k], {"prompt_tokens": 0, "completion_tokens": 0}

            except Exception as e:
                last_err = e
                if debug:
                    print(f"[{SHOP_ID}] tool call failed args={args}: {type(e).__name__}: {e}")
                continue

        if debug and last_err:
            print(f"[{SHOP_ID}] All tool attempts failed. Last error: {type(last_err).__name__}: {last_err}")

        return [], {"prompt_tokens": 0, "completion_tokens": 0}

    finally:
        try:
            if hasattr(client, "close"):
                if asyncio.iscoroutinefunction(client.close):
                    await client.close()
                else:
                    client.close()
        except Exception:
            pass

            
@app.post("/run_task")
async def run_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Shop agent /run_task.
    - Search-only gate for Specific_Product.
    - Tool-only execution (no LLM/ReAct) to avoid token blow-up.
    """
    task_id = payload.get("task_id")
    task_text = payload.get("task")
    task_category = (payload.get("task_category") or payload.get("category") or "").strip()
    top_k = int(payload.get("top_k", TOP_K_DEFAULT) or TOP_K_DEFAULT)

    if not task_id or not isinstance(task_text, str) or not task_text.strip():
        raise HTTPException(status_code=400, detail="Missing task_id or non-empty task")

    SEARCH_CATEGORIES = {
        "Specific_Product",
        "Cheapest_Product",
        "Best_Fit_Specific",
        "Best_Fit_Vague",
        "Cheapest_Best_Fit_Specific",
        }

    # Search-only gate
    if task_category not in SEARCH_CATEGORIES:
        return {
            "shop_id": SHOP_ID,
            "backend": BACKEND,
            "task_category": task_category,
            "candidates": [],
            "final_urls": [],
            "raw_model_output": "SKIPPED_NON_SEARCH_CATEGORY",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "skipped": True,
        }

    # Tool-only path
    try:
        urls, usage = await asyncio.wait_for(
            run_tool_only_search(task_text.strip(), top_k),
            timeout=TIMEOUT_SEC,
        )

        # Normalize: list[str], de-dup, trim
        if not isinstance(urls, list):
            urls = []

        seen = set()
        clean: List[str] = []
        for u in urls:
            if isinstance(u, str) and u.startswith("http") and u not in seen:
                seen.add(u)
                clean.append(u)

        clean = clean[:top_k]

        return {
            "shop_id": SHOP_ID,
            "backend": BACKEND,
            "task_category": task_category,
            "candidates": [{"url": u} for u in clean],
            "final_urls": clean,
            "raw_model_output": json.dumps({"urls": clean}, ensure_ascii=False),
            "usage": usage if isinstance(usage, dict) else {"prompt_tokens": 0, "completion_tokens": 0},
        }

    except asyncio.TimeoutError:
        return {
            "shop_id": SHOP_ID,
            "backend": BACKEND,
            "task_category": task_category,
            "candidates": [],
            "final_urls": [],
            "raw_model_output": "TIMEOUT",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "timeout": True,
        }
    except Exception as e:
        return {
            "shop_id": SHOP_ID,
            "backend": BACKEND,
            "task_category": task_category,
            "candidates": [],
            "final_urls": [],
            "raw_model_output": f"ERROR: {type(e).__name__}: {str(e)}",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "error": True,
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SHOP_PORT", "8011"))
    uvicorn.run(app, host="0.0.0.0", port=port)
