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

MODEL_NAME = os.getenv("SHOP_LLM_MODEL", "openai:gpt-5-mini")

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
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

    system = SystemMessage(
        content=(
            f"You are the Shop agent for shop_id={SHOP_ID} using backend={BACKEND}. "
            "You can call tools to search, add to cart, or checkout if needed.\n\n"
            "Return ONLY a JSON object with these keys:\n"
            '  "urls": list of product/offer URLs (ranked, best first)\n'
            '  "cart_only_urls": URLs of products added to cart (if task is Add_To_Cart)\n'
            '  "checkout_only_urls": URLs of products successfully checked out (if task is Checkout/FindAndOrder)\n'
            '  "checkout_successful": true/false\n\n'
            "Do not include any other text besides the JSON."
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


@app.post("/run_task")
async def run_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input (from Buyer):
    {
      "task_id": "...",
      "task": "...",
      "task_category": "...",
      "top_k": 3
    }
    """
    task_id = payload.get("task_id")
    task_text = payload.get("task")
    task_category = payload.get("task_category", "")
    top_k = int(payload.get("top_k", TOP_K_DEFAULT) or TOP_K_DEFAULT)

    if not task_id or not isinstance(task_text, str) or not task_text.strip():
        raise HTTPException(status_code=400, detail="Missing task_id or non-empty task")

    try:
        final_text, usage = await asyncio.wait_for(
            _run_agent(task_text.strip(), top_k),
            timeout=TIMEOUT_SEC,
        )
        obj = _extract_json_obj(final_text)

        urls = obj.get("urls", []) if isinstance(obj.get("urls", []), list) else []
        cart_only_urls = obj.get("cart_only_urls", []) if isinstance(obj.get("cart_only_urls", []), list) else []
        checkout_only_urls = obj.get("checkout_only_urls", []) if isinstance(obj.get("checkout_only_urls", []), list) else []
        checkout_successful = bool(obj.get("checkout_successful", False))

        return {
            "shop_id": SHOP_ID,
            "backend": BACKEND,
            "task_category": task_category,
            "candidates": _to_candidates([u for u in urls if isinstance(u, str)], top_k),
            "final_urls": [u for u in urls if isinstance(u, str)],
            "cart_only_urls": [u for u in cart_only_urls if isinstance(u, str)],
            "checkout_only_urls": [u for u in checkout_only_urls if isinstance(u, str)],
            "checkout_successful": checkout_successful,
            "raw_model_output": final_text,  # debug; you can remove later
            "usage": usage,
        }
    except asyncio.TimeoutError:
        return {
            "shop_id": SHOP_ID,
            "backend": BACKEND,
            "task_category": task_category,
            "candidates": [],
            "final_urls": [],
            "cart_only_urls": [],
            "checkout_only_urls": [],
            "checkout_successful": False,
            "raw_model_output": "TIMEOUT",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "error": "timeout",
        }
    except Exception as e:
        return {
            "shop_id": SHOP_ID,
            "backend": BACKEND,
            "task_category": task_category,
            "candidates": [],
            "final_urls": [],
            "cart_only_urls": [],
            "checkout_only_urls": [],
            "checkout_successful": False,
            "raw_model_output": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SHOP_PORT", "8011"))
    uvicorn.run(app, host="0.0.0.0", port=port)
