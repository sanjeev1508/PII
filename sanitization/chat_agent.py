"""
Chat agent: stores report Q&A conversations in a separate Qdrant collection,
and answers questions about the report using OpenAI with retrieved context.
"""
import asyncio
import hashlib
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

CHAT_COLLECTION = os.getenv("QDRANT_CHAT_COLLECTION", "report_chat_history")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
VECTOR_DIM = 12
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")


# --------------------------------------------------------------------------- #
#  Vector helpers
# --------------------------------------------------------------------------- #

def _text_to_vector(text: str, dim: int = VECTOR_DIM) -> list[float]:
    values: list[float] = []
    seed_text = text or "empty"
    for idx in range(dim):
        digest = hashlib.sha256(f"{seed_text}:{idx}".encode()).digest()
        num = int.from_bytes(digest[:8], "big", signed=False)
        values.append(((num % 2_000_000) / 1_000_000.0) - 1.0)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


# --------------------------------------------------------------------------- #
#  Qdrant helpers
# --------------------------------------------------------------------------- #

def _ensure_chat_collection() -> None:
    url = f"{QDRANT_URL}/collections/{CHAT_COLLECTION}"
    if requests.get(url, timeout=5).status_code == 200:
        return
    requests.put(
        url,
        json={"vectors": {"size": VECTOR_DIM, "distance": "Cosine"}},
        timeout=5,
    ).raise_for_status()


def _search_similar(query: str, top_k: int = 6) -> list[dict]:
    vec = _text_to_vector(query)
    res = requests.post(
        f"{QDRANT_URL}/collections/{CHAT_COLLECTION}/points/search",
        json={"vector": vec, "limit": top_k, "with_payload": True},
        timeout=5,
    )
    if res.status_code != 200:
        return []
    return [hit["payload"] for hit in res.json().get("result", [])]


def _store_turn(session_id: str, role: str, content: str, source_file: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    turn_id = int(
        hashlib.sha256(f"{session_id}:{role}:{content}:{now}".encode()).hexdigest()[:16],
        16,
    )
    payload = {
        "session_id": session_id,
        "source_file": source_file,
        "role": role,
        "content": content,
        "stored_at_utc": now,
    }
    requests.put(
        f"{QDRANT_URL}/collections/{CHAT_COLLECTION}/points?wait=true",
        json={"points": [{"id": turn_id, "vector": _text_to_vector(content), "payload": payload}]},
        timeout=5,
    ).raise_for_status()


# --------------------------------------------------------------------------- #
#  Async OpenAI chat
# --------------------------------------------------------------------------- #

async def _ask_openai(question: str, report_text: str, history: list[dict]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API key not configured."

    client = AsyncOpenAI(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert assistant analyzing a PII sanitization report. "
                "Answer questions clearly and concisely based on the report context provided. "
                "If the answer is not in the context, say so honestly.\n\n"
                f"=== REPORT CONTEXT ===\n{report_text[:6000]}\n=== END CONTEXT ==="
            ),
        }
    ]

    # Include recent history turns
    for turn in history[-8:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": question})

    response = await client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------------- #

def chat_with_report(
    session_id: str,
    question: str,
    report_text: str,
    source_file: str,
) -> str:
    """Store the question, retrieve context, call LLM, store answer, return answer."""
    _ensure_chat_collection()

    # Retrieve similar past turns for context
    past = _search_similar(question, top_k=6)
    history = [{"role": p["role"], "content": p["content"]} for p in past if "role" in p and "content" in p]

    # Store user turn
    _store_turn(session_id, "user", question, source_file)

    # Call LLM
    answer = _run_async(_ask_openai(question, report_text, history))

    # Store assistant turn
    _store_turn(session_id, "assistant", answer, source_file)

    return answer
