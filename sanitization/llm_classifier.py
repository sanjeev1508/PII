"""
LLM-based sensitivity classifier for borderline entity candidates.
"""
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
import requests


load_dotenv()


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=api_key)


def _llm_is_sensitive_local(
    *,
    entity_type: str,
    value: str,
    confidence: float,
    context: str,
    model: str,
) -> bool:
    client = _get_client()
    prompt = (
        "You are a strict PII/legal-sensitive classifier.\n"
        "Answer with exactly one token: SENSITIVE or NOT_SENSITIVE.\n"
        "Mark as SENSITIVE only when the span should be masked in legal/business documents.\n\n"
        f"Entity type: {entity_type}\n"
        f"Detected span: {value}\n"
        f"Detector confidence: {confidence:.3f}\n"
        f"Local context: {context}\n"
    )
    text = ""
    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            temperature=0,
            max_output_tokens=5,
            input=prompt,
        )
        text = (getattr(response, "output_text", "") or "").strip().upper()
    else:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=5,
            messages=[
                {"role": "system", "content": "You are a strict PII/legal-sensitive classifier."},
                {"role": "user", "content": prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip().upper()
    return "SENSITIVE" in text and "NOT_SENSITIVE" not in text


def _call_mcp_llm(
    *,
    entity_type: str,
    value: str,
    confidence: float,
    context: str,
    model: str,
) -> bool:
    server_url = os.getenv("MCP_SERVER_URL")
    tool_name = os.getenv("MCP_LLM_TOOL", "openai_sensitive_review")
    if not server_url:
        raise ValueError("MCP_SERVER_URL is not configured for MCP LLM review.")

    payload = {
        "entity_type": entity_type,
        "value": value,
        "confidence": confidence,
        "context": context,
        "model": model,
    }
    try:
        import mcp_use  # type: ignore
    except Exception:
        mcp_use = None

    result = None
    if mcp_use is not None and hasattr(mcp_use, "call_tool"):
        result = mcp_use.call_tool(server_url=server_url, tool_name=tool_name, arguments=payload)
    else:
        resp = requests.post(
            f"{server_url.rstrip('/')}/call_tool",
            json={"tool_name": tool_name, "arguments": payload},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
    if isinstance(result, dict):
        if "sensitive" in result:
            return bool(result["sensitive"])
        if "result" in result:
            return bool(result["result"])
    return bool(result)


def llm_is_sensitive(
    *,
    entity_type: str,
    value: str,
    confidence: float,
    context: str,
    model: str,
) -> bool:
    """
    Return True if model classifies the candidate as sensitive.
    """
    if os.getenv("USE_MCP_LLM", "false").lower() == "true":
        try:
            return _call_mcp_llm(
                entity_type=entity_type,
                value=value,
                confidence=confidence,
                context=context,
                model=model,
            )
        except Exception as exc:
            print(f"[LLM] MCP review failed: {exc}")

    return _llm_is_sensitive_local(
        entity_type=entity_type,
        value=value,
        confidence=confidence,
        context=context,
        model=model,
    )
