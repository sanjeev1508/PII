"""
LLM-based sensitivity classifier for borderline entity candidates.
"""
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


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
    return _llm_is_sensitive_local(
        entity_type=entity_type,
        value=value,
        confidence=confidence,
        context=context,
        model=model,
    )


def llm_classify_batch(
    candidates: list[dict],
    model: str,
    batch_size: int = 50,
    reference_path: str = "outputs/llm/reference_scores.json",
    skills_path: str = "SKILLS.md",
) -> tuple[dict[str, str], list[dict]]:
    """
    Classify borderline candidates in batches.
    Returns:
      - decision map: candidate_id -> ("MASK" | "KEEP" | "ERROR")
      - json rows suitable for report rendering
    """
    if not candidates:
        return {}, []

    client = _get_client()
    decisions: dict[str, str] = {}
    json_rows: list[dict] = []
    skills_text = _read_text_file(skills_path, default="No additional skill profile provided.")
    reference_text = _read_text_file(reference_path, default="{}")

    for i in range(0, len(candidates), batch_size):
        chunk = candidates[i : i + batch_size]
        payload = {
            "instructions": "Return JSON only. For each item decide MASK or KEEP.",
            "items": [
                {
                    "id": item["id"],
                    "entity_type": item["entity_type"],
                    "value": item["value"],
                    "confidence": item["confidence"],
                    "context": item["context"],
                }
                for item in chunk
            ],
            "response_schema": {"results": [{"id": "string", "decision": "MASK|KEEP"}]},
        }
        prompt = (
            "You are a strict PII/legal-sensitive classifier.\n"
            "Use the skill profile and references to decide whether each candidate should be masked.\n"
            "Output strictly JSON with this shape:\n"
            '{"results":[{"id":"<id>","decision":"MASK|KEEP","reason":"<short>"}]}\n'
            f"SKILL PROFILE:\n{skills_text}\n"
            f"REFERENCES (local confidence snapshot):\n{reference_text}\n"
            f"Input:\n{json.dumps(payload, ensure_ascii=True)}"
        )

        try:
            text = ""
            if hasattr(client, "responses"):
                response = client.responses.create(
                    model=model,
                    temperature=0,
                    max_output_tokens=1200,
                    input=prompt,
                )
                text = (getattr(response, "output_text", "") or "").strip()
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    max_tokens=1200,
                    messages=[
                        {"role": "system", "content": "Return JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                )
                text = (response.choices[0].message.content or "").strip()

            parsed = _extract_json(text)
            result_items = parsed.get("results", [])
            result_map = {str(x["id"]): str(x.get("decision", "")).upper() for x in result_items if "id" in x}
            reason_map = {str(x["id"]): str(x.get("reason", "")) for x in result_items if "id" in x}
            for item in chunk:
                decision = "MASK" if result_map.get(item["id"]) == "MASK" else "KEEP"
                decisions[item["id"]] = decision
                json_rows.append(
                    {
                        "id": item["id"],
                        "entity_type": item["entity_type"],
                        "value": item["value"],
                        "confidence": item["confidence"],
                        "decision": decision,
                        "reason": reason_map.get(item["id"], ""),
                    }
                )
        except Exception:
            for item in chunk:
                decisions[item["id"]] = "ERROR"
                json_rows.append(
                    {
                        "id": item["id"],
                        "entity_type": item["entity_type"],
                        "value": item["value"],
                        "confidence": item["confidence"],
                        "decision": "ERROR",
                        "reason": "Batch classification error",
                    }
                )

    return decisions, json_rows


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON found in model response")
    return json.loads(match.group(0))


def _read_text_file(path: str, default: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return default
