"""
Qdrant-backed async LLM judge pipeline for candidates above confidence threshold.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


def _normalize_confidence_threshold(raw_threshold: float) -> float:
    return raw_threshold / 100.0 if raw_threshold > 1 else raw_threshold


def _text_to_vector(text: str, dim: int = 12) -> list[float]:
    values: list[float] = []
    seed_text = text or "empty"
    for idx in range(dim):
        digest = hashlib.sha256(f"{seed_text}:{idx}".encode("utf-8")).digest()
        num = int.from_bytes(digest[:8], "big", signed=False)
        values.append(((num % 2000000) / 1000000.0) - 1.0)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def _ensure_collection(base_url: str, collection: str, vector_dim: int, timeout_s: float) -> None:
    get_url = f"{base_url}/collections/{collection}"
    res = requests.get(get_url, timeout=timeout_s)
    if res.status_code == 200:
        return
    create_res = requests.put(
        f"{base_url}/collections/{collection}",
        json={"vectors": {"size": vector_dim, "distance": "Cosine"}},
        timeout=timeout_s,
    )
    create_res.raise_for_status()


def _extract_json(text: str) -> dict:
    body = (text or "").strip()
    try:
        return json.loads(body)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", body)
    if not match:
        return {"results": []}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {"results": []}


async def _classify_chunk(*, client: AsyncOpenAI, model: str, chunk: list[dict]) -> dict[str, dict]:
    payload = {
        "instructions": "Return JSON only. Decide PII or NOT_PII for each item.",
        "items": [{"id": item["id"], "entity_type": item["entity_type"], "value": item["value"]} for item in chunk],
    }
    response = await client.responses.create(
        model=model,
        temperature=0,
        max_output_tokens=1000,
        input=[
            {
                "role": "system",
                "content": (
                    "You classify legal/business spans. For each item choose exactly one label: PII or NOT_PII.\n"
                    "Mark as PII when it belongs to either of these classes:\n"
                    "1) Standard PII: PERSON, EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD, ADDRESS.\n"
                    "2) Corporate confidentiality in legal agreements: PARTY_NAME, FINANCIAL_AMOUNT, JURISDICTION.\n"
                    "Be permissive for these listed classes. Return NOT_PII only for clear non-sensitive/generic noise."
                ),
            },
            {
                "role": "user",
                "content": 'Return strict JSON: {"results":[{"id":"...","decision":"PII|NOT_PII"}]}\n'
                f"Input:\n{json.dumps(payload, ensure_ascii=True)}",
            },
        ],
    )
    parsed = _extract_json(getattr(response, "output_text", "") or "")
    out: dict[str, dict] = {}
    for item in parsed.get("results", []):
        cid = str(item.get("id", ""))
        if not cid:
            continue
        raw_decision = str(item.get("decision", "")).upper()
        is_pii = raw_decision == "PII"
        out[cid] = {
            "accepted_as_pii": is_pii,
            "pii_decision": "accepted_pii" if is_pii else "not_accepted",
            "pii_reason": f"llm_binary_decision:{raw_decision or 'EMPTY'}",
        }
    return out


async def _classify_candidates_async(candidates: list[dict], model: str, batch_size: int) -> dict[str, dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not candidates:
        return {}
    client = AsyncOpenAI(api_key=api_key)
    tasks = []
    for i in range(0, len(candidates), batch_size):
        tasks.append(_classify_chunk(client=client, model=model, chunk=candidates[i : i + batch_size]))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    merged: dict[str, dict] = {}
    for result in results:
        if not isinstance(result, Exception):
            merged.update(result)
    return merged


def _run_async_job(coro):
    """
    Run async work safely from both sync and async contexts.
    FastAPI request handlers already run inside an event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


def process_candidates_above_threshold(candidates: list[dict], source_file: str, config: dict | None = None) -> dict:
    cfg = config or {}
    qdrant_cfg = cfg.get("qdrant", {})
    if not bool(qdrant_cfg.get("enabled", True)):
        return {
            "stored_count": 0,
            "accepted_rows": [],
            "reviewed_count": 0,
            "review_rows": [],
            "entities_to_mask": [],
        }

    base_url = (qdrant_cfg.get("url") or os.getenv("QDRANT_URL") or "http://localhost:6333").rstrip("/")
    collection = qdrant_cfg.get("collection") or os.getenv("QDRANT_COLLECTION") or "unreported_candidates"
    min_confidence = _normalize_confidence_threshold(
        float(qdrant_cfg.get("min_confidence_percent", os.getenv("QDRANT_MIN_CONFIDENCE_PERCENT", 50)))
    )
    timeout_s = float(qdrant_cfg.get("timeout_seconds", 8))
    vector_dim = int(qdrant_cfg.get("vector_dim", 12))
    model = str(qdrant_cfg.get("agent_model", "gpt-4.1-mini"))
    batch_size = int(qdrant_cfg.get("agent_batch_size", 40))
    direct_mask_threshold = float((cfg.get("orchestration", {}) or {}).get("direct_spacy_confidence", 0.90))
    strongly_sensitive_types = {
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "ADDRESS",
        "PARTY_NAME",
        "FINANCIAL_AMOUNT",
        "JURISDICTION",
    }

    filtered_raw = [dict(row) for row in candidates if float(row.get("confidence", 0.0)) > min_confidence]
    if not filtered_raw:
        return {
            "stored_count": 0,
            "accepted_rows": [],
            "reviewed_count": 0,
            "review_rows": [],
            "entities_to_mask": [],
        }

    _ensure_collection(base_url, collection, vector_dim=vector_dim, timeout_s=timeout_s)

    review_rows: list[dict] = []
    llm_candidates: list[dict] = []
    item_by_id: dict[str, dict] = {}
    for idx, item in enumerate(filtered_raw, start=1):
        cid = f"u_{idx}"
        entity_type = str(item.get("type", "UNKNOWN"))
        value = str(item.get("value", ""))
        item_by_id[cid] = item
        confidence = float(item.get("confidence", 0.0))
        if confidence > direct_mask_threshold:
            review_rows.append(
                {
                    "id": cid,
                    "page": item.get("page", -1),
                    "type": entity_type,
                    "value": value,
                    "confidence": confidence,
                    "route": "direct_spacy",
                    "similarity": "-",
                    "pii_decision": "accepted_pii",
                    "accepted_as_pii": True,
                    "pii_reason": "above_direct_spacy_threshold",
                }
            )
        else:
            llm_candidates.append({"id": cid, "entity_type": entity_type, "value": value})

    if llm_candidates:
        llm_decisions = _run_async_job(
            _classify_candidates_async(llm_candidates, model=model, batch_size=batch_size)
        )
        for cand in llm_candidates:
            cid = cand["id"]
            item = item_by_id[cid]
            dec = llm_decisions.get(
                cid,
                {"accepted_as_pii": False, "pii_decision": "not_accepted", "pii_reason": "classification_unavailable"},
            )
            entity_type = str(cand["entity_type"]).upper()
            # Guardrail: keep listed business/legal sensitive classes from being over-rejected.
            if entity_type in strongly_sensitive_types and not dec.get("accepted_as_pii", False):
                dec = {
                    "accepted_as_pii": True,
                    "pii_decision": "accepted_pii",
                    "pii_reason": "policy_sensitive_type_override",
                }
            review_rows.append(
                {
                    "id": cid,
                    "page": item.get("page", -1),
                    "type": cand["entity_type"],
                    "value": cand["value"],
                    "confidence": item.get("confidence", 0.0),
                    "route": "llm",
                    "similarity": "-",
                    "pii_decision": dec["pii_decision"],
                    "accepted_as_pii": dec["accepted_as_pii"],
                    "pii_reason": dec["pii_reason"],
                }
            )

    trace_map = {row["id"]: row for row in review_rows}
    enriched_rows = []
    accepted_rows = []
    entities_to_mask = []
    entity_id_counter: dict[str, int] = {}
    for idx, item in enumerate(filtered_raw, start=1):
        cid = f"u_{idx}"
        trace = trace_map.get(cid, {})
        enriched = {
            **item,
            "accepted_as_pii": bool(trace.get("accepted_as_pii", False)),
            "pii_decision": str(trace.get("pii_decision", "not_accepted")),
            "pii_reason": str(trace.get("pii_reason", "")),
            "decision_route": str(trace.get("route", "unknown")),
            "memory_similarity": trace.get("similarity", "-"),
        }
        enriched_rows.append(enriched)
        if enriched["accepted_as_pii"]:
            accepted_rows.append(enriched)
            entity_type = str(item.get("type", "UNKNOWN"))
            entity_id_counter[entity_type] = entity_id_counter.get(entity_type, 0) + 1
            eid = entity_id_counter[entity_type]
            entities_to_mask.append(
                {
                    "page": int(item.get("page", -1)),
                    "type": entity_type,
                    "value": str(item.get("value", "")),
                    "start": int(item.get("start", 0)),
                    "end": int(item.get("end", 0)),
                    "confidence": float(item.get("confidence", 0.0)),
                    "mask": str(item.get("mask_template", f"[{entity_type}-{{id}}]")).format(id=eid),
                    "llm_reviewed": True,
                    "llm_classification": "PII",
                    "review_method": "LLM",
                }
            )

    now = datetime.now(timezone.utc).isoformat()
    points = []
    for idx, item in enumerate(enriched_rows, start=1):
        value = str(item.get("value", "")).strip()
        entity_type = str(item.get("type", "UNKNOWN"))
        page = int(item.get("page", -1))
        confidence = float(item.get("confidence", 0.0))
        reason = str(item.get("reason", ""))
        point_id = int(
            hashlib.sha256(f"{source_file}|{entity_type}|{value}|{page}|{confidence:.6f}|{reason}".encode("utf-8")).hexdigest()[
                :16
            ],
            16,
        )
        payload = {
            "source_file": source_file,
            "entity_type": entity_type,
            "value": value,
            "page": page,
            "confidence": confidence,
            "decision": "REVIEWED",
            "reason": reason,
            "review_method": "LLM",
            "accepted_as_pii": item.get("accepted_as_pii", False),
            "pii_decision": item.get("pii_decision", "not_accepted"),
            "pii_reason": item.get("pii_reason", ""),
            "decision_route": item.get("decision_route", "unknown"),
            "memory_similarity": item.get("memory_similarity", "-"),
            "stored_at_utc": now,
            "sequence": idx,
        }
        points.append({"id": point_id, "vector": _text_to_vector(f"{entity_type} {value}", dim=vector_dim), "payload": payload})

    # Clear all existing points before inserting fresh batch
    delete_res = requests.post(
        f"{base_url}/collections/{collection}/points/delete?wait=true",
        json={"filter": {}},
        timeout=timeout_s,
    )
    if delete_res.status_code not in (200, 404):
        delete_res.raise_for_status()
    print(f"[QDRANT] Cleared existing points from collection '{collection}'")

    upsert_res = requests.put(
        f"{base_url}/collections/{collection}/points?wait=true",
        json={"points": points},
        timeout=timeout_s,
    )
    upsert_res.raise_for_status()
    return {
        "stored_count": len(points),
        "accepted_rows": accepted_rows,
        "reviewed_count": len(enriched_rows),
        "review_rows": sorted(review_rows, key=lambda x: x.get("id", "")),
        "entities_to_mask": entities_to_mask,
    }

