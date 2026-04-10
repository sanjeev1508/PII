"""
Agentic orchestration layer for sanitization pipeline.
"""
import os
import time

from sanitization.detector import detect_entities
from sanitization.extractor import extract_pages
from sanitization.masker import mask_pdf
from sanitization.qdrant_agent import process_candidates_above_threshold
from sanitization.reporter import generate_report


def run_sanitization_pipeline(input_path: str, source_file: str, config: dict, work_dir: str) -> dict:
    """
    Orchestrate hybrid sanitization agents:
    Extraction (OCR/native) -> Detection (spaCy/Presidio + LLM fallback)
    -> Memory agent (Qdrant pre-LLM reuse + async LLM judge)
    -> Masking -> Reporting.
    """
    start = time.time()
    pages = extract_pages(input_path)
    total_pages = len(pages)
    mode_summary: dict[str, int] = {}
    for p in pages:
        mode_summary[p["method"]] = mode_summary.get(p["method"], 0) + 1

    _, _, _, candidate_pool = detect_entities(pages, config)

    agent_result = {
        "stored_count": 0,
        "reviewed_count": 0,
        "accepted_rows": [],
        "review_rows": [],
        "entities_to_mask": [],
    }
    try:
        agent_result = process_candidates_above_threshold(
            candidates=candidate_pool,
            source_file=source_file,
            config=config,
        )
        print(
            "[AGENT] Reviewed "
            f"{agent_result.get('reviewed_count', 0)} candidates (>50 conf), "
            f"accepted {len(agent_result.get('accepted_rows', []))}, "
            f"stored {agent_result.get('stored_count', 0)}"
        )
    except Exception as err:
        print(f"[AGENT] Judge/Store step skipped due to error: {err}")

    masked_path = os.path.join(work_dir, "masked.pdf")
    final_entities_to_mask = agent_result.get("entities_to_mask", [])
    mask_pdf(input_path, masked_path, final_entities_to_mask)

    review_log = []
    for row in agent_result.get("review_rows", []):
        is_masked = str(row.get("pii_decision", "")) == "accepted_pii"
        route = str(row.get("route", "llm"))
        review_log.append(
            {
                "page": int(row.get("page", -1)),
                "type": str(row.get("type", "")),
                "value": str(row.get("value", "")),
                "confidence": float(row.get("confidence", 0.0)),
                "decision": "MASKED" if is_masked else "REJECTED",
                "reason": (
                    "above_direct_spacy_threshold"
                    if route == "direct_spacy"
                    else "llm_pii"
                    if is_masked
                    else "llm_not_pii"
                ),
                "review_method": "THRESHOLD" if route == "direct_spacy" else "LLM",
                "llm_classification": None if route == "direct_spacy" else ("PII" if is_masked else "NOT_PII"),
            }
        )
    llm_json_results = [
        {
            "id": str(row.get("id", "")),
            "entity_type": str(row.get("type", "")),
            "value": str(row.get("value", "")),
            "decision": "MASK" if str(row.get("pii_decision", "")) == "accepted_pii" else "KEEP",
            "reason": str(row.get("pii_reason", "")),
            "confidence": row.get("confidence", ""),
        }
        for row in agent_result.get("review_rows", [])
    ]

    report_path = os.path.join(work_dir, "report.pdf")
    elapsed = round(time.time() - start, 3)
    generate_report(
        output_path=report_path,
        source_file=source_file,
        total_pages=total_pages,
        extraction_mode_summary=mode_summary,
        entities=final_entities_to_mask,
        review_log=review_log,
        llm_json_results=llm_json_results,
        processing_seconds=elapsed,
        accepted_agent_rows=agent_result.get("accepted_rows", []),
        agent_review_rows=agent_result.get("review_rows", []),
    )
    return {
        "masked_path": masked_path,
        "report_path": report_path,
    }
