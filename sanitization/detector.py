"""
Entity detection pipeline using Microsoft Presidio + spaCy + custom regex patterns.
Applies validators to reduce false positives.
"""
import json
from pathlib import Path

import yaml
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from sanitization.llm_classifier import llm_classify_batch

from sanitization.validators import (
    is_real_address,
    is_real_financial_amount,
    is_real_party_name,
    is_real_ssn,
    luhn_check,
)


def load_config(config_path: str = "configs/entities.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_analyzer() -> AnalyzerEngine:
    """Build Presidio analyzer with spaCy NLP backend and custom recognizers."""
    print("[DETECT] Building analyzer")
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    analyzer.registry.add_recognizer(_financial_amount_recognizer())
    analyzer.registry.add_recognizer(_party_name_recognizer())
    analyzer.registry.add_recognizer(_jurisdiction_recognizer())
    return analyzer


def _financial_amount_recognizer() -> PatternRecognizer:
    patterns = [
        Pattern("FINANCIAL_AMOUNT_DOLLAR", r"\$[\d,]+(?:\.\d{1,2})?(?:\s?(?:million|billion|thousand))?", 0.85),
        Pattern("FINANCIAL_AMOUNT_PLAIN", r"\b\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?\b", 0.70),
    ]
    return PatternRecognizer(supported_entity="FINANCIAL_AMOUNT", patterns=patterns, supported_language="en")


def _party_name_recognizer() -> PatternRecognizer:
    patterns = [
        Pattern(
            "CORP_ENTITY",
            r"\b[A-Z][A-Za-z&\s\.\,\']{2,60}(?:Corporation|Corp|LLC|Ltd|Inc|L\.P\.|LLP|Company|Co\.|PLC|N\.A\.)\b",
            0.80,
        )
    ]
    return PatternRecognizer(supported_entity="PARTY_NAME", patterns=patterns, supported_language="en")


def _jurisdiction_recognizer() -> PatternRecognizer:
    patterns = [
        Pattern(
            "JURISDICTION_COURT",
            r"\b(?:Court of|District Court|Supreme Court|Circuit Court|Court of Appeals)[^\n]{0,60}",
            0.75,
        ),
        Pattern(
            "JURISDICTION_STATE_LAW",
            r"\b(?:laws? of the State of|governed by the laws? of|organized under the laws? of)\s+[A-Z][a-zA-Z\s]+",
            0.75,
        ),
    ]
    return PatternRecognizer(supported_entity="JURISDICTION", patterns=patterns, supported_language="en")


def detect_entities(pages: list[dict], config: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Run detection on all extracted pages.
    Returns list of detected entity dicts with page, type, value, start, end, confidence.
    """
    print("[DETECT] Starting entity detection")
    try:
        analyzer = build_analyzer()
    except Exception as e:
        print(f"[DETECT] Analyzer init failed: {e}")
        return [], [], [], []

    enabled_types = {e["type"]: e for e in config["entities"] if e.get("enabled", True)}
    llm_cfg = config.get("llm_review", {})
    llm_enabled = bool(llm_cfg.get("enabled", False))
    llm_min_conf = float(llm_cfg.get("min_confidence", 0.5))
    llm_model = llm_cfg.get("model", "gpt-4.1-mini")
    llm_fail_open = bool(llm_cfg.get("fail_open_on_error", True))
    llm_review_entity_types = set(llm_cfg.get("review_entity_types", []))
    llm_max_reviews = int(llm_cfg.get("max_reviews_per_document", 250))
    llm_batch_size = int(llm_cfg.get("batch_size", 50))
    llm_reference_path = llm_cfg.get("reference_file", "outputs/llm/reference_scores.json")
    llm_skills_path = llm_cfg.get("skills_file", "SKILLS.md")
    llm_decision_cache: dict[tuple[str, str], tuple[bool, str]] = {}
    pending_reviews: list[dict] = []
    pending_unique: dict[tuple[str, str], dict] = {}
    llm_json_results: list[dict] = []
    all_entities = []
    review_log = []
    candidate_pool = []
    entity_id_counter = {}
    score_accumulator: dict[str, list[float]] = {}

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        if not text.strip():
            continue

        presidio_entities = _map_to_presidio_types(list(enabled_types.keys()))
        try:
            results = analyzer.analyze(text=text, language="en", entities=presidio_entities)
        except Exception as e:
            print(f"[DETECT] Analyze failed on page {page_num}: {e}")
            continue

        if not results:
            continue

        for result in results:
            entity_type = _map_from_presidio(result.entity_type)
            if entity_type not in enabled_types:
                continue
            cfg = enabled_types[entity_type]
            max_conf = float(cfg.get("confidence_threshold", 0.5))
            score = float(result.score)
            score_accumulator.setdefault(entity_type, []).append(score)
            if score < llm_min_conf:
                review_log.append(
                    {
                        "page": page_num,
                        "type": entity_type,
                        "value": text[result.start : result.end].strip(),
                        "confidence": round(score, 3),
                        "decision": "REJECTED",
                        "reason": "below_min_threshold",
                        "review_method": "THRESHOLD",
                    }
                )
                continue

            value = text[result.start : result.end].strip()
            if not _validate(entity_type, value, cfg):
                review_log.append(
                    {
                        "page": page_num,
                        "type": entity_type,
                        "value": value,
                        "confidence": round(score, 3),
                        "decision": "REJECTED",
                        "reason": "validator_rejected",
                        "review_method": "VALIDATOR",
                    }
                )
                continue

            candidate_pool.append(
                {
                    "page": page_num,
                    "type": entity_type,
                    "value": value,
                    "start": result.start,
                    "end": result.end,
                    "confidence": round(score, 3),
                    "mask_template": cfg.get("mask_template", f"[{entity_type}-{{id}}]"),
                }
            )

            allow_mask = score >= max_conf
            should_review_type = not llm_review_entity_types or entity_type in llm_review_entity_types
            if not allow_mask and llm_enabled and score < max_conf and should_review_type:
                context_start = max(0, result.start - 80)
                context_end = min(len(text), result.end + 80)
                context = text[context_start:context_end].replace("\n", " ").strip()
                cache_key = (entity_type, value.lower())
                if cache_key not in pending_unique and len(pending_unique) < llm_max_reviews:
                    candidate_id = f"cand_{len(pending_unique) + 1}"
                    pending_unique[cache_key] = {
                        "id": candidate_id,
                        "entity_type": entity_type,
                        "value": value,
                        "confidence": round(score, 3),
                        "context": context,
                    }
                pending_reviews.append(
                    {
                        "page": page_num,
                        "type": entity_type,
                        "value": value,
                        "start": result.start,
                        "end": result.end,
                        "confidence": round(score, 3),
                        "mask_template": cfg.get("mask_template", f"[{entity_type}-{{id}}]"),
                        "cache_key": cache_key,
                    }
                )
                continue

            if not allow_mask:
                review_log.append(
                    {
                        "page": page_num,
                        "type": entity_type,
                        "value": value,
                        "confidence": round(score, 3),
                        "decision": "REJECTED",
                        "reason": (
                            "below_max_threshold"
                        ),
                        "review_method": "THRESHOLD",
                    }
                )
                continue

            entity_id_counter[entity_type] = entity_id_counter.get(entity_type, 0) + 1
            eid = entity_id_counter[entity_type]
            all_entities.append(
                {
                    "page": page_num,
                    "type": entity_type,
                    "value": value,
                    "start": result.start,
                    "end": result.end,
                    "confidence": round(score, 3),
                    "mask": cfg.get("mask_template", f"[{entity_type}-{{id}}]").format(id=eid),
                    "llm_reviewed": False,
                    "llm_classification": None,
                    "review_method": "THRESHOLD",
                }
            )
            review_log.append(
                {
                    "page": page_num,
                    "type": entity_type,
                    "value": value,
                    "confidence": round(score, 3),
                    "decision": "MASKED",
                    "reason": "above_max_threshold",
                    "review_method": "THRESHOLD",
                }
            )

    # Always persist confidence reference snapshot for this run, even if no LLM review happens.
    _write_reference_file(llm_reference_path, score_accumulator)

    if pending_reviews:
        if pending_unique:
            batch_candidates = list(pending_unique.values())
            print(f"[DETECT] LLM batch review candidates: {len(batch_candidates)}")
            llm_results, llm_json_results = llm_classify_batch(
                batch_candidates,
                model=llm_model,
                batch_size=llm_batch_size,
                reference_path=llm_reference_path,
                skills_path=llm_skills_path,
            )
            for key, item in pending_unique.items():
                decision = llm_results.get(item["id"], "ERROR")
                if decision == "MASK":
                    llm_decision_cache[key] = (True, "SENSITIVE")
                elif decision == "KEEP":
                    llm_decision_cache[key] = (False, "NOT_SENSITIVE")
                else:
                    llm_decision_cache[key] = (llm_fail_open, "ERROR")

        for item in pending_reviews:
            cache_key = item["cache_key"]
            if cache_key not in llm_decision_cache:
                llm_decision_cache[cache_key] = (llm_fail_open, "SKIPPED_LIMIT")
            allow_mask, llm_classification = llm_decision_cache[cache_key]

            if allow_mask:
                entity_type = item["type"]
                entity_id_counter[entity_type] = entity_id_counter.get(entity_type, 0) + 1
                eid = entity_id_counter[entity_type]
                all_entities.append(
                    {
                        "page": item["page"],
                        "type": entity_type,
                        "value": item["value"],
                        "start": item["start"],
                        "end": item["end"],
                        "confidence": item["confidence"],
                        "mask": item["mask_template"].format(id=eid),
                        "llm_reviewed": True,
                        "llm_classification": llm_classification,
                        "review_method": "LLM",
                    }
                )
                review_log.append(
                    {
                        "page": item["page"],
                        "type": entity_type,
                        "value": item["value"],
                        "confidence": item["confidence"],
                        "decision": "MASKED",
                        "reason": (
                            "llm_sensitive"
                            if llm_classification == "SENSITIVE"
                            else "llm_error_masked"
                            if llm_classification == "ERROR"
                            else "llm_skipped_limit_masked"
                        ),
                        "review_method": "LLM",
                        "llm_classification": llm_classification,
                    }
                )
            else:
                review_log.append(
                    {
                        "page": item["page"],
                        "type": item["type"],
                        "value": item["value"],
                        "confidence": item["confidence"],
                        "decision": "REJECTED",
                        "reason": "llm_not_sensitive",
                        "review_method": "LLM",
                        "llm_classification": llm_classification,
                    }
                )
    else:
        print("[DETECT] LLM batch review candidates: 0")

    print(f"[DETECT] Detection complete. Found {len(all_entities)} entities")
    return all_entities, review_log, llm_json_results, candidate_pool


def _validate(entity_type: str, value: str, cfg: dict) -> bool:
    if entity_type == "CREDIT_CARD":
        return luhn_check(value) if cfg.get("validate_luhn") else True
    if entity_type == "ADDRESS":
        return is_real_address(value) if cfg.get("validate_format") else True
    if entity_type == "PARTY_NAME":
        return is_real_party_name(value, cfg.get("max_token_length", 10))
    if entity_type == "FINANCIAL_AMOUNT":
        return is_real_financial_amount(value)
    if entity_type == "US_SSN":
        return is_real_ssn(value)
    return True


def _map_to_presidio_types(types: list[str]) -> list[str]:
    mapping = {
        "PERSON": "PERSON",
        "EMAIL_ADDRESS": "EMAIL_ADDRESS",
        "PHONE_NUMBER": "PHONE_NUMBER",
        "CREDIT_CARD": "CREDIT_CARD",
        "US_SSN": "US_SSN",
        "FINANCIAL_AMOUNT": "FINANCIAL_AMOUNT",
        "PARTY_NAME": "PARTY_NAME",
        "ADDRESS": "LOCATION",
        "JURISDICTION": "JURISDICTION",
    }
    return list({mapping.get(t, t) for t in types})


def _map_from_presidio(presidio_type: str) -> str:
    reverse = {"LOCATION": "ADDRESS"}
    return reverse.get(presidio_type, presidio_type)


def _write_reference_file(reference_path: str, score_accumulator: dict[str, list[float]]) -> None:
    path = Path(reference_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        entity_type: {
            "count": len(scores),
            "avg_confidence": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "min_confidence": round(min(scores), 4) if scores else 0.0,
            "max_confidence": round(max(scores), 4) if scores else 0.0,
        }
        for entity_type, scores in score_accumulator.items()
    }
    path.write_text(json.dumps({"entity_confidence_summary": summary}, indent=2), encoding="utf-8")
