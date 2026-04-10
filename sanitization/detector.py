"""
Entity detection pipeline using Microsoft Presidio + spaCy + custom regex patterns.
Applies validators to reduce false positives.
"""
from pathlib import Path

import yaml
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from sanitization.llm_classifier import llm_is_sensitive

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


def detect_entities(pages: list[dict], config: dict) -> tuple[list[dict], list[dict]]:
    """
    Run detection on all extracted pages.
    Returns list of detected entity dicts with page, type, value, start, end, confidence.
    """
    print("[DETECT] Starting entity detection")
    try:
        analyzer = build_analyzer()
    except Exception as e:
        print(f"[DETECT] Analyzer init failed: {e}")
        return [], []

    enabled_types = {e["type"]: e for e in config["entities"] if e.get("enabled", True)}
    llm_cfg = config.get("llm_review", {})
    llm_enabled = bool(llm_cfg.get("enabled", False))
    llm_min_conf = float(llm_cfg.get("min_confidence", 0.5))
    llm_model = llm_cfg.get("model", "gpt-4.1-mini")
    llm_fail_open = bool(llm_cfg.get("fail_open_on_error", True))
    all_entities = []
    review_log = []
    entity_id_counter = {}

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

            allow_mask = score >= max_conf
            llm_reviewed = False
            llm_classification = None
            if not allow_mask and llm_enabled and score < max_conf:
                context_start = max(0, result.start - 80)
                context_end = min(len(text), result.end + 80)
                context = text[context_start:context_end].replace("\n", " ").strip()
                llm_reviewed = True
                try:
                    allow_mask = llm_is_sensitive(
                        entity_type=entity_type,
                        value=value,
                        confidence=score,
                        context=context,
                        model=llm_model,
                    )
                    llm_classification = "SENSITIVE" if allow_mask else "NOT_SENSITIVE"
                except Exception as e:
                    print(f"[DETECT] LLM review failed ({entity_type}): {e}")
                    allow_mask = llm_fail_open
                    llm_classification = "ERROR"
            if not allow_mask:
                review_log.append(
                    {
                        "page": page_num,
                        "type": entity_type,
                        "value": value,
                        "confidence": round(score, 3),
                        "decision": "REJECTED",
                        "reason": (
                            "llm_not_sensitive"
                            if llm_reviewed and llm_classification != "ERROR"
                            else "llm_error_rejected"
                            if llm_reviewed
                            else "below_max_threshold"
                        ),
                        "review_method": "LLM" if llm_reviewed else "THRESHOLD",
                        "llm_classification": llm_classification,
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
                    "llm_reviewed": llm_reviewed,
                    "llm_classification": llm_classification,
                    "review_method": "LLM" if llm_reviewed else "THRESHOLD",
                }
            )
            review_log.append(
                {
                    "page": page_num,
                    "type": entity_type,
                    "value": value,
                    "confidence": round(score, 3),
                    "decision": "MASKED",
                    "reason": (
                        "above_max_threshold"
                        if not llm_reviewed
                        else "llm_error_masked"
                        if llm_classification == "ERROR"
                        else "llm_sensitive"
                    ),
                    "review_method": "LLM" if llm_reviewed else "THRESHOLD",
                    "llm_classification": llm_classification,
                }
            )

    print(f"[DETECT] Detection complete. Found {len(all_entities)} entities")
    return all_entities, review_log


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
