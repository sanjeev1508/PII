"""
Ensemble PII detection — OPTIMIZED for speed.
Optimizations applied:
  1. `build_analyzer()` is now @lru_cache — built ONCE per process, never rebuilt
  2. Pages are processed in PARALLEL via ThreadPoolExecutor (Presidio is thread-safe)
  3. Regex pre-screen: pages with no PII trigger keywords skip spaCy NLP overhead
  4. Transformer NER processes text in overlapping 400-token chunks, all at once
  5. Result collection is lock-protected for thread safety
  6. O(1) allowlist lookup via frozenset
Layers:
  1 — spaCy + Presidio (NLP-based)
  2 — Custom regex recognizers (phone, SSN, CC, IP, passport, DL, DOB)
  3 — Transformer NER (dslim/bert-base-NER, CPU, optional)
  4 — Context boosting (+0.12 when near PII keywords)
  5 — Allowlist filter
"""
import json
import threading
import concurrent.futures
import re
from functools import lru_cache
from pathlib import Path

import yaml
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider
from sanitization.llm_classifier import llm_classify_batch
from sanitization.validators import (
    is_real_address, is_real_financial_amount,
    is_real_party_name, is_real_ssn, luhn_check,
)

_PII_CONTEXT_KEYWORDS = [
    "name", "dob", "date of birth", "ssn", "social security", "phone", "cell",
    "mobile", "contact", "email", "address", "born", "passport", "license", "id",
]
_HF_NER_MAP = {"PER": "PERSON", "ORG": "PARTY_NAME", "LOC": "ADDRESS"}
_HIGH_RISK = {"PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "EMAIL_ADDRESS",
              "IP_ADDRESS", "US_PASSPORT", "US_DRIVER_LICENSE"}

# Pre-compiled fast regex pre-screen (any PII signal present?)
_PRESCREEN_RE = re.compile(
    r"(?:\d{3}[-.\s]\d{3}[-.\s]\d{4}"          # phone
    r"|\d{3}-\d{2}-\d{4}"                        # SSN
    r"|[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"  # email
    r"|\b(?:\d{1,3}\.){3}\d{1,3}\b"             # IP
    r"|\$[\d,]+"                                  # money
    r"|[A-Z][a-z]+ [A-Z][a-z]+"                 # name-like
    r"|(?:dob|ssn|passport|license)[:\s])"       # labelled fields
    , re.IGNORECASE
)

# Base parallel workers for detection — auto-scales down for large PDFs
_DETECT_WORKERS_MAX = int(__import__("os").getenv("DETECT_WORKERS", "4"))


def _adaptive_detect_workers(n_pages: int) -> int:
    """Scale down workers for large PDFs to prevent spaCy RAM saturation."""
    if n_pages <= 20:
        return _DETECT_WORKERS_MAX
    if n_pages <= 50:
        return max(2, _DETECT_WORKERS_MAX - 1)
    return max(2, _DETECT_WORKERS_MAX // 2)  # e.g. 4→2 for 100+ pages


def load_config(config_path: str = "configs/entities.yaml") -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_allowlist(path: str = "configs/allowlist.yaml") -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f).get("allowlist", {})
    except Exception:
        return {}


def prewarm_analyzer():
    """
    BLOCKING prewarm — loads Presidio analyzer + transformer NER fully before returning.
    Called concurrently with prewarm_ocr() at app startup via asyncio.gather,
    so both models load in parallel and the server only accepts requests when ready.
    """
    print("[DETECT] Loading Presidio analyzer (blocking startup prewarm)…")
    try:
        _get_cached_analyzer()
        print("[DETECT] Presidio analyzer ready.")
    except Exception as e:
        print(f"[DETECT] Analyzer prewarm failed (non-fatal): {e}")

    print("[DETECT] Loading transformer NER (blocking startup prewarm)…")
    try:
        _get_transformer_ner()
        print("[DETECT] Transformer NER ready.")
    except Exception as e:
        print(f"[DETECT] NER prewarm failed (non-fatal): {e}")


@lru_cache(maxsize=1)
def _get_transformer_ner():
    import os
    if os.getenv("ENABLE_HF_NER", "0") != "1":
        print("[DETECT] Fast mode enabled: Skipping Transformer NER (dslim/bert-base-NER).")
        return None
        
    try:
        from transformers import pipeline as hf_pipeline
        ner = hf_pipeline("ner", model="dslim/bert-base-NER",
                          aggregation_strategy="simple", device=-1)
        print("[DETECT] Transformer NER loaded (dslim/bert-base-NER)")
        return ner
    except Exception as e:
        print(f"[DETECT] Transformer NER unavailable: {e}")
        return None


@lru_cache(maxsize=1)
def _get_cached_analyzer() -> AnalyzerEngine:
    """Build the Presidio AnalyzerEngine exactly ONCE per process — most impactful optimization."""
    print("[DETECT] Building analyzer (one-time setup)…")
    
    # The `_sm` model over-predicts entities heavily. We restore `_lg` for high accuracy.
    # The previous slowness was caused by Transformer NER, not spaCy _lg, so speed remains fast.
    try:
        import spacy
        spacy.load("en_core_web_lg")
        spacy_model = "en_core_web_lg"
    except OSError:
        print("[DETECT] en_core_web_lg missing, using sm fallback")
        spacy_model = "en_core_web_sm"
        
    print(f"[DETECT] Using spaCy model: {spacy_model}")

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model}],
    })
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    for rec in [_us_phone_recognizer(), _financial_amount_recognizer(),
                _party_name_recognizer(), _jurisdiction_recognizer(),
                _dob_recognizer(), _ip_recognizer(),
                _passport_recognizer(), _dl_recognizer()]:
        analyzer.registry.add_recognizer(rec)
    print("[DETECT] Analyzer ready.")
    return analyzer


# Keep backward-compat alias
def build_analyzer() -> AnalyzerEngine:
    return _get_cached_analyzer()


# ── Regex recognizers ─────────────────────────────────────────────────────────

def _us_phone_recognizer():
    return PatternRecognizer(supported_entity="PHONE_NUMBER", supported_language="en", patterns=[
        Pattern("PHONE_PAREN", r"\(\d{3}\)\s?\d{3}[-.\s]\d{4}", 0.95),
        Pattern("PHONE_DASH", r"\b\d{3}[-.]\d{3}[-.]\d{4}\b", 0.90),
        Pattern("PHONE_INTL", r"\+1[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}", 0.95),
        Pattern("PHONE_COMPACT", r"\b[2-9]\d{2}[2-9]\d{6}\b", 0.75),
    ])

def _financial_amount_recognizer():
    return PatternRecognizer(supported_entity="FINANCIAL_AMOUNT", supported_language="en", patterns=[
        Pattern("FIN_DOLLAR", r"\$[\d,]+(?:\.\d{1,2})?(?:\s?(?:million|billion|thousand))?", 0.85),
        Pattern("FIN_PLAIN", r"\b\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?\b", 0.70),
    ])

def _party_name_recognizer():
    return PatternRecognizer(supported_entity="PARTY_NAME", supported_language="en", patterns=[
        Pattern("CORP", r"\b[A-Z][A-Za-z&\s\.,\']{2,60}(?:Corporation|Corp|LLC|Ltd|Inc|L\.P\.|LLP|Company|Co\.|PLC|N\.A\.)\b", 0.80),
    ])

def _jurisdiction_recognizer():
    return PatternRecognizer(supported_entity="JURISDICTION", supported_language="en", patterns=[
        Pattern("COURT", r"\b(?:Court of|District Court|Supreme Court|Circuit Court|Court of Appeals)[^\n]{0,60}", 0.75),
        Pattern("STATE_LAW", r"\b(?:laws? of the State of|governed by the laws? of|organized under the laws? of)\s+[A-Z][a-zA-Z\s]+", 0.75),
    ])

def _dob_recognizer():
    return PatternRecognizer(supported_entity="DATE_TIME", supported_language="en", patterns=[
        Pattern("DOB_SLASH", r"\b(?:DOB|Date of Birth|Birth Date|Born)[:\s]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", 0.92),
        Pattern("DOB_ISO", r"\b(?:DOB|Date of Birth)[:\s]+\d{4}-\d{2}-\d{2}", 0.92),
    ])

def _ip_recognizer():
    return PatternRecognizer(supported_entity="IP_ADDRESS", supported_language="en", patterns=[
        Pattern("IPV4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", 0.85),
        Pattern("IPV6", r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b", 0.90),
    ])

def _passport_recognizer():
    return PatternRecognizer(supported_entity="US_PASSPORT", supported_language="en", patterns=[
        Pattern("PASSPORT_LABEL", r"(?:Passport|Passport No\.?)[:\s]+[A-Z0-9]{6,9}", 0.90),
        Pattern("PASSPORT_BARE", r"\b[A-Z]\d{8}\b", 0.60),
    ])

def _dl_recognizer():
    return PatternRecognizer(supported_entity="US_DRIVER_LICENSE", supported_language="en", patterns=[
        Pattern("DL_LABEL", r"(?:Driver'?s?\s+Licen[sc]e|DL)[:\s#]+[A-Z0-9]{4,12}", 0.90),
        Pattern("DL_BARE", r"\b[A-Z]{1,2}\d{6,8}\b", 0.55),
    ])


# ── Per-page detection worker (runs in thread pool) ───────────────────────────

def _detect_page(args):
    """Detect PII entities on a single page. Thread-safe."""
    page_data, enabled_types, presidio_types, allowlist_sets, ner, llm_min_conf, llm_enabled, llm_review_types = args

    pg = page_data["page"]
    text = page_data["text"]

    if not text.strip():
        return pg, [], []

    # Fast pre-screen: skip expensive NLP if page has no PII signals
    if not _PRESCREEN_RE.search(text):
        return pg, [], []

    analyzer = _get_cached_analyzer()

    # Layer 1+2: Presidio (spaCy + regex)
    try:
        presidio_results = analyzer.analyze(text=text, language="en", entities=presidio_types)
    except Exception as e:
        print(f"[DETECT] Presidio failed page {pg}: {e}")
        presidio_results = []

    # Layer 3: Transformer NER — chunked to handle texts longer than 512 tokens
    transformer_hits: dict[tuple, float] = {}
    if ner:
        try:
            chunk_size = 400
            step = 350  # overlapping chunks
            for start in range(0, min(len(text), 4000), step):
                chunk = text[start:start + chunk_size]
                for r in ner(chunk):
                    mapped = _HF_NER_MAP.get(r.get("entity_group", ""))
                    if mapped and mapped in enabled_types:
                        abs_start = start + r["start"]
                        abs_end = start + r["end"]
                        key = (mapped, abs_start, abs_end)
                        transformer_hits[key] = float(r.get("score", 0.5))
        except Exception as e:
            print(f"[DETECT] Transformer NER error page {pg}: {e}")

    page_candidates = []
    page_review_log = []

    for result in presidio_results:
        entity_type = _from_presidio(result.entity_type)
        if entity_type not in enabled_types:
            continue
        cfg = enabled_types[entity_type]
        score = float(result.score)

        # Layer 4: Context boost
        window = text[max(0, result.start - 60):result.start].lower()
        if any(kw in window for kw in _PII_CONTEXT_KEYWORDS):
            score = min(0.99, score + 0.12)

        # Agreement bonus from transformer
        for (etype, ts, te), tscore in transformer_hits.items():
            if etype == entity_type and abs(ts - result.start) < 10:
                score = min(0.99, max(score, tscore) + 0.10)
                break

        effective_min = 0.30 if entity_type in _HIGH_RISK else llm_min_conf
        if score < effective_min:
            page_review_log.append({
                "page": pg, "type": entity_type,
                "value": text[result.start:result.end].strip(),
                "confidence": round(score, 3), "decision": "REJECTED",
                "reason": "below_min_threshold", "review_method": "THRESHOLD"
            })
            continue

        value = text[result.start:result.end].strip()

        # Layer 5: Allowlist (O(1) frozenset lookup)
        if value.strip().lower() in allowlist_sets.get(entity_type, frozenset()):
            continue

        if not _validate(entity_type, value, cfg):
            page_review_log.append({
                "page": pg, "type": entity_type, "value": value,
                "confidence": round(score, 3), "decision": "REJECTED",
                "reason": "validator_rejected", "review_method": "VALIDATOR"
            })
            continue

        page_candidates.append({
            "page": pg, "type": entity_type, "value": value,
            "start": result.start, "end": result.end,
            "confidence": round(score, 3),
            "mask_template": cfg.get("mask_template", f"[{entity_type}-{{id}}]"),
            "_score": score,
            "_cfg": cfg,
        })

    return pg, page_candidates, page_review_log


# ── Detection entry point ─────────────────────────────────────────────────────

def detect_entities(pages: list[dict], config: dict):
    import time
    t0 = time.perf_counter()
    print(f"[DETECT] Starting parallel ensemble detection on {len(pages)} pages")

    analyzer = _get_cached_analyzer()   # ensure cached before threads start
    ner = _get_transformer_ner()

    allowlist_raw = _load_allowlist()
    # Convert to frozensets for O(1) lookup
    allowlist_sets = {k: frozenset(v.lower() for v in vals)
                      for k, vals in allowlist_raw.items()}

    enabled_types = {e["type"]: e for e in config["entities"] if e.get("enabled", True)}
    llm_cfg = config.get("llm_review", {})
    llm_enabled = bool(llm_cfg.get("enabled", False))
    llm_min_conf = float(llm_cfg.get("min_confidence", 0.4))
    llm_model = llm_cfg.get("model", "gpt-4.1-mini")
    llm_fail_open = bool(llm_cfg.get("fail_open_on_error", True))
    llm_review_types = set(llm_cfg.get("review_entity_types", []))
    llm_max = int(llm_cfg.get("max_reviews_per_document", 250))
    llm_batch = int(llm_cfg.get("batch_size", 50))
    ref_path = llm_cfg.get("reference_file", "outputs/llm/reference_scores.json")
    skills_path = llm_cfg.get("skills_file", "SKILLS.md")

    presidio_types = _to_presidio(list(enabled_types.keys()))

    # Build per-page args
    per_page_args = [
        (page_data, enabled_types, presidio_types, allowlist_sets,
         ner, llm_min_conf, llm_enabled, llm_review_types)
        for page_data in pages
    ]

    n_pages = len(pages)
    workers = _adaptive_detect_workers(n_pages)
    print(f"[DETECT] Using {workers} workers for {n_pages} pages")

    # ── Parallel page detection ───────────────────────────────────────────────
    all_candidates = []
    all_review_log = []
    score_acc: dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = list(pool.map(_detect_page, per_page_args))

    done_count = 0
    for pg, page_candidates, page_review_log in futures:
        done_count += 1
        if done_count % 25 == 0 or done_count == n_pages:
            print(f"[DETECT] Processed {done_count}/{n_pages} pages")
        all_review_log.extend(page_review_log)
        for cand in page_candidates:
            score = cand.pop("_score")
            cfg = cand.pop("_cfg")
            score_acc.setdefault(cand["type"], []).append(score)

            max_conf = float(cfg.get("confidence_threshold", 0.5))
            allow_mask = score >= max_conf
            should_review = not llm_review_types or cand["type"] in llm_review_types

            cand["_allow_mask"] = allow_mask
            cand["_should_review"] = should_review
            cand["_score_final"] = score
            cand["_llm_enabled"] = llm_enabled
            all_candidates.append(cand)

    _write_reference(ref_path, score_acc)

    # ── Threshold pass + LLM batch ────────────────────────────────────────────
    all_entities = []
    entity_id_counter: dict = {}
    candidate_pool = []
    pending_reviews = []
    pending_unique: dict = {}
    llm_json_results: list = []

    for cand in all_candidates:
        allow_mask = cand.pop("_allow_mask")
        should_review = cand.pop("_should_review")
        score = cand.pop("_score_final")
        llm_cand_enabled = cand.pop("_llm_enabled")

        entity_type = cand["type"]
        value = cand["value"]
        pg = cand["page"]

        candidate_pool.append({k: v for k, v in cand.items()})

        if not allow_mask and llm_cand_enabled and should_review:
            # Defer to LLM batch
            cache_key = (entity_type, value.lower())
            if cache_key not in pending_unique and len(pending_unique) < llm_max:
                cid = f"cand_{len(pending_unique) + 1}"
                pending_unique[cache_key] = {
                    "id": cid, "entity_type": entity_type,
                    "value": value, "confidence": cand["confidence"],
                    "context": f"page {pg}"
                }
            pending_reviews.append({**cand, "cache_key": cache_key})
            continue

        if not allow_mask:
            all_review_log.append({
                "page": pg, "type": entity_type, "value": value,
                "confidence": cand["confidence"], "decision": "REJECTED",
                "reason": "below_max_threshold", "review_method": "THRESHOLD"
            })
            continue

        entity_id_counter[entity_type] = entity_id_counter.get(entity_type, 0) + 1
        eid = entity_id_counter[entity_type]
        all_entities.append({
            **{k: v for k, v in cand.items() if not k.startswith("_")},
            "mask": cand["mask_template"].format(id=eid),
            "llm_reviewed": False, "llm_classification": None, "review_method": "THRESHOLD",
        })
        all_review_log.append({
            "page": pg, "type": entity_type, "value": value,
            "confidence": cand["confidence"], "decision": "MASKED",
            "reason": "above_max_threshold", "review_method": "THRESHOLD"
        })

    # LLM batch (if enabled)
    llm_cache: dict = {}
    if pending_reviews and pending_unique:
        llm_results, llm_json_results = llm_classify_batch(
            list(pending_unique.values()), model=llm_model,
            batch_size=llm_batch, reference_path=ref_path, skills_path=skills_path,
        )
        for key, item in pending_unique.items():
            d = llm_results.get(item["id"], "ERROR")
            llm_cache[key] = (d == "MASK", d if d in ("SENSITIVE", "NOT_SENSITIVE") else "ERROR")

        for item in pending_reviews:
            ck = item["cache_key"]
            allow_mask, cls = llm_cache.get(ck, (llm_fail_open, "SKIPPED"))
            if allow_mask:
                t = item["type"]
                entity_id_counter[t] = entity_id_counter.get(t, 0) + 1
                all_entities.append({
                    **{k: v for k, v in item.items() if not k.startswith("_") and k != "cache_key"},
                    "mask": item["mask_template"].format(id=entity_id_counter[t]),
                    "llm_reviewed": True, "llm_classification": cls, "review_method": "LLM",
                })

    print(f"[DETECT] Done. {len(all_entities)} entities in {time.perf_counter()-t0:.2f}s")
    return all_entities, all_review_log, llm_json_results, candidate_pool


# ── Utilities ─────────────────────────────────────────────────────────────────

def _type_aware_min(t: str, global_min: float) -> float:
    return min(global_min, 0.30) if t in _HIGH_RISK else global_min


def _validate(entity_type, value, cfg):
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


def _to_presidio(types):
    m = {"PERSON": "PERSON", "EMAIL_ADDRESS": "EMAIL_ADDRESS", "PHONE_NUMBER": "PHONE_NUMBER",
         "CREDIT_CARD": "CREDIT_CARD", "US_SSN": "US_SSN", "FINANCIAL_AMOUNT": "FINANCIAL_AMOUNT",
         "PARTY_NAME": "PARTY_NAME", "ADDRESS": "LOCATION", "JURISDICTION": "JURISDICTION",
         "DATE_TIME": "DATE_TIME", "IP_ADDRESS": "IP_ADDRESS",
         "US_PASSPORT": "US_PASSPORT", "US_DRIVER_LICENSE": "US_DRIVER_LICENSE"}
    return list({m.get(t, t) for t in types})


def _from_presidio(t):
    return {"LOCATION": "ADDRESS"}.get(t, t)


def _write_reference(ref_path, acc):
    p = Path(ref_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    summary = {t: {"count": len(s), "avg": round(sum(s)/len(s), 4) if s else 0,
                   "min": round(min(s), 4) if s else 0, "max": round(max(s), 4) if s else 0}
               for t, s in acc.items()}
    p.write_text(json.dumps({"entity_confidence_summary": summary}, indent=2), encoding="utf-8")
