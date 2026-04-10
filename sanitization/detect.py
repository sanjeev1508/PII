import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from .schemas import EntityDetection, WordBox

PII_PATTERNS: Dict[str, str] = {
    "EMAIL": r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE": r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PENALTY_AMOUNT": r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
}

ADDRESS_HINT_RE = re.compile(
    r"\b(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|suite|apt|floor)\b", re.IGNORECASE
)
JURISDICTION_RE = re.compile(
    r"\b(?:governed by|laws of|jurisdiction of|state of|court of|venue)\b", re.IGNORECASE
)
PARTY_CUE_RE = re.compile(
    r"\b(?:this nda is between|between|disclosing party|receiving party|party a|party b)\b", re.IGNORECASE
)
NLP_TRIGGER_RE = re.compile(r"\b(?:agreement|nda|party|between|governed|jurisdiction|address|located)\b", re.IGNORECASE)


def _label_for_output(entity_type: str) -> str:
    mapping = {
        "PHONE": "number",
        "CREDIT_CARD": "card",
        "EMAIL": "email",
        "SSN": "ssn",
        "ADDRESS": "address",
        "PARTY_NAME": "party",
        "JURISDICTION": "jurisdiction",
        "PENALTY_AMOUNT": "penalty",
    }
    return mapping.get(entity_type, entity_type.lower())


def _placeholder_for_type(entity_type: str) -> str:
    mapping = {
        "PHONE": "<mobile_num>",
        "CREDIT_CARD": "<card>",
        "EMAIL": "<email>",
        "SSN": "<ssn>",
        "ADDRESS": "<address>",
        "PARTY_NAME": "<party_name>",
        "JURISDICTION": "<jurisdiction>",
        "PENALTY_AMOUNT": "<amount>",
    }
    return mapping.get(entity_type, "<value>")


def _line_text_and_spans(line_words: List[WordBox]) -> Tuple[str, List[Tuple[int, int, int]]]:
    line_words.sort(key=lambda x: x.word_no)
    parts: List[str] = []
    spans: List[Tuple[int, int, int]] = []
    cursor = 0
    for i, w in enumerate(line_words):
        if i > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(w.raw_text)
        cursor += len(w.raw_text)
        spans.append((start, cursor, i))
    return "".join(parts), spans


def _boxes_for_span(line_words: List[WordBox], spans: List[Tuple[int, int, int]], start: int, end: int):
    idxs = [i for s, e, i in spans if s < end and e > start]
    if not idxs:
        return None
    selected = [line_words[i] for i in idxs]
    x1 = min(w.bbox[0] for w in selected)
    y1 = min(w.bbox[1] for w in selected)
    x2 = max(w.bbox[2] for w in selected)
    y2 = max(w.bbox[3] for w in selected)
    return (x1, y1, x2, y2)


def _try_spacy_model():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _legal_heuristics(line_text: str, line_words: List[WordBox], spans: List[Tuple[int, int, int]], page_idx: int):
    out: List[EntityDetection] = []
    if JURISDICTION_RE.search(line_text):
        box = _boxes_for_span(line_words, spans, 0, len(line_text))
        if box:
            out.append(
                EntityDetection(
                    entity_type="JURISDICTION",
                    value=line_text.strip(),
                    page=page_idx,
                    bbox=box,
                    detector="legal_rule",
                    confidence=0.7,
                )
            )
    if PARTY_CUE_RE.search(line_text):
        box = _boxes_for_span(line_words, spans, 0, len(line_text))
        if box:
            out.append(
                EntityDetection(
                    entity_type="PARTY_NAME",
                    value=line_text.strip(),
                    page=page_idx,
                    bbox=box,
                    detector="legal_rule",
                    confidence=0.65,
                )
            )
    if ADDRESS_HINT_RE.search(line_text):
        box = _boxes_for_span(line_words, spans, 0, len(line_text))
        if box:
            out.append(
                EntityDetection(
                    entity_type="ADDRESS",
                    value=line_text.strip(),
                    page=page_idx,
                    bbox=box,
                    detector="address_rule",
                    confidence=0.6,
                )
            )
    return out


def detect_entities(words_by_page: Dict[int, List[WordBox]], entity_types: Optional[Iterable[str]] = None) -> List[EntityDetection]:
    selected = set(entity_types or PII_PATTERNS.keys())
    patterns = {k: re.compile(v, re.IGNORECASE) for k, v in PII_PATTERNS.items() if k in selected}
    nlp = _try_spacy_model() if any(t in selected for t in ("PARTY_NAME", "ADDRESS")) else None
    detections: List[EntityDetection] = []

    for page_idx, page_words in words_by_page.items():
        lines = defaultdict(list)
        for w in page_words:
            lines[(w.block, w.line)].append(w)

        for _, line_words in lines.items():
            line_text, spans = _line_text_and_spans(line_words)
            if not line_text.strip():
                continue

            for e_type, pat in patterns.items():
                for m in pat.finditer(line_text):
                    box = _boxes_for_span(line_words, spans, *m.span())
                    if not box:
                        continue
                    detections.append(
                        EntityDetection(
                            entity_type=e_type,
                            value=m.group(0).strip(),
                            page=page_idx,
                            bbox=box,
                            confidence=0.95 if e_type in ("EMAIL", "PHONE", "SSN") else 0.9,
                            detector="regex",
                        )
                    )

            if any(t in selected for t in ("JURISDICTION", "PARTY_NAME", "ADDRESS")):
                detections.extend(_legal_heuristics(line_text, line_words, spans, page_idx))

            if nlp and any(t in selected for t in ("PARTY_NAME", "ADDRESS")) and NLP_TRIGGER_RE.search(line_text):
                doc = nlp(line_text)
                for ent in doc.ents:
                    mapped = None
                    if ent.label_ in ("ORG", "PERSON") and "PARTY_NAME" in selected:
                        mapped = "PARTY_NAME"
                    elif ent.label_ in ("GPE", "LOC", "FAC") and "ADDRESS" in selected:
                        mapped = "ADDRESS"
                    if not mapped:
                        continue
                    box = _boxes_for_span(line_words, spans, ent.start_char, ent.end_char)
                    if not box:
                        continue
                    detections.append(
                        EntityDetection(
                            entity_type=mapped,
                            value=ent.text.strip(),
                            page=page_idx,
                            bbox=box,
                            confidence=0.55,
                            detector=f"spacy:{ent.label_}",
                        )
                    )

    unique: List[EntityDetection] = []
    seen = set()
    for d in detections:
        key = (d.page, d.entity_type, d.value.lower(), tuple(round(v, 2) for v in d.bbox))
        if key in seen:
            continue
        seen.add(key)
        d.formatted = f"{_label_for_output(d.entity_type)}: {_placeholder_for_type(d.entity_type)}"
        unique.append(d)
    return unique


def weak_supervision_labels(detections: List[EntityDetection], min_confidence: float = 0.9) -> List[dict]:
    # Silver labels from high-confidence detections.
    return [
        {
            "entity_type": d.entity_type,
            "value": d.value,
            "page": d.page,
            "bbox": d.bbox,
            "confidence": d.confidence,
            "detector": d.detector,
        }
        for d in detections
        if d.confidence >= min_confidence
    ]

