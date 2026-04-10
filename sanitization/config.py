import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "pipeline": {
        "mode": "redact",  # redact|highlight
        "workers": 4,
        "timeout_seconds": 180,
        "retry_count": 1,
    },
    "extraction": {
        "ocr_scale": 1,
        "native_text_min_words": 30,
        "use_ocr_fallback": True,
    },
    "detection": {
        "entity_types": [
            "EMAIL",
            "PHONE",
            "SSN",
            "CREDIT_CARD",
            "ADDRESS",
            "PARTY_NAME",
            "JURISDICTION",
            "PENALTY_AMOUNT",
        ]
    },
    "output": {
        "pdf_dir": "outputs/pdfs",
        "report_dir": "outputs/reports",
        "log_dir": "outputs/logs",
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(config_path: str = "configs/pipeline.yaml") -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG
    path = Path(config_path)
    if not path.exists():
        return cfg

    data: Dict[str, Any] = {}
    if path.suffix.lower() in {".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            # Keep runtime robust if PyYAML is unavailable.
            data = {}

    if not isinstance(data, dict):
        data = {}
    return _deep_merge(cfg, data)

