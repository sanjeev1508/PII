from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


BBox = Tuple[float, float, float, float]


@dataclass
class WordBox:
    raw_text: str
    text: str
    bbox: BBox
    block: int
    line: int
    word_no: int


@dataclass
class EntityDetection:
    entity_type: str
    value: str
    page: int
    bbox: BBox
    confidence: float = 1.0
    detector: str = "rule"
    formatted: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FileSummary:
    doc_id: str
    source_file: str
    output_file: str
    pages: int
    extraction_mode: str
    processing_seconds: float
    entity_counts: Dict[str, int] = field(default_factory=dict)
    entities: List[EntityDetection] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["entities"] = [e.to_dict() for e in self.entities]
        return payload

