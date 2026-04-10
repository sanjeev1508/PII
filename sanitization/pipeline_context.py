"""Shared state dataclass passed through the agent pipeline."""
import asyncio
from dataclasses import dataclass, field


@dataclass
class PipelineContext:
    session_id: str
    input_path: str
    source_file: str
    config: dict
    work_dir: str

    # Filled by agents
    pages: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    image_texts: list = field(default_factory=list)
    raw_candidates: list = field(default_factory=list)
    resolved_entities: list = field(default_factory=list)
    final_entities: list = field(default_factory=list)
    masked_path: str = ""
    report_path: str = ""

    # Metadata
    total_pages: int = 0
    extraction_mode_summary: dict = field(default_factory=dict)
    review_log: list = field(default_factory=list)
    llm_json_results: list = field(default_factory=list)
    agent_review_rows: list = field(default_factory=list)
    redaction_manifest: dict = field(default_factory=dict)
    user_annotations: list = field(default_factory=list)

    # Agentic infra
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    timings: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
