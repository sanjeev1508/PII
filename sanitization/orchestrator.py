"""
Agentic orchestrator — chains 6 agents with SSE event streaming.
OPTIMIZED: fixed syntax bug, added per-agent timing, pipeline ready
for future producer-consumer streaming between agents.
"""
import asyncio
import time

from sanitization.pipeline_context import PipelineContext
from sanitization.agents.extraction_agent import ExtractionAgent
from sanitization.agents.detection_agent import DetectionAgent
from sanitization.agents.resolution_agent import ResolutionAgent
from sanitization.agents.review_agent import ReviewAgent
from sanitization.agents.masking_agent import MaskingAgent
from sanitization.agents.reporting_agent import ReportingAgent

# Registry: session_id -> asyncio.Queue (for SSE)
_QUEUES: dict[str, asyncio.Queue] = {}


def get_session_queue(session_id: str) -> asyncio.Queue | None:
    return _QUEUES.get(session_id)


def create_session_queue(session_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _QUEUES[session_id] = q
    return q


async def run_pipeline_async(
    session_id: str,
    input_path: str,
    source_file: str,
    config: dict,
    work_dir: str,
    user_annotations: list | None = None,
    event_queue: asyncio.Queue | None = None,
) -> PipelineContext:

    q = event_queue or create_session_queue(session_id)

    ctx = PipelineContext(
        session_id=session_id,
        input_path=input_path,
        source_file=source_file,
        config=config,
        work_dir=work_dir,
        events=q,
        user_annotations=user_annotations or [],
    )

    agents = [
        ExtractionAgent(),
        DetectionAgent(),
        ResolutionAgent(),
        ReviewAgent(),
        MaskingAgent(),
        ReportingAgent(),
    ]

    await q.put({
        "type": "pipeline_start",
        "total_agents": len(agents),
        "agents": [a.name for a in agents],
    })

    pipeline_start = time.perf_counter()

    for agent in agents:
        agent_start = time.perf_counter()
        ctx = await agent.run(ctx)
        elapsed = round(time.perf_counter() - agent_start, 2)

        # Emit per-agent timing to SSE stream
        await q.put({
            "type": "agent_timing",
            "agent": agent.name,
            "elapsed_sec": elapsed,
        })

        if agent.name == "ExtractionAgent" and ctx.errors:
            break  # Non-recoverable extraction failure

    total_elapsed = round(time.perf_counter() - pipeline_start, 2)

    await q.put({
        "type": "pipeline_complete",
        "masked_url": f"/pdf/{session_id}/masked",
        "report_url": f"/pdf/{session_id}/report",
        "original_url": f"/pdf/{session_id}/original",
        "entity_count": len(ctx.final_entities),
        "timings": ctx.timings,
        "total_elapsed_sec": total_elapsed,
        "errors": ctx.errors,
    })
    await q.put(None)  # SSE sentinel — closes stream

    return ctx