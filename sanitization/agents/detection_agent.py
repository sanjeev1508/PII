"""Detection agent — parallel Presidio+NER ensemble with context boosting."""
import asyncio
from sanitization.agents.base_agent import BaseAgent
from sanitization.pipeline_context import PipelineContext
from sanitization.detector import detect_entities, _get_cached_analyzer, _get_transformer_ner


class DetectionAgent(BaseAgent):
    name = "DetectionAgent"
    max_retries = 1

    async def _execute(self, ctx: PipelineContext) -> PipelineContext:
        # Pre-warm analyzer + NER in the current event loop before spawning threads
        # (lru_cache means this is a no-op after first call)
        await asyncio.to_thread(_get_cached_analyzer)
        await asyncio.to_thread(_get_transformer_ner)

        await ctx.events.put({
            "type": "progress", "agent": self.name,
            "msg": f"Running parallel PII detection on {len(ctx.pages)} pages…"
        })

        _, review_log, llm_json, candidate_pool = await asyncio.to_thread(
            detect_entities, ctx.pages, ctx.config
        )
        ctx.raw_candidates = candidate_pool
        ctx.review_log = review_log
        ctx.llm_json_results = llm_json

        await ctx.events.put({
            "type": "progress",
            "agent": self.name,
            "msg": f"{len(candidate_pool)} candidates detected",
        })
        return ctx
