"""Review agent — Qdrant vector lookup + LLM batch judgement."""
import asyncio
from sanitization.agents.base_agent import BaseAgent
from sanitization.pipeline_context import PipelineContext
from sanitization.qdrant_agent import process_candidates_above_threshold


class ReviewAgent(BaseAgent):
    name = "ReviewAgent"
    max_retries = 1

    async def _execute(self, ctx: PipelineContext) -> PipelineContext:
        await ctx.events.put({"type": "progress", "agent": self.name, "msg": "LLM + Qdrant review…"})
        result = await asyncio.to_thread(
            process_candidates_above_threshold,
            ctx.resolved_entities,
            ctx.source_file,
            ctx.config,
        )
        ctx.final_entities = result.get("entities_to_mask", [])
        ctx.agent_review_rows = result.get("review_rows", [])
        await ctx.events.put({
            "type": "progress",
            "agent": self.name,
            "msg": f"{len(ctx.final_entities)} entities approved for masking",
        })
        return ctx
