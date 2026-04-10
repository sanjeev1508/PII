"""Abstract base class for all pipeline agents."""
import asyncio
import time
from abc import ABC, abstractmethod

from sanitization.pipeline_context import PipelineContext


class BaseAgent(ABC):
    name: str = "BaseAgent"
    max_retries: int = 1

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        start = time.time()
        await ctx.events.put({"type": "agent_start", "agent": self.name})
        last_err = None

        for attempt in range(self.max_retries + 1):
            try:
                ctx = await self._execute(ctx)
                elapsed = round(time.time() - start, 2)
                ctx.timings[self.name] = elapsed
                await ctx.events.put({
                    "type": "agent_done",
                    "agent": self.name,
                    "elapsed": elapsed,
                })
                return ctx
            except Exception as e:
                last_err = e
                print(f"[{self.name}] Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)

        ctx.errors.append(f"{self.name}: {last_err}")
        await ctx.events.put({
            "type": "agent_error",
            "agent": self.name,
            "error": str(last_err),
        })
        return ctx

    @abstractmethod
    async def _execute(self, ctx: PipelineContext) -> PipelineContext: ...
