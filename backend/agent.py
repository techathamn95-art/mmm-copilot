from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from schemas import AgentResponse

try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.google import GoogleModel
except Exception:  # pragma: no cover
    Agent = None
    RunContext = Any
    GoogleModel = None


@dataclass
class AgentDeps:
    engine: Any
    session_id: str


SYSTEM_PROMPT = """
You are an expert marketing mix modeling analyst.
You help users analyze uploaded CSV marketing data, estimate channel effectiveness, optimize budgets, and forecast scenarios.
Always use the available tools instead of inventing metrics.
Respond in concise business language.
When tools return charts or tables, summarize the key takeaway and keep the result structured for the UI.
""".strip()


class MMMAgentService:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.agent = self._build_agent()

    def _build_agent(self) -> Agent | None:
        if Agent is None or GoogleModel is None or not os.getenv("GOOGLE_API_KEY"):
            return None

        model = GoogleModel("gemini-2.0-flash")
        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            output_type=AgentResponse,
            system_prompt=SYSTEM_PROMPT,
        )

        @agent.tool
        def load_data(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
            """Load the uploaded MMM dataset summary and preview."""
            return {
                "summary": ctx.deps.engine.get_summary(ctx.deps.session_id),
                "preview": ctx.deps.engine.get_preview(ctx.deps.session_id),
            }

        @agent.tool
        def fit_model(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
            """Fit the baseline ridge MMM model."""
            return ctx.deps.engine.fit_model(ctx.deps.session_id).model_dump()

        @agent.tool
        def get_roas(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
            """Get modeled ROAS by channel."""
            return ctx.deps.engine.get_roas(ctx.deps.session_id).model_dump()

        @agent.tool
        def optimize_budget(ctx: RunContext[AgentDeps], monthly_budget: float) -> dict[str, Any]:
            """Optimize monthly budget based on modeled channel ROAS."""
            return ctx.deps.engine.optimize_budget(ctx.deps.session_id, monthly_budget).model_dump()

        @agent.tool
        def forecast(
            ctx: RunContext[AgentDeps],
            scenario_name: str,
            channel_multipliers: dict[str, float],
        ) -> dict[str, Any]:
            """Forecast a scenario by multiplying current spend for one or more channels."""
            return ctx.deps.engine.forecast(ctx.deps.session_id, scenario_name, channel_multipliers).model_dump()

        @agent.tool
        def compare_scenarios(
            ctx: RunContext[AgentDeps],
            scenarios: dict[str, dict[str, float]],
        ) -> dict[str, Any]:
            """Compare explicit scenario budget maps across channels."""
            return ctx.deps.engine.compare_scenarios(ctx.deps.session_id, scenarios).model_dump()

        return agent

    async def run(self, session_id: str, message: str) -> AgentResponse:
        if self.agent is None:
            return self._fallback_response(session_id, message)

        deps = AgentDeps(engine=self.engine, session_id=session_id)
        result = await self.agent.run(message, deps=deps)
        return result.output

    def _fallback_response(self, session_id: str, message: str) -> AgentResponse:
        text = message.lower()
        tool_results = []

        if any(term in text for term in ["analyze", "analysis", "roas", "performance", "marketing spend"]):
            tool_results.append(self.engine.fit_model(session_id))
            return AgentResponse(
                text="I fitted the baseline MMM and summarized channel efficiency. Google and Facebook typically dominate efficient spend in this dataset, while TV adds scale with lower ROAS.",
                tool_results=tool_results,
                suggested_prompts=[
                    "Optimize my budget for $100K/month",
                    "What if I double TikTok spend?",
                ],
            )

        if "optimiz" in text or "budget" in text:
            budget = self._extract_budget(message) or 100000.0
            tool_results.append(self.engine.optimize_budget(session_id, budget))
            return AgentResponse(
                text=f"I optimized a ${budget:,.0f} monthly budget using modeled ROAS and simple share constraints. The recommendation shifts spend toward the highest-efficiency channels while preserving diversification.",
                tool_results=tool_results,
                suggested_prompts=[
                    "Compare current plan versus this recommendation",
                    "What if I increase TV by 20% instead?",
                ],
            )

        if "double tiktok" in text:
            tool_results.append(
                self.engine.forecast(session_id, "Double TikTok", {"TikTok": 2.0})
            )
            return AgentResponse(
                text="I forecasted a scenario where TikTok spend doubles from the current baseline. Review the scenario chart to see whether the added spend improves expected revenue enough relative to the current plan.",
                tool_results=tool_results,
                suggested_prompts=["Optimize my budget for $100K/month"],
            )

        tool_results.append(self.engine.get_roas(session_id))
        return AgentResponse(
            text="I can analyze channel ROAS, optimize budget allocation, and compare forecast scenarios. Start with a spend analysis or ask for a budget target.",
            tool_results=tool_results,
            suggested_prompts=[
                "Analyze my marketing spend",
                "Optimize my budget for $100K/month",
                "What if I double TikTok spend?",
            ],
        )

    def _extract_budget(self, message: str) -> float | None:
        match = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)\s*([kKmM]?)", message)
        if not match:
            return None
        raw_value = float(match.group(1).replace(",", ""))
        suffix = match.group(2).lower()
        if suffix == "k":
            raw_value *= 1000
        elif suffix == "m":
            raw_value *= 1_000_000
        return raw_value


def serialize_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"
