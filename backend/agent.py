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

# Try to import other model providers
try:
    from pydantic_ai.models.openai import OpenAIModel
except Exception:
    OpenAIModel = None

try:
    from pydantic_ai.models.anthropic import AnthropicModel
except Exception:
    AnthropicModel = None

try:
    from pydantic_ai.models.groq import GroqModel
except Exception:
    GroqModel = None


@dataclass
class AgentDeps:
    engine: Any
    session_id: str


SYSTEM_PROMPT = """
You are an expert marketing mix modeling analyst.
You help users analyze uploaded CSV marketing data, estimate channel effectiveness, optimize budgets, forecast scenarios, and explain adstock, saturation, and marginal ROI.
Always use the available tools instead of inventing metrics.
When a user asks a compound question, call every relevant tool needed to answer it.
Respond in concise business language.
When tools return charts or tables, summarize the key takeaway and keep the result structured for the UI.
""".strip()


class MMMAgentService:
    def __init__(self, engine: Any, llm_config: dict | None = None) -> None:
        self.engine = engine
        self.llm_config = llm_config or {}
        self.agent = self._build_agent()

    def _build_agent(self) -> Agent | None:
        if Agent is None:
            return None

        provider = self.llm_config.get("provider")
        api_key = self.llm_config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        model_name = self.llm_config.get("model")

        if not api_key:
            return None

        model = self._create_model(provider, api_key, model_name)
        if model is None:
            return None

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
            """Fit the transformed weekly ridge MMM model."""
            return ctx.deps.engine.fit_model(ctx.deps.session_id).model_dump()

        @agent.tool
        def get_roas(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
            """Get modeled ROAS by channel."""
            return ctx.deps.engine.get_roas(ctx.deps.session_id).model_dump()

        @agent.tool
        def optimize_budget(ctx: RunContext[AgentDeps], monthly_budget: float) -> dict[str, Any]:
            """Optimize monthly budget with a constrained nonlinear objective."""
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

        @agent.tool
        def analyze_adstock(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
            """Show adstock decay curves and carryover settings per channel."""
            return ctx.deps.engine.analyze_adstock(ctx.deps.session_id).model_dump()

        @agent.tool
        def analyze_saturation(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
            """Show spend saturation curves per channel."""
            return ctx.deps.engine.analyze_saturation(ctx.deps.session_id).model_dump()

        @agent.tool
        def get_marginal_roi(ctx: RunContext[AgentDeps]) -> dict[str, Any]:
            """Show incremental ROI per channel at current spend levels."""
            return ctx.deps.engine.get_marginal_roi(ctx.deps.session_id).model_dump()

        @agent.tool
        def channel_deep_dive(ctx: RunContext[AgentDeps], channel: str) -> dict[str, Any]:
            """Return a detailed breakdown for one channel."""
            return ctx.deps.engine.channel_deep_dive(ctx.deps.session_id, channel).model_dump()

        return agent

    def _create_model(self, provider: str | None, api_key: str, model_name: str | None) -> Any:
        """Create a Pydantic AI model instance from provider config."""
        provider = (provider or "google").lower()

        if provider == "google" and GoogleModel is not None:
            import os as _os
            _os.environ["GOOGLE_API_KEY"] = api_key
            return GoogleModel(model_name or "gemini-2.0-flash")

        if provider == "openai" and OpenAIModel is not None:
            import os as _os
            _os.environ["OPENAI_API_KEY"] = api_key
            return OpenAIModel(model_name or "gpt-4o-mini")

        if provider == "anthropic" and AnthropicModel is not None:
            import os as _os
            _os.environ["ANTHROPIC_API_KEY"] = api_key
            return AnthropicModel(model_name or "claude-sonnet-4-20250514")

        if provider == "groq" and GroqModel is not None:
            import os as _os
            _os.environ["GROQ_API_KEY"] = api_key
            return GroqModel(model_name or "llama-3.3-70b-versatile")

        if provider == "openrouter" and OpenAIModel is not None:
            import os as _os
            _os.environ["OPENROUTER_API_KEY"] = api_key
            return OpenAIModel(
                model_name or "openai/gpt-4o-mini",
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

        # Fallback to Google if available
        if GoogleModel is not None:
            return GoogleModel(model_name or "gemini-2.0-flash")

        return None

    async def run(self, session_id: str, message: str) -> AgentResponse:
        if self.agent is None:
            return self._fallback_response(session_id, message)

        deps = AgentDeps(engine=self.engine, session_id=session_id)
        result = await self.agent.run(message, deps=deps)
        return result.output

    def _fallback_response(self, session_id: str, message: str) -> AgentResponse:
        text = message.lower()
        channels = self._available_channels(session_id)
        requested_channels = self._extract_channels(message, channels)
        intents = self._detect_intents(text, requested_channels)

        tool_results = []
        notes: list[str] = []

        if "analysis" in intents:
            tool_results.append(self.engine.fit_model(session_id))
            tool_results.append(self.engine.get_roas(session_id))
            notes.append("I ran the transformed weekly MMM fit and refreshed ROAS.")

        if "adstock" in intents:
            tool_results.append(self.engine.analyze_adstock(session_id))
            notes.append("I included adstock carryover curves.")

        if "saturation" in intents:
            tool_results.append(self.engine.analyze_saturation(session_id))
            notes.append("I included saturation response curves.")

        if "marginal_roi" in intents:
            tool_results.append(self.engine.get_marginal_roi(session_id))
            notes.append("I calculated incremental ROI at current spend levels.")

        if "deep_dive" in intents:
            if requested_channels:
                for channel in requested_channels[:2]:
                    tool_results.append(self.engine.channel_deep_dive(session_id, channel))
                notes.append(f"I added a channel deep dive for {', '.join(requested_channels[:2])}.")
            else:
                notes.append("I detected a deep-dive request but no channel name matched the uploaded dataset.")

        if "optimize" in intents:
            budget = self._extract_budget(message)
            if budget is None:
                budget = sum(
                    row["spend"] for row in self.engine.get_summary(session_id).get("channel_summary", [])
                ) / max(
                    self.engine.get_dataframe(session_id)["date"].dt.to_period("M").nunique(),
                    1,
                )
            tool_results.append(self.engine.optimize_budget(session_id, budget))
            notes.append(f"I optimized a ${budget:,.0f} monthly plan with nonlinear response constraints.")

        if "forecast" in intents:
            multipliers = self._extract_channel_multipliers(message, channels)
            if multipliers:
                scenario_name = self._scenario_name(multipliers)
                tool_results.append(self.engine.forecast(session_id, scenario_name, multipliers))
                notes.append("I forecasted the requested scenario changes.")
            else:
                notes.append("I detected a forecast request, but I could not infer a channel-level spend change from the message.")

        if not tool_results:
            tool_results.append(self.engine.get_roas(session_id))
            notes.append("I defaulted to a ROAS readout because the request did not map cleanly to a stronger action.")

        return AgentResponse(
            text=" ".join(notes),
            tool_results=tool_results,
            suggested_prompts=self._suggested_prompts(requested_channels),
        )

    def _available_channels(self, session_id: str) -> list[str]:
        summary = self.engine.get_summary(session_id)
        return [str(row["channel"]) for row in summary.get("channel_summary", [])]

    def _extract_channels(self, message: str, channels: list[str]) -> list[str]:
        found = []
        lowered = message.lower()
        for channel in channels:
            if re.search(rf"\b{re.escape(channel.lower())}\b", lowered):
                found.append(channel)
        return found

    def _detect_intents(self, text: str, requested_channels: list[str]) -> list[str]:
        intents: list[str] = []
        patterns = {
            "analysis": r"\b(analy[sz]e|analysis|performance|roas|contribution|which channel|best|worst)\b",
            "optimize": r"\b(optimi[sz]e|budget|allocate|reallocate|best split)\b",
            "forecast": r"\b(what if|scenario|forecast|simulate|double|increase|cut|decrease|reduce)\b",
            "adstock": r"\b(adstock|carryover|decay)\b",
            "saturation": r"\b(saturation|diminishing returns?|response curve)\b",
            "marginal_roi": r"\b(marginal roi|incremental roi|next dollar|incremental return)\b",
            "deep_dive": r"\b(tell me about|deep dive|breakdown|details on|more about)\b",
        }
        for intent, pattern in patterns.items():
            if re.search(pattern, text):
                intents.append(intent)

        if requested_channels and "deep_dive" not in intents:
            if re.search(r"\b(tell me about|details|breakdown|deep dive|more about)\b", text):
                intents.append("deep_dive")

        if not intents:
            intents.append("analysis")

        ordered = ["analysis", "adstock", "saturation", "marginal_roi", "deep_dive", "optimize", "forecast"]
        return [intent for intent in ordered if intent in intents]

    def _extract_budget(self, message: str) -> float | None:
        currency_match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)\s*([kKmM]?)", message)
        if currency_match:
            match = currency_match
        else:
            match = re.search(
                r"\b(?:budget|allocate|allocation|spend)\b[^\d$]*([\d,]+(?:\.\d+)?)\s*([kKmM]?)\b",
                message,
                flags=re.IGNORECASE,
            )
        if not match:
            return None
        raw_value = float(match.group(1).replace(",", ""))
        suffix = match.group(2).lower()
        if suffix == "k":
            raw_value *= 1000
        elif suffix == "m":
            raw_value *= 1_000_000
        return raw_value

    def _extract_channel_multipliers(self, message: str, channels: list[str]) -> dict[str, float]:
        text = message.lower()
        multipliers: dict[str, float] = {}

        for channel in channels:
            channel_pattern = re.escape(channel.lower())

            if re.search(rf"\bdouble\s+{channel_pattern}\b", text) or re.search(
                rf"\b{channel_pattern}\b.*\bdouble\b",
                text,
            ):
                multipliers[channel] = 2.0
                continue

            if re.search(rf"\bhalve\s+{channel_pattern}\b", text) or re.search(
                rf"\b{channel_pattern}\b.*\bhalve\b",
                text,
            ):
                multipliers[channel] = 0.5
                continue

            increase_patterns = [
                rf"\b(?:increase|raise|boost)\s+{channel_pattern}(?:\s+spend)?\s+by\s+(\d+(?:\.\d+)?)%",
                rf"\b{channel_pattern}\b.*\b(?:increase|raise|boost)\b.*\b(\d+(?:\.\d+)?)%",
            ]
            decrease_patterns = [
                rf"\b(?:cut|decrease|reduce|lower)\s+{channel_pattern}(?:\s+spend)?\s+by\s+(\d+(?:\.\d+)?)%",
                rf"\b{channel_pattern}\b.*\b(?:cut|decrease|reduce|lower)\b.*\b(\d+(?:\.\d+)?)%",
            ]

            for pattern in increase_patterns:
                match = re.search(pattern, text)
                if match:
                    multipliers[channel] = 1.0 + float(match.group(1)) / 100.0
                    break

            if channel in multipliers:
                continue

            for pattern in decrease_patterns:
                match = re.search(pattern, text)
                if match:
                    multipliers[channel] = max(0.0, 1.0 - float(match.group(1)) / 100.0)
                    break

        return multipliers

    def _scenario_name(self, multipliers: dict[str, float]) -> str:
        if len(multipliers) == 1:
            channel, multiplier = next(iter(multipliers.items()))
            if abs(multiplier - 2.0) < 1e-9:
                return f"Double {channel}"
            if multiplier > 1.0:
                return f"Increase {channel}"
            if multiplier < 1.0:
                return f"Cut {channel}"
        return "Custom Scenario"

    def _suggested_prompts(self, requested_channels: list[str]) -> list[str]:
        prompts = [
            "Optimize my budget for $100K/month",
            "What if I double TikTok spend?",
            "Show marginal ROI by channel",
        ]
        if requested_channels:
            prompts.insert(0, f"Give me a deep dive on {requested_channels[0]}")
        else:
            prompts.insert(0, "Show adstock and saturation curves")
        return prompts[:4]


def serialize_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"
