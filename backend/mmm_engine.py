from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from schemas import ChartDataPoint, ChartPayload, ChartSeries, ColumnMapping, ToolResult


CHANNEL_COLORS = {
    "Facebook": "#3b82f6",
    "Google": "#22c55e",
    "TikTok": "#f97316",
    "TV": "#eab308",
}


@dataclass
class MMMArtifacts:
    model: Ridge
    feature_columns: list[str]
    coefficients: dict[str, float]
    intercept: float
    roas: dict[str, float]
    contributions: dict[str, float]
    monthly_summary: list[dict[str, Any]]
    r2: float


class MMMEngine:
    def __init__(self) -> None:
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.mappings: dict[str, ColumnMapping] = {}
        self.models: dict[str, MMMArtifacts] = {}

    def parse_csv(
        self,
        session_id: str,
        file_bytes: bytes,
        file_name: str,
        mapping: ColumnMapping | None = None,
    ) -> dict[str, Any]:
        chosen_mapping = mapping or ColumnMapping()
        frame = pd.read_csv(pd.io.common.BytesIO(file_bytes))
        normalized = self._normalize_columns(frame, chosen_mapping)
        self._validate_dataframe(normalized)
        normalized["date"] = pd.to_datetime(normalized["date"])
        normalized["channel"] = normalized["channel"].astype(str)
        normalized["spend"] = normalized["spend"].astype(float)
        normalized["revenue"] = normalized["revenue"].astype(float)
        normalized = normalized.sort_values("date").reset_index(drop=True)

        self.dataframes[session_id] = normalized
        self.mappings[session_id] = chosen_mapping

        return {
            "file_name": file_name,
            "columns": frame.columns.tolist(),
            "mapping": chosen_mapping.model_dump(),
            "summary": self.get_summary(session_id),
            "preview": self.get_preview(session_id),
        }

    def get_dataframe(self, session_id: str) -> pd.DataFrame:
        if session_id not in self.dataframes:
            raise ValueError("No dataset loaded for this session.")
        return self.dataframes[session_id]

    def get_summary(self, session_id: str) -> dict[str, Any]:
        df = self.get_dataframe(session_id)
        channel_summary = (
            df.groupby("channel", as_index=False)
            .agg(
                spend=("spend", "sum"),
                revenue=("revenue", "sum"),
                avg_daily_spend=("spend", "mean"),
            )
            .sort_values("spend", ascending=False)
        )
        return {
            "rows": int(len(df)),
            "date_range": {
                "start": df["date"].min().strftime("%Y-%m-%d"),
                "end": df["date"].max().strftime("%Y-%m-%d"),
            },
            "channels": df["channel"].nunique(),
            "total_spend": round(float(df["spend"].sum()), 2),
            "total_revenue": round(float(df["revenue"].sum()), 2),
            "overall_roas": round(float(df["revenue"].sum() / max(df["spend"].sum(), 1.0)), 4),
            "channel_summary": channel_summary.round(2).to_dict(orient="records"),
        }

    def get_preview(self, session_id: str, limit: int = 12) -> list[dict[str, Any]]:
        df = self.get_dataframe(session_id).copy()
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df.head(limit).round(2).to_dict(orient="records")

    def fit_model(self, session_id: str) -> ToolResult:
        df = self.get_dataframe(session_id)
        feature_frame = (
            df.assign(month=df["date"].dt.to_period("M").astype(str))
            .pivot_table(index="month", columns="channel", values="spend", aggfunc="sum", fill_value=0.0)
            .sort_index()
        )
        revenue_target = (
            df.assign(month=df["date"].dt.to_period("M").astype(str))
            .groupby("month")["revenue"]
            .sum()
            .reindex(feature_frame.index)
        )

        model = Ridge(alpha=1.0)
        model.fit(feature_frame, revenue_target)
        predictions = model.predict(feature_frame)
        r2 = float(model.score(feature_frame, revenue_target))
        coefficients = {col: float(coef) for col, coef in zip(feature_frame.columns.tolist(), model.coef_)}
        raw_contributions = {
            channel: float(feature_frame[channel].sum() * coefficients.get(channel, 0.0))
            for channel in feature_frame.columns
        }
        total_contribution = sum(max(value, 0.0) for value in raw_contributions.values()) or 1.0
        contributions = {
            channel: max(value, 0.0) / total_contribution for channel, value in raw_contributions.items()
        }

        spend_by_channel = df.groupby("channel")["spend"].sum().to_dict()
        roas = {
            channel: round(
                (max(raw_contributions.get(channel, 0.0), 0.0) + 1e-6) / max(spend_by_channel.get(channel, 1.0), 1.0),
                4,
            )
            for channel in spend_by_channel
        }
        monthly_summary = []
        for month, actual, predicted in zip(feature_frame.index.tolist(), revenue_target.tolist(), predictions.tolist()):
            monthly_summary.append(
                {
                    "month": month,
                    "actual_revenue": round(float(actual), 2),
                    "predicted_revenue": round(float(predicted), 2),
                }
            )

        self.models[session_id] = MMMArtifacts(
            model=model,
            feature_columns=feature_frame.columns.tolist(),
            coefficients=coefficients,
            intercept=float(model.intercept_),
            roas=roas,
            contributions=contributions,
            monthly_summary=monthly_summary,
            r2=r2,
        )

        return ToolResult(
            tool="fit_model",
            title="Marketing mix model fitted",
            summary="A ridge-regression baseline was trained on monthly spend by channel to estimate channel contribution and efficiency.",
            metrics={
                "r2": round(r2, 4),
                "months_modeled": len(feature_frame.index),
                "predicted_revenue": round(float(sum(item["predicted_revenue"] for item in monthly_summary)), 2),
            },
            tables={"monthly_performance": monthly_summary},
            charts=[
                self._build_roas_chart(roas),
                self._build_contribution_chart(contributions, df),
                self._build_efficiency_scatter(df, roas),
            ],
        )

    def get_roas(self, session_id: str) -> ToolResult:
        artifacts = self._ensure_model(session_id)
        return ToolResult(
            tool="get_roas",
            title="Channel ROAS",
            summary="ROAS is estimated from modeled revenue contribution divided by spend for each channel.",
            metrics={"best_roas": round(max(artifacts.roas.values()), 4)},
            tables={
                "roas": [
                    {"channel": channel, "roas": round(value, 4)}
                    for channel, value in sorted(artifacts.roas.items(), key=lambda item: item[1], reverse=True)
                ]
            },
            charts=[self._build_roas_chart(artifacts.roas)],
        )

    def optimize_budget(
        self,
        session_id: str,
        monthly_budget: float,
        min_share: float = 0.1,
        max_share: float = 0.5,
    ) -> ToolResult:
        artifacts = self._ensure_model(session_id)
        channels = list(artifacts.roas.keys())
        weights = np.array([max(artifacts.roas[channel], 0.05) for channel in channels], dtype=float)
        weights = weights / weights.sum()

        min_amount = monthly_budget * min_share
        max_amount = monthly_budget * max_share
        proposed = {}
        remaining_budget = monthly_budget
        remaining_weights = weights.copy()

        for idx, channel in enumerate(channels):
            allocation = monthly_budget * weights[idx]
            bounded = min(max(allocation, min_amount), max_amount)
            proposed[channel] = bounded
            remaining_budget -= bounded
            remaining_weights[idx] = 0.0

        if abs(remaining_budget) > 1e-6:
            adjustable = [c for c in channels if min_amount < proposed[c] < max_amount]
            if not adjustable:
                adjustable = channels
            per_channel_delta = remaining_budget / len(adjustable)
            for channel in adjustable:
                proposed[channel] = min(max(proposed[channel] + per_channel_delta, min_amount), max_amount)

        current = self._current_monthly_allocation(session_id, channels)
        incremental_revenue = {
            channel: round(proposed[channel] * artifacts.roas[channel], 2) for channel in channels
        }
        charts = [self._build_budget_chart(current, proposed), self._build_forecast_chart(current, proposed, artifacts.roas)]
        table = [
            {
                "channel": channel,
                "current_budget": round(current[channel], 2),
                "recommended_budget": round(proposed[channel], 2),
                "estimated_revenue": incremental_revenue[channel],
            }
            for channel in channels
        ]
        return ToolResult(
            tool="optimize_budget",
            title="Optimized budget allocation",
            summary="The allocation leans into higher modeled ROAS while respecting simple minimum and maximum share constraints.",
            metrics={
                "monthly_budget": round(monthly_budget, 2),
                "projected_revenue": round(sum(incremental_revenue.values()), 2),
            },
            tables={"budget_recommendation": table},
            charts=charts,
        )

    def forecast(
        self,
        session_id: str,
        scenario_name: str,
        channel_multipliers: dict[str, float],
    ) -> ToolResult:
        artifacts = self._ensure_model(session_id)
        base_budget = self._current_monthly_allocation(session_id, list(artifacts.roas.keys()))
        scenario_budget = {
            channel: round(base_budget[channel] * channel_multipliers.get(channel, 1.0), 2)
            for channel in artifacts.roas
        }
        return self.compare_scenarios(
            session_id,
            scenarios={
                "Current Plan": base_budget,
                scenario_name: scenario_budget,
            },
        )

    def compare_scenarios(self, session_id: str, scenarios: dict[str, dict[str, float]]) -> ToolResult:
        artifacts = self._ensure_model(session_id)
        scenario_table: list[dict[str, Any]] = []
        line_points: list[ChartDataPoint] = []
        for scenario_name, budget_map in scenarios.items():
            forecast_revenue = sum(budget_map[channel] * artifacts.roas[channel] for channel in artifacts.roas)
            lower = forecast_revenue * 0.9
            upper = forecast_revenue * 1.1
            row = {
                "scenario": scenario_name,
                "budget": round(sum(budget_map.values()), 2),
                "forecast_revenue": round(forecast_revenue, 2),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
            }
            scenario_table.append(row)
            line_points.append(
                ChartDataPoint(
                    label=scenario_name,
                    values={
                        "forecast_revenue": row["forecast_revenue"],
                        "lower_bound": row["lower_bound"],
                        "upper_bound": row["upper_bound"],
                    },
                )
            )

        return ToolResult(
            tool="compare_scenarios",
            title="Scenario comparison forecast",
            summary="Scenario outputs are derived from the fitted ROAS profile and include a simple +/-10% confidence band for decision support.",
            tables={"scenarios": scenario_table},
            charts=[
                ChartPayload(
                    id="scenario-forecast",
                    title="Scenario forecast",
                    description="Forecasted revenue and confidence band by scenario.",
                    type="line",
                    data=line_points,
                    series=[
                        ChartSeries(key="forecast_revenue", label="Forecast Revenue", color="#60a5fa"),
                        ChartSeries(key="lower_bound", label="Lower Bound", color="#1d4ed8"),
                        ChartSeries(key="upper_bound", label="Upper Bound", color="#93c5fd"),
                    ],
                    x_key="label",
                    y_key="forecast_revenue",
                )
            ],
            metrics={"best_forecast": max(row["forecast_revenue"] for row in scenario_table)},
        )

    def _ensure_model(self, session_id: str) -> MMMArtifacts:
        if session_id not in self.models:
            self.fit_model(session_id)
        return self.models[session_id]

    def _normalize_columns(self, frame: pd.DataFrame, mapping: ColumnMapping) -> pd.DataFrame:
        missing = [value for value in mapping.model_dump().values() if value not in frame.columns]
        if missing:
            raise ValueError(f"Missing expected columns in CSV: {', '.join(missing)}")
        return frame.rename(
            columns={
                mapping.date: "date",
                mapping.channel: "channel",
                mapping.spend: "spend",
                mapping.revenue: "revenue",
            }
        )[["date", "channel", "spend", "revenue"]]

    def _validate_dataframe(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError("Uploaded CSV is empty.")
        if frame[["date", "channel", "spend", "revenue"]].isnull().any().any():
            raise ValueError("CSV contains null values in required columns.")

    def _build_roas_chart(self, roas: dict[str, float]) -> ChartPayload:
        rows = [
            ChartDataPoint(label=channel, value=round(value, 4), color=CHANNEL_COLORS.get(channel, "#94a3b8"))
            for channel, value in sorted(roas.items(), key=lambda item: item[1], reverse=True)
        ]
        return ChartPayload(
            id="roas-chart",
            title="ROAS by channel",
            description="Modeled return on ad spend for each channel.",
            type="bar",
            data=rows,
            series=[ChartSeries(key="value", label="ROAS", color="#60a5fa")],
            x_key="value",
            y_key="label",
        )

    def _build_contribution_chart(self, contributions: dict[str, float], df: pd.DataFrame) -> ChartPayload:
        rows = []
        for channel, share in sorted(contributions.items(), key=lambda item: item[1], reverse=True):
            rows.append(
                ChartDataPoint(
                    label=channel,
                    value=round(share * 100, 2),
                    color=CHANNEL_COLORS.get(channel, "#94a3b8"),
                    meta={"spend": round(float(df[df["channel"] == channel]["spend"].sum()), 2)},
                )
            )
        return ChartPayload(
            id="contribution-pie",
            title="Revenue contribution share",
            description="Estimated share of modeled revenue contribution by channel.",
            type="pie",
            data=rows,
            series=[ChartSeries(key="value", label="Contribution %")],
        )

    def _build_efficiency_scatter(self, df: pd.DataFrame, roas: dict[str, float]) -> ChartPayload:
        grouped = df.groupby("channel", as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"))
        rows = []
        for row in grouped.to_dict(orient="records"):
            rows.append(
                ChartDataPoint(
                    label=row["channel"],
                    values={"spend": round(float(row["spend"]), 2), "revenue": round(float(row["revenue"]), 2)},
                    color=CHANNEL_COLORS.get(str(row["channel"]), "#94a3b8"),
                    meta={"roas": roas.get(str(row["channel"]), 0.0)},
                )
            )
        return ChartPayload(
            id="efficiency-scatter",
            title="Spend vs revenue",
            description="Channel efficiency across total spend and observed revenue.",
            type="scatter",
            data=rows,
            series=[
                ChartSeries(key="spend", label="Spend", color="#38bdf8"),
                ChartSeries(key="revenue", label="Revenue", color="#34d399"),
            ],
            x_key="spend",
            y_key="revenue",
        )

    def _build_budget_chart(self, current: dict[str, float], proposed: dict[str, float]) -> ChartPayload:
        rows = [
            ChartDataPoint(
                label=channel,
                values={"current": round(current[channel], 2), "recommended": round(proposed[channel], 2)},
                color=CHANNEL_COLORS.get(channel, "#94a3b8"),
            )
            for channel in current
        ]
        return ChartPayload(
            id="budget-optimization",
            title="Budget reallocation",
            description="Current versus recommended monthly budget allocation.",
            type="stacked_bar",
            data=rows,
            series=[
                ChartSeries(key="current", label="Current", color="#475569"),
                ChartSeries(key="recommended", label="Recommended", color="#22c55e"),
            ],
            x_key="label",
            y_key="recommended",
        )

    def _build_forecast_chart(
        self,
        current: dict[str, float],
        proposed: dict[str, float],
        roas: dict[str, float],
    ) -> ChartPayload:
        scenarios = {
            "Current": current,
            "Recommended": proposed,
        }
        rows = []
        for name, allocation in scenarios.items():
            forecast = sum(allocation[channel] * roas[channel] for channel in allocation)
            rows.append(
                ChartDataPoint(
                    label=name,
                    values={
                        "forecast_revenue": round(forecast, 2),
                        "lower_bound": round(forecast * 0.9, 2),
                        "upper_bound": round(forecast * 1.1, 2),
                    },
                )
            )
        return ChartPayload(
            id="forecast-compare",
            title="Forecast comparison",
            description="Projected revenue for current and recommended budget plans.",
            type="line",
            data=rows,
            series=[
                ChartSeries(key="forecast_revenue", label="Forecast Revenue", color="#f59e0b"),
                ChartSeries(key="lower_bound", label="Lower Bound", color="#fbbf24"),
                ChartSeries(key="upper_bound", label="Upper Bound", color="#fde68a"),
            ],
            x_key="label",
            y_key="forecast_revenue",
        )

    def _current_monthly_allocation(self, session_id: str, channels: list[str]) -> dict[str, float]:
        df = self.get_dataframe(session_id)
        total_months = max(df["date"].dt.to_period("M").nunique(), 1)
        current = df.groupby("channel")["spend"].sum().to_dict()
        return {
            channel: round(float(current.get(channel, 0.0) / total_months), 2)
            for channel in channels
        }
