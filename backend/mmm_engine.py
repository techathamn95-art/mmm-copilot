from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from schemas import ChartDataPoint, ChartPayload, ChartSeries, ColumnMapping, ToolResult


CHANNEL_COLORS = {
    "Facebook": "#3b82f6",
    "Google": "#22c55e",
    "TikTok": "#f97316",
    "TV": "#eab308",
}

DEFAULT_ADSTOCK_DECAY = 0.5
DEFAULT_HILL_SLOPE = 2.0
BOOTSTRAP_SAMPLES = 200
DAYS_PER_MONTH = 30.4375
WEEKS_PER_MONTH = DAYS_PER_MONTH / 7.0


def adstock_transform(spend_series: np.ndarray, decay_rate: float = DEFAULT_ADSTOCK_DECAY) -> np.ndarray:
    """Geometric adstock carryover applied on a daily spend series."""
    spend = np.asarray(spend_series, dtype=float)
    if spend.size == 0:
        return spend.copy()
    adstock = np.zeros_like(spend, dtype=float)
    adstock[0] = spend[0]
    for idx in range(1, len(spend)):
        adstock[idx] = spend[idx] + decay_rate * adstock[idx - 1]
    return adstock


def hill_saturation(
    adstock: np.ndarray,
    half_saturation: float,
    slope: float = DEFAULT_HILL_SLOPE,
) -> np.ndarray:
    """Hill saturation curve with diminishing returns."""
    x = np.maximum(np.asarray(adstock, dtype=float), 0.0)
    half = max(float(half_saturation), 1e-6)
    exp = max(float(slope), 1e-6)
    numerator = np.power(x, exp)
    denominator = numerator + np.power(half, exp)
    return np.divide(numerator, np.maximum(denominator, 1e-12))


def hill_derivative(adstock: np.ndarray, half_saturation: float, slope: float = DEFAULT_HILL_SLOPE) -> np.ndarray:
    """Derivative of the Hill curve with respect to adstock."""
    x = np.maximum(np.asarray(adstock, dtype=float), 1e-12)
    half = max(float(half_saturation), 1e-6)
    exp = max(float(slope), 1e-6)
    half_term = np.power(half, exp)
    numerator = exp * np.power(x, exp - 1.0) * half_term
    denominator = np.power(np.power(x, exp) + half_term, 2.0)
    return np.divide(numerator, np.maximum(denominator, 1e-12))


@dataclass
class MMMArtifacts:
    model: Ridge
    feature_columns: list[str]
    coefficients: dict[str, float]
    intercept: float
    roas: dict[str, float]
    contributions: dict[str, float]
    weekly_summary: list[dict[str, Any]]
    r2: float
    weekly_features: pd.DataFrame
    weekly_target: pd.Series
    weekly_predictions: np.ndarray
    daily_spend: pd.DataFrame
    daily_adstock: pd.DataFrame
    daily_saturation: pd.DataFrame
    weekly_spend: pd.DataFrame
    weekly_contribution: pd.DataFrame
    spend_by_channel: dict[str, float]
    channel_params: dict[str, dict[str, float]]
    bootstrap_intercepts: np.ndarray
    bootstrap_coefficients: np.ndarray
    bootstrap_roas: dict[str, np.ndarray]
    bootstrap_contributions: dict[str, np.ndarray]
    bootstrap_predictions: np.ndarray
    current_monthly_allocation: dict[str, float]


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
        self.models.pop(session_id, None)

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
        prepared = self._prepare_model_inputs(df)

        model = Ridge(alpha=1.0)
        model.fit(prepared["weekly_features"], prepared["weekly_target"])
        predictions = model.predict(prepared["weekly_features"])
        r2 = float(model.score(prepared["weekly_features"], prepared["weekly_target"]))

        feature_columns = prepared["weekly_features"].columns.tolist()
        coefficients = {col: float(coef) for col, coef in zip(feature_columns, model.coef_)}
        raw_contributions = {
            channel: float(prepared["weekly_features"][channel].sum() * coefficients.get(channel, 0.0))
            for channel in feature_columns
        }
        positive_total = sum(max(value, 0.0) for value in raw_contributions.values()) or 1.0
        contributions = {
            channel: max(value, 0.0) / positive_total
            for channel, value in raw_contributions.items()
        }
        spend_by_channel = df.groupby("channel")["spend"].sum().to_dict()
        roas = {
            channel: round(
                max(raw_contributions.get(channel, 0.0), 0.0) / max(spend_by_channel.get(channel, 1.0), 1.0),
                4,
            )
            for channel in feature_columns
        }

        bootstrap = self._bootstrap_model(
            weekly_features=prepared["weekly_features"],
            weekly_target=prepared["weekly_target"],
            fitted=predictions,
            residuals=prepared["weekly_target"].to_numpy(dtype=float) - predictions,
            feature_columns=feature_columns,
            spend_by_channel=spend_by_channel,
        )

        weekly_summary = self._build_weekly_summary(
            weekly_index=prepared["weekly_features"].index,
            actual=prepared["weekly_target"].to_numpy(dtype=float),
            predicted=predictions,
            bootstrap_predictions=bootstrap["predictions"],
        )

        weekly_contribution = pd.DataFrame(
            {
                channel: prepared["weekly_features"][channel] * coefficients.get(channel, 0.0)
                for channel in feature_columns
            },
            index=prepared["weekly_features"].index,
        )
        current_monthly_allocation = self._current_monthly_allocation(session_id, feature_columns)

        self.models[session_id] = MMMArtifacts(
            model=model,
            feature_columns=feature_columns,
            coefficients=coefficients,
            intercept=float(model.intercept_),
            roas=roas,
            contributions=contributions,
            weekly_summary=weekly_summary,
            r2=r2,
            weekly_features=prepared["weekly_features"],
            weekly_target=prepared["weekly_target"],
            weekly_predictions=predictions,
            daily_spend=prepared["daily_spend"],
            daily_adstock=prepared["daily_adstock"],
            daily_saturation=prepared["daily_saturation"],
            weekly_spend=prepared["weekly_spend"],
            weekly_contribution=weekly_contribution,
            spend_by_channel=spend_by_channel,
            channel_params=prepared["channel_params"],
            bootstrap_intercepts=bootstrap["intercepts"],
            bootstrap_coefficients=bootstrap["coefficients"],
            bootstrap_roas=bootstrap["roas"],
            bootstrap_contributions=bootstrap["contributions"],
            bootstrap_predictions=bootstrap["predictions"],
            current_monthly_allocation=current_monthly_allocation,
        )

        return ToolResult(
            tool="fit_model",
            title="Marketing mix model fitted",
            summary="A ridge MMM was trained on weekly outcomes using daily adstock and Hill saturation transforms, with bootstrap intervals for fitted revenue.",
            metrics={
                "r2": round(r2, 4),
                "weeks_modeled": len(prepared["weekly_features"].index),
                "predicted_revenue": round(float(np.sum(predictions)), 2),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
            },
            tables={
                "weekly_performance": weekly_summary,
                "channel_parameters": self._channel_parameter_rows(prepared["channel_params"]),
            },
            charts=[
                self._build_roas_chart(roas, bootstrap["roas"]),
                self._build_contribution_chart(contributions, bootstrap["contributions"], df),
                self._build_efficiency_scatter(df, roas),
                self._build_weekly_fit_chart(weekly_summary),
            ],
        )

    def get_roas(self, session_id: str) -> ToolResult:
        artifacts = self._ensure_model(session_id)
        rows = []
        for channel, value in sorted(artifacts.roas.items(), key=lambda item: item[1], reverse=True):
            lower, upper = self._percentile_interval(artifacts.bootstrap_roas[channel])
            rows.append(
                {
                    "channel": channel,
                    "roas": round(value, 4),
                    "lower_bound": round(lower, 4),
                    "upper_bound": round(upper, 4),
                }
            )
        return ToolResult(
            tool="get_roas",
            title="Channel ROAS",
            summary="ROAS reflects modeled channel contribution after adstock and saturation, with bootstrap 5th and 95th percentile intervals.",
            metrics={"best_roas": round(max(artifacts.roas.values()), 4)},
            tables={"roas": rows},
            charts=[self._build_roas_chart(artifacts.roas, artifacts.bootstrap_roas)],
        )

    def optimize_budget(
        self,
        session_id: str,
        monthly_budget: float,
        min_share: float = 0.1,
        max_share: float = 0.5,
    ) -> ToolResult:
        artifacts = self._ensure_model(session_id)
        channels = artifacts.feature_columns

        lower_share = max(min_share, 0.0)
        upper_share = max(max_share, 0.0)
        if lower_share * len(channels) > 1.0:
            lower_share = 0.0
        if upper_share * len(channels) < 1.0:
            upper_share = 1.0

        lower_bound = monthly_budget * lower_share
        upper_bound = monthly_budget * upper_share
        bounds = [(lower_bound, upper_bound) for _ in channels]

        current = self._current_monthly_allocation(session_id, channels)
        start = np.array([current[channel] for channel in channels], dtype=float)
        if start.sum() <= 0:
            start = np.full(len(channels), monthly_budget / max(len(channels), 1), dtype=float)
        start = self._project_allocation_to_bounds(start, monthly_budget, lower_bound, upper_bound)

        def objective(values: np.ndarray) -> float:
            allocation = {channel: float(value) for channel, value in zip(channels, values)}
            return -self._predict_monthly_revenue(artifacts, allocation)

        result = minimize(
            objective,
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "eq", "fun": lambda x: np.sum(x) - monthly_budget}],
        )

        if result.success:
            proposed_values = self._project_allocation_to_bounds(result.x, monthly_budget, lower_bound, upper_bound)
        else:
            weights = np.array(
                [max(self._marginal_roi_for_budget(artifacts, channel, current[channel]), 1e-6) for channel in channels],
                dtype=float,
            )
            weights = weights / max(weights.sum(), 1e-9)
            proposed_values = self._project_allocation_to_bounds(
                weights * monthly_budget,
                monthly_budget,
                lower_bound,
                upper_bound,
            )

        proposed = {channel: float(value) for channel, value in zip(channels, proposed_values)}
        current_forecast, current_lower, current_upper = self._forecast_interval(artifacts, current)
        proposed_forecast, proposed_lower, proposed_upper = self._forecast_interval(artifacts, proposed)

        table = []
        for channel in channels:
            channel_revenue = self._channel_monthly_revenue(artifacts, channel, proposed[channel])
            table.append(
                {
                    "channel": channel,
                    "current_budget": round(current[channel], 2),
                    "recommended_budget": round(proposed[channel], 2),
                    "estimated_revenue": round(channel_revenue, 2),
                    "marginal_roi": round(self._marginal_roi_for_budget(artifacts, channel, proposed[channel]), 4),
                }
            )

        charts = [
            self._build_budget_chart(current, proposed),
            self._build_forecast_chart(
                {
                    "Current": (current_forecast, current_lower, current_upper),
                    "Recommended": (proposed_forecast, proposed_lower, proposed_upper),
                }
            ),
        ]
        return ToolResult(
            tool="optimize_budget",
            title="Optimized budget allocation",
            summary="Budget was optimized with constrained nonlinear response curves rather than proportional ROAS weighting, so diminishing returns are respected channel by channel.",
            metrics={
                "monthly_budget": round(monthly_budget, 2),
                "projected_revenue": round(proposed_forecast, 2),
                "optimizer_success": str(bool(result.success)),
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
        base_budget = self._current_monthly_allocation(session_id, artifacts.feature_columns)
        scenario_budget = {
            channel: round(base_budget[channel] * channel_multipliers.get(channel, 1.0), 2)
            for channel in artifacts.feature_columns
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
        forecast_map: dict[str, tuple[float, float, float]] = {}

        for scenario_name, budget_map in scenarios.items():
            normalized_budget = {
                channel: float(budget_map.get(channel, 0.0))
                for channel in artifacts.feature_columns
            }
            forecast_revenue, lower, upper = self._forecast_interval(artifacts, normalized_budget)
            row = {
                "scenario": scenario_name,
                "budget": round(sum(normalized_budget.values()), 2),
                "forecast_revenue": round(forecast_revenue, 2),
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
            }
            scenario_table.append(row)
            forecast_map[scenario_name] = (forecast_revenue, lower, upper)

        return ToolResult(
            tool="compare_scenarios",
            title="Scenario comparison forecast",
            summary="Scenario forecasts use the nonlinear transformed MMM and bootstrap percentile intervals instead of fixed percentage bands.",
            tables={"scenarios": scenario_table},
            charts=[self._build_forecast_chart(forecast_map)],
            metrics={"best_forecast": max(row["forecast_revenue"] for row in scenario_table)},
        )

    def analyze_adstock(self, session_id: str) -> ToolResult:
        artifacts = self._ensure_model(session_id)
        horizon = 14
        rows = []
        for day in range(horizon + 1):
            values: dict[str, float] = {}
            for channel in artifacts.feature_columns:
                impulse = np.zeros(horizon + 1, dtype=float)
                impulse[0] = artifacts.channel_params[channel]["median_daily_spend"]
                values[channel] = round(
                    adstock_transform(impulse, artifacts.channel_params[channel]["decay_rate"])[day],
                    4,
                )
            rows.append(ChartDataPoint(label=f"Day {day}", values=values))

        table = []
        for channel in artifacts.feature_columns:
            decay = artifacts.channel_params[channel]["decay_rate"]
            half_life = np.log(0.5) / np.log(decay) if 0.0 < decay < 1.0 else 0.0
            table.append(
                {
                    "channel": channel,
                    "decay_rate": round(decay, 4),
                    "half_life_days": round(float(half_life), 2),
                    "median_daily_spend": round(artifacts.channel_params[channel]["median_daily_spend"], 2),
                }
            )

        return ToolResult(
            tool="analyze_adstock",
            title="Adstock carryover analysis",
            summary="Adstock curves show how a one-day spend impulse persists over time under the modeled geometric decay.",
            tables={"adstock_parameters": table},
            charts=[
                ChartPayload(
                    id="adstock-curves",
                    title="Adstock decay curves",
                    description="Effective carryover from a one-day spend impulse using the channel adstock settings.",
                    type="line",
                    data=rows,
                    series=[
                        ChartSeries(key=channel, label=channel, color=CHANNEL_COLORS.get(channel, "#94a3b8"))
                        for channel in artifacts.feature_columns
                    ],
                    x_key="label",
                    y_key=artifacts.feature_columns[0] if artifacts.feature_columns else None,
                )
            ],
        )

    def analyze_saturation(self, session_id: str) -> ToolResult:
        artifacts = self._ensure_model(session_id)
        rows = []
        spend_grid = np.linspace(0.0, 2.0, 11)
        for multiplier in spend_grid:
            values: dict[str, float] = {}
            label = f"{multiplier:.1f}x"
            for channel in artifacts.feature_columns:
                baseline = max(artifacts.current_monthly_allocation[channel], 1e-6)
                values[channel] = round(
                    self._channel_monthly_revenue(artifacts, channel, baseline * multiplier),
                    4,
                )
            rows.append(ChartDataPoint(label=label, values=values))

        table = []
        for channel in artifacts.feature_columns:
            params = artifacts.channel_params[channel]
            table.append(
                {
                    "channel": channel,
                    "half_saturation": round(params["half_saturation"], 4),
                    "hill_slope": round(params["slope"], 4),
                    "current_monthly_spend": round(artifacts.current_monthly_allocation[channel], 2),
                }
            )

        return ToolResult(
            tool="analyze_saturation",
            title="Saturation curve analysis",
            summary="Saturation curves map higher spend into smaller incremental response as each channel approaches its modeled saturation point.",
            tables={"saturation_parameters": table},
            charts=[
                ChartPayload(
                    id="saturation-curves",
                    title="Monthly spend response curves",
                    description="Modeled monthly revenue contribution as spend scales from zero to 2x the current monthly baseline.",
                    type="line",
                    data=rows,
                    series=[
                        ChartSeries(key=channel, label=channel, color=CHANNEL_COLORS.get(channel, "#94a3b8"))
                        for channel in artifacts.feature_columns
                    ],
                    x_key="label",
                    y_key=artifacts.feature_columns[0] if artifacts.feature_columns else None,
                )
            ],
        )

    def get_marginal_roi(self, session_id: str) -> ToolResult:
        artifacts = self._ensure_model(session_id)
        rows = []
        marginal_roi: dict[str, float] = {}
        for channel in artifacts.feature_columns:
            current_spend = artifacts.current_monthly_allocation[channel]
            value = self._marginal_roi_for_budget(artifacts, channel, current_spend)
            marginal_roi[channel] = value
            rows.append(
                {
                    "channel": channel,
                    "current_monthly_spend": round(current_spend, 2),
                    "marginal_roi": round(value, 4),
                    "average_roas": round(artifacts.roas[channel], 4),
                }
            )
        return ToolResult(
            tool="get_marginal_roi",
            title="Marginal ROI by channel",
            summary="Marginal ROI estimates the next dollar's modeled return for each channel after adstock and saturation effects.",
            tables={"marginal_roi": rows},
            charts=[
                ChartPayload(
                    id="marginal-roi",
                    title="Marginal ROI",
                    description="Incremental modeled return on the next dollar at current spend levels.",
                    type="bar",
                    data=[
                        ChartDataPoint(
                            label=channel,
                            value=round(value, 4),
                            color=CHANNEL_COLORS.get(channel, "#94a3b8"),
                        )
                        for channel, value in sorted(marginal_roi.items(), key=lambda item: item[1], reverse=True)
                    ],
                    series=[ChartSeries(key="value", label="Marginal ROI", color="#60a5fa")],
                    x_key="value",
                    y_key="label",
                )
            ],
        )

    def channel_deep_dive(self, session_id: str, channel: str) -> ToolResult:
        artifacts = self._ensure_model(session_id)
        resolved = self._resolve_channel_name(channel, artifacts.feature_columns)
        if resolved is None:
            raise ValueError(f"Channel '{channel}' was not found in the dataset.")

        params = artifacts.channel_params[resolved]
        weekly_breakdown = pd.DataFrame(
            {
                "week": [idx.strftime("%Y-%m-%d") for idx in artifacts.weekly_spend.index],
                "spend": artifacts.weekly_spend[resolved].round(2).tolist(),
                "adstock": artifacts.daily_adstock[resolved].resample("W-SUN").mean().round(4).tolist(),
                "saturation": artifacts.weekly_features[resolved].round(4).tolist(),
                "contribution": artifacts.weekly_contribution[resolved].round(4).tolist(),
            }
        ).tail(12)

        summary_row = [
            {
                "channel": resolved,
                "total_spend": round(artifacts.spend_by_channel[resolved], 2),
                "roas": round(artifacts.roas[resolved], 4),
                "contribution_share": round(artifacts.contributions[resolved] * 100.0, 2),
                "marginal_roi": round(
                    self._marginal_roi_for_budget(artifacts, resolved, artifacts.current_monthly_allocation[resolved]),
                    4,
                ),
                "decay_rate": round(params["decay_rate"], 4),
                "half_saturation": round(params["half_saturation"], 4),
                "hill_slope": round(params["slope"], 4),
            }
        ]

        recent_rows = [
            ChartDataPoint(
                label=row["week"],
                values={
                    "spend": float(row["spend"]),
                    "adstock": float(row["adstock"]),
                    "saturation": float(row["saturation"]),
                    "contribution": float(row["contribution"]),
                },
                color=CHANNEL_COLORS.get(resolved, "#94a3b8"),
            )
            for row in weekly_breakdown.to_dict(orient="records")
        ]

        return ToolResult(
            tool="channel_deep_dive",
            title=f"{resolved} channel deep dive",
            summary="This view combines spend trend, carryover, saturation, and modeled contribution for a single channel.",
            tables={
                "channel_summary": summary_row,
                "weekly_breakdown": weekly_breakdown.to_dict(orient="records"),
            },
            charts=[
                ChartPayload(
                    id=f"{resolved.lower()}-trend",
                    title=f"{resolved} spend and adstock",
                    description="Weekly spend against average weekly adstocked pressure.",
                    type="line",
                    data=recent_rows,
                    series=[
                        ChartSeries(key="spend", label="Spend", color="#38bdf8"),
                        ChartSeries(key="adstock", label="Adstock", color="#f59e0b"),
                    ],
                    x_key="label",
                    y_key="spend",
                ),
                ChartPayload(
                    id=f"{resolved.lower()}-response",
                    title=f"{resolved} saturation and contribution",
                    description="Weekly transformed feature value and modeled contribution over time.",
                    type="line",
                    data=recent_rows,
                    series=[
                        ChartSeries(key="saturation", label="Saturation", color="#22c55e"),
                        ChartSeries(key="contribution", label="Contribution", color="#6366f1"),
                    ],
                    x_key="label",
                    y_key="contribution",
                ),
            ],
        )

    def _ensure_model(self, session_id: str) -> MMMArtifacts:
        if session_id not in self.models:
            self.fit_model(session_id)
        return self.models[session_id]

    def _prepare_model_inputs(self, df: pd.DataFrame) -> dict[str, Any]:
        channels = sorted(df["channel"].astype(str).unique().tolist())
        full_index = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

        daily_spend = (
            df.pivot_table(index="date", columns="channel", values="spend", aggfunc="sum", fill_value=0.0)
            .reindex(full_index, fill_value=0.0)
            .sort_index()
        )
        daily_spend.index.name = "date"
        daily_spend = daily_spend.reindex(columns=channels, fill_value=0.0)

        daily_revenue = (
            df.groupby("date")["revenue"]
            .sum()
            .reindex(full_index, fill_value=0.0)
            .sort_index()
        )
        daily_revenue.index.name = "date"

        daily_adstock = pd.DataFrame(index=full_index)
        daily_saturation = pd.DataFrame(index=full_index)
        channel_params: dict[str, dict[str, float]] = {}
        for channel in channels:
            spend_values = daily_spend[channel].to_numpy(dtype=float)
            median_spend = float(np.median(spend_values[spend_values > 0])) if np.any(spend_values > 0) else 1.0
            channel_params[channel] = {
                "decay_rate": DEFAULT_ADSTOCK_DECAY,
                "half_saturation": max(median_spend, 1e-6),
                "slope": DEFAULT_HILL_SLOPE,
                "median_daily_spend": max(median_spend, 1e-6),
            }
            adstocked = adstock_transform(spend_values, DEFAULT_ADSTOCK_DECAY)
            saturated = hill_saturation(adstocked, median_spend, DEFAULT_HILL_SLOPE)
            daily_adstock[channel] = adstocked
            daily_saturation[channel] = saturated

        weekly_features = daily_saturation.resample("W-SUN").sum()
        weekly_target = daily_revenue.resample("W-SUN").sum().reindex(weekly_features.index, fill_value=0.0)
        weekly_spend = daily_spend.resample("W-SUN").sum().reindex(weekly_features.index, fill_value=0.0)

        return {
            "daily_spend": daily_spend,
            "daily_adstock": daily_adstock,
            "daily_saturation": daily_saturation,
            "weekly_features": weekly_features,
            "weekly_target": weekly_target,
            "weekly_spend": weekly_spend,
            "channel_params": channel_params,
        }

    def _bootstrap_model(
        self,
        weekly_features: pd.DataFrame,
        weekly_target: pd.Series,
        fitted: np.ndarray,
        residuals: np.ndarray,
        feature_columns: list[str],
        spend_by_channel: dict[str, float],
    ) -> dict[str, Any]:
        rng = np.random.default_rng(42)
        coefficient_samples = np.zeros((BOOTSTRAP_SAMPLES, len(feature_columns)), dtype=float)
        intercept_samples = np.zeros(BOOTSTRAP_SAMPLES, dtype=float)
        prediction_samples = np.zeros((BOOTSTRAP_SAMPLES, len(weekly_target)), dtype=float)
        roas_samples = {channel: np.zeros(BOOTSTRAP_SAMPLES, dtype=float) for channel in feature_columns}
        contribution_samples = {channel: np.zeros(BOOTSTRAP_SAMPLES, dtype=float) for channel in feature_columns}

        x = weekly_features
        fitted_array = np.asarray(fitted, dtype=float)
        residual_array = np.asarray(residuals, dtype=float)
        for idx in range(BOOTSTRAP_SAMPLES):
            sampled_residuals = rng.choice(residual_array, size=len(residual_array), replace=True)
            y_boot = fitted_array + sampled_residuals
            boot_model = Ridge(alpha=1.0)
            boot_model.fit(x, y_boot)
            boot_predictions = boot_model.predict(x)
            prediction_samples[idx] = boot_predictions
            intercept_samples[idx] = float(boot_model.intercept_)
            coefficient_samples[idx] = boot_model.coef_

            raw_contributions = {
                channel: float(x[channel].sum() * coef)
                for channel, coef in zip(feature_columns, boot_model.coef_)
            }
            positive_total = sum(max(value, 0.0) for value in raw_contributions.values()) or 1.0
            for channel in feature_columns:
                roas_samples[channel][idx] = max(raw_contributions.get(channel, 0.0), 0.0) / max(
                    spend_by_channel.get(channel, 1.0),
                    1.0,
                )
                contribution_samples[channel][idx] = max(raw_contributions.get(channel, 0.0), 0.0) / positive_total

        return {
            "coefficients": coefficient_samples,
            "intercepts": intercept_samples,
            "predictions": prediction_samples,
            "roas": roas_samples,
            "contributions": contribution_samples,
        }

    def _build_weekly_summary(
        self,
        weekly_index: pd.Index,
        actual: np.ndarray,
        predicted: np.ndarray,
        bootstrap_predictions: np.ndarray,
    ) -> list[dict[str, Any]]:
        lower = np.percentile(bootstrap_predictions, 5, axis=0)
        upper = np.percentile(bootstrap_predictions, 95, axis=0)
        rows = []
        for idx, date_value in enumerate(weekly_index):
            rows.append(
                {
                    "week": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                    "actual_revenue": round(float(actual[idx]), 2),
                    "predicted_revenue": round(float(predicted[idx]), 2),
                    "lower_bound": round(float(lower[idx]), 2),
                    "upper_bound": round(float(upper[idx]), 2),
                }
            )
        return rows

    def _predict_monthly_revenue(self, artifacts: MMMArtifacts, allocation: dict[str, float]) -> float:
        feature_vector = np.array(
            [self._steady_state_weekly_feature(artifacts, channel, allocation.get(channel, 0.0)) for channel in artifacts.feature_columns],
            dtype=float,
        )
        coefficients = np.array([artifacts.coefficients[channel] for channel in artifacts.feature_columns], dtype=float)
        return float(artifacts.intercept * WEEKS_PER_MONTH + np.dot(feature_vector, coefficients))

    def _forecast_interval(self, artifacts: MMMArtifacts, allocation: dict[str, float]) -> tuple[float, float, float]:
        mean_forecast = self._predict_monthly_revenue(artifacts, allocation)
        feature_vector = np.array(
            [
                self._steady_state_weekly_feature(artifacts, channel, allocation.get(channel, 0.0))
                for channel in artifacts.feature_columns
            ],
            dtype=float,
        )
        bootstrap_forecasts = []
        for intercept, coefficients in zip(artifacts.bootstrap_intercepts, artifacts.bootstrap_coefficients):
            bootstrap_forecasts.append(float(intercept * WEEKS_PER_MONTH + np.dot(feature_vector, coefficients)))
        lower, upper = self._percentile_interval(np.array(bootstrap_forecasts, dtype=float))
        return mean_forecast, lower, upper

    def _steady_state_weekly_feature(self, artifacts: MMMArtifacts, channel: str, monthly_budget: float) -> float:
        params = artifacts.channel_params[channel]
        daily_spend = max(float(monthly_budget), 0.0) / DAYS_PER_MONTH
        steady_adstock = daily_spend / max(1.0 - params["decay_rate"], 1e-6)
        saturated = hill_saturation(np.array([steady_adstock]), params["half_saturation"], params["slope"])[0]
        return float(saturated * 7.0)

    def _channel_monthly_revenue(self, artifacts: MMMArtifacts, channel: str, monthly_budget: float) -> float:
        coefficient = artifacts.coefficients.get(channel, 0.0)
        return float(coefficient * self._steady_state_weekly_feature(artifacts, channel, monthly_budget) * WEEKS_PER_MONTH)

    def _marginal_roi_for_budget(self, artifacts: MMMArtifacts, channel: str, monthly_budget: float) -> float:
        params = artifacts.channel_params[channel]
        coefficient = artifacts.coefficients.get(channel, 0.0)
        daily_spend = max(float(monthly_budget), 0.0) / DAYS_PER_MONTH
        steady_adstock = daily_spend / max(1.0 - params["decay_rate"], 1e-6)
        saturation_slope = hill_derivative(np.array([steady_adstock]), params["half_saturation"], params["slope"])[0]
        d_adstock_d_budget = 1.0 / (DAYS_PER_MONTH * max(1.0 - params["decay_rate"], 1e-6))
        return float(WEEKS_PER_MONTH * coefficient * 7.0 * saturation_slope * d_adstock_d_budget)

    def _resolve_channel_name(self, channel: str, channels: list[str]) -> str | None:
        channel_map = {value.lower(): value for value in channels}
        return channel_map.get(channel.lower())

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

    def _build_roas_chart(self, roas: dict[str, float], bootstrap_roas: dict[str, np.ndarray]) -> ChartPayload:
        rows = []
        for channel, value in sorted(roas.items(), key=lambda item: item[1], reverse=True):
            lower, upper = self._percentile_interval(bootstrap_roas[channel])
            rows.append(
                ChartDataPoint(
                    label=channel,
                    value=round(value, 4),
                    color=CHANNEL_COLORS.get(channel, "#94a3b8"),
                    meta={"lower_bound": round(lower, 4), "upper_bound": round(upper, 4)},
                )
            )
        return ChartPayload(
            id="roas-chart",
            title="ROAS by channel",
            description="Modeled return on ad spend by channel with bootstrap confidence intervals.",
            type="bar",
            data=rows,
            series=[ChartSeries(key="value", label="ROAS", color="#60a5fa")],
            x_key="value",
            y_key="label",
        )

    def _build_contribution_chart(
        self,
        contributions: dict[str, float],
        bootstrap_contributions: dict[str, np.ndarray],
        df: pd.DataFrame,
    ) -> ChartPayload:
        rows = []
        for channel, share in sorted(contributions.items(), key=lambda item: item[1], reverse=True):
            lower, upper = self._percentile_interval(bootstrap_contributions[channel] * 100.0)
            rows.append(
                ChartDataPoint(
                    label=channel,
                    value=round(share * 100.0, 2),
                    color=CHANNEL_COLORS.get(channel, "#94a3b8"),
                    meta={
                        "spend": round(float(df[df["channel"] == channel]["spend"].sum()), 2),
                        "lower_bound": round(lower, 2),
                        "upper_bound": round(upper, 2),
                    },
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
            description="Channel efficiency across observed spend and revenue.",
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

    def _build_forecast_chart(self, scenarios: dict[str, tuple[float, float, float]]) -> ChartPayload:
        rows = [
            ChartDataPoint(
                label=name,
                values={
                    "forecast_revenue": round(values[0], 2),
                    "lower_bound": round(values[1], 2),
                    "upper_bound": round(values[2], 2),
                },
            )
            for name, values in scenarios.items()
        ]
        return ChartPayload(
            id="forecast-compare",
            title="Forecast comparison",
            description="Projected revenue by scenario with bootstrap percentile intervals.",
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

    def _build_weekly_fit_chart(self, weekly_summary: list[dict[str, Any]]) -> ChartPayload:
        return ChartPayload(
            id="weekly-fit",
            title="Weekly actual vs predicted revenue",
            description="Observed weekly revenue, model fit, and bootstrap interval over the modeled period.",
            type="line",
            data=[
                ChartDataPoint(
                    label=row["week"],
                    values={
                        "actual_revenue": float(row["actual_revenue"]),
                        "predicted_revenue": float(row["predicted_revenue"]),
                        "lower_bound": float(row["lower_bound"]),
                        "upper_bound": float(row["upper_bound"]),
                    },
                )
                for row in weekly_summary
            ],
            series=[
                ChartSeries(key="actual_revenue", label="Actual Revenue", color="#0f172a"),
                ChartSeries(key="predicted_revenue", label="Predicted Revenue", color="#60a5fa"),
                ChartSeries(key="lower_bound", label="Lower Bound", color="#93c5fd"),
                ChartSeries(key="upper_bound", label="Upper Bound", color="#dbeafe"),
            ],
            x_key="label",
            y_key="predicted_revenue",
        )

    def _current_monthly_allocation(self, session_id: str, channels: list[str]) -> dict[str, float]:
        df = self.get_dataframe(session_id)
        total_months = max(df["date"].dt.to_period("M").nunique(), 1)
        current = df.groupby("channel")["spend"].sum().to_dict()
        return {
            channel: round(float(current.get(channel, 0.0) / total_months), 2)
            for channel in channels
        }

    def _channel_parameter_rows(self, channel_params: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
        rows = []
        for channel, params in channel_params.items():
            rows.append(
                {
                    "channel": channel,
                    "adstock_decay": round(params["decay_rate"], 4),
                    "half_saturation": round(params["half_saturation"], 4),
                    "hill_slope": round(params["slope"], 4),
                }
            )
        return rows

    def _percentile_interval(self, values: np.ndarray) -> tuple[float, float]:
        array = np.asarray(values, dtype=float)
        return float(np.percentile(array, 5)), float(np.percentile(array, 95))

    def _project_allocation_to_bounds(
        self,
        values: np.ndarray,
        total_budget: float,
        lower_bound: float,
        upper_bound: float,
    ) -> np.ndarray:
        projected = np.clip(np.asarray(values, dtype=float), lower_bound, upper_bound)
        for _ in range(20):
            delta = float(total_budget - projected.sum())
            if abs(delta) <= 1e-6:
                break
            if delta > 0:
                room = upper_bound - projected
            else:
                room = projected - lower_bound
            adjustable = room > 1e-9
            if not np.any(adjustable):
                break
            weights = room[adjustable] / max(room[adjustable].sum(), 1e-9)
            projected[adjustable] += np.sign(delta) * min(abs(delta), room[adjustable].sum()) * weights
            projected = np.clip(projected, lower_bound, upper_bound)
        return projected
