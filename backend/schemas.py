from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ChartType = Literal["bar", "pie", "scatter", "line", "stacked_bar"]
MessageRole = Literal["user", "assistant", "system"]


class ColumnMapping(BaseModel):
    date: str = "date"
    channel: str = "channel"
    spend: str = "spend"
    revenue: str = "revenue"


class UploadResponse(BaseModel):
    session_id: str
    file_name: str
    columns: list[str]
    mapping: ColumnMapping
    summary: dict[str, Any]
    preview: list[dict[str, Any]]


class ChartSeries(BaseModel):
    key: str
    label: str
    color: str | None = None


class ChartDataPoint(BaseModel):
    label: str
    value: float | None = None
    values: dict[str, float] | None = None
    color: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class ChartPayload(BaseModel):
    id: str
    title: str
    description: str
    type: ChartType
    data: list[ChartDataPoint]
    series: list[ChartSeries] = Field(default_factory=list)
    x_key: str | None = None
    y_key: str | None = None


class ToolResult(BaseModel):
    tool: str
    title: str
    summary: str
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    tables: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    charts: list[ChartPayload] = Field(default_factory=list)


class AgentResponse(BaseModel):
    text: str
    tool_results: list[ToolResult] = Field(default_factory=list)
    suggested_prompts: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class HistoryMessage(BaseModel):
    id: str
    role: MessageRole
    content: str
    created_at: datetime
    tool_results: list[ToolResult] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]
