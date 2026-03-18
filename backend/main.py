from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from agent import MMMAgentService, serialize_sse
from mmm_engine import MMMEngine
from schemas import (
    ChatHistoryResponse,
    ChatRequest,
    ColumnMapping,
    HistoryMessage,
    UploadResponse,
)

SAMPLE_SESSION_ID = "demo"

_llm_config: dict = {"provider": None, "api_key": None, "model": None}


class LLMConfigRequest(BaseModel):
    provider: str  # google, openai, anthropic, groq
    api_key: str
    model: str | None = None


class LLMTestRequest(BaseModel):
    provider: str
    api_key: str
    model: str | None = None


app = FastAPI(title="MMM AI App", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = MMMEngine()
agent_service = MMMAgentService(engine)
chat_history: dict[str, list[HistoryMessage]] = {}


# ── LLM Config ──────────────────────────────────────────────
@app.get("/api/llm-config")
async def get_llm_config() -> JSONResponse:
    return JSONResponse({
        "configured": _llm_config["provider"] is not None,
        "provider": _llm_config["provider"],
        "model": _llm_config["model"],
    })


@app.post("/api/llm-config")
async def set_llm_config(request: LLMConfigRequest) -> JSONResponse:
    global agent_service, _llm_config
    _llm_config = {"provider": request.provider, "api_key": request.api_key, "model": request.model}
    agent_service = MMMAgentService(engine, llm_config=_llm_config)
    return JSONResponse({"status": "ok", "provider": request.provider, "model": request.model})


@app.post("/api/llm-test")
async def test_llm(request: LLMTestRequest) -> JSONResponse:
    """Test the LLM connection with a simple prompt."""
    try:
        test_service = MMMAgentService(engine, llm_config={
            "provider": request.provider,
            "api_key": request.api_key,
            "model": request.model,
        })
        result = await test_service.run(SAMPLE_SESSION_ID, "Say hello in one sentence.")
        return JSONResponse({"status": "ok", "response": result.text[:200]})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"status": "error", "detail": str(exc)})


def append_message(session_id: str, role: str, content: str, tool_results: list | None = None) -> None:
    history = chat_history.setdefault(session_id, [])
    history.append(
        HistoryMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            created_at=datetime.utcnow(),
            tool_results=tool_results or [],
        )
    )


# Auto-load sample data on startup
_sample_csv = Path(__file__).parent.parent / "sample_data.csv"
if _sample_csv.exists():
    engine.parse_csv(SAMPLE_SESSION_ID, _sample_csv.read_bytes(), "sample_data.csv", ColumnMapping())
    append_message(SAMPLE_SESSION_ID, "system", "Demo dataset loaded (6 months, 4 channels, 724 rows).")


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/api/demo", response_model=UploadResponse)
async def demo_session() -> UploadResponse:
    """Return the pre-loaded sample dataset so the frontend can start immediately."""
    try:
        summary = engine.get_summary(SAMPLE_SESSION_ID)
        preview = engine.get_preview(SAMPLE_SESSION_ID)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Demo data not loaded") from exc
    return UploadResponse(
        session_id=SAMPLE_SESSION_ID,
        file_name="sample_data.csv",
        columns=["date", "channel", "spend", "revenue"],
        mapping=ColumnMapping(),
        summary=summary,
        preview=preview,
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    mapping_json: str | None = Form(default=None),
) -> UploadResponse:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    mapping = ColumnMapping.model_validate(json.loads(mapping_json)) if mapping_json else ColumnMapping()
    file_bytes = await file.read()
    session_id = str(uuid.uuid4())

    try:
        upload_result = engine.parse_csv(session_id, file_bytes, file.filename, mapping)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    append_message(
        session_id,
        "system",
        f"Dataset {file.filename} uploaded and validated.",
    )
    return UploadResponse(session_id=session_id, **upload_result)


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    try:
        engine.get_dataframe(request.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    append_message(request.session_id, "user", request.message)

    async def event_stream() -> AsyncGenerator[str, None]:
        yield serialize_sse("start", {"session_id": request.session_id})
        agent_response = await agent_service.run(request.session_id, request.message)
        chunks = _chunk_text(agent_response.text)
        assembled = ""
        for chunk in chunks:
            assembled += chunk
            yield serialize_sse("delta", {"content": chunk})
            await asyncio.sleep(0.03)

        append_message(
            request.session_id,
            "assistant",
            assembled,
            [result.model_dump() for result in agent_response.tool_results],
        )
        yield serialize_sse(
            "message",
            {
                "content": assembled,
                "tool_results": [result.model_dump() for result in agent_response.tool_results],
                "suggested_prompts": agent_response.suggested_prompts,
            },
        )
        yield serialize_sse("done", {"session_id": request.session_id})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/chat/{session_id}/history", response_model=ChatHistoryResponse)
async def history(session_id: str) -> ChatHistoryResponse:
    return ChatHistoryResponse(session_id=session_id, messages=chat_history.get(session_id, []))


def _chunk_text(text: str, chunk_size: int = 24) -> list[str]:
    return [text[idx : idx + chunk_size] for idx in range(0, len(text), chunk_size)] or [text]
