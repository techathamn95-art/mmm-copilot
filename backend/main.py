from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from agent import MMMAgentService, serialize_sse
from mmm_engine import MMMEngine
from schemas import (
    ChatHistoryResponse,
    ChatRequest,
    ColumnMapping,
    HistoryMessage,
    UploadResponse,
)


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


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


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
