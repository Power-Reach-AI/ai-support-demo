"""FastAPI entrypoint for the streaming chatbot API."""

from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()

import json
import os
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .db import initialize_database, is_session_locked, unlock_session
from .chain import stream_response
from .guardrails import (
    SESSION_LOCKED_MESSAGE,
    GuardrailDecision,
    check_input,
    handle_guardrail_block,
    resolve_guardrails_enabled,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize local persistence before serving API requests."""
    initialize_database()
    yield


app = FastAPI(title="Business AI Chatbot API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class HistoryTurn(BaseModel):
    role: str = Field(min_length=1, max_length=32)
    content: str = Field(min_length=1, max_length=20000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    session_id: str = Field(min_length=1, max_length=128)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=100)
    guardrails_enabled: bool | None = None


def _sse_headers(guardrails_enabled: bool) -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Guardrails-Enabled": "true" if guardrails_enabled else "false",
    }


def _blocked_response(decision: GuardrailDecision, guardrails_enabled: bool) -> StreamingResponse:
    """Return a frontend-compatible SSE refusal without calling the main LLM."""

    async def events() -> AsyncIterator[str]:
        yield f"data: {json.dumps(decision.event_payload())}\n\n"
        yield f"data: {json.dumps(decision.message)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers=_sse_headers(guardrails_enabled),
    )


def _session_locked_response(guardrails_enabled: bool) -> StreamingResponse:
    """Return the escalated-session message without invoking the LLM."""
    decision = GuardrailDecision(
        allowed=False,
        layer="session",
        code="session_locked",
        message=SESSION_LOCKED_MESSAGE,
    )
    return _blocked_response(decision, guardrails_enabled)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a lightweight liveness response for the API service."""
    return {"status": "ok"}


@app.post("/admin/unlock/{session_id}")
async def admin_unlock(session_id: str) -> dict[str, str]:
    """Unlock a session after a human has reviewed its anomalies."""
    unlock_session(session_id)
    return {"status": "unlocked", "session_id": session_id}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream a session-aware assistant response as server-sent events."""
    guardrails_enabled = resolve_guardrails_enabled(request.guardrails_enabled)

    if is_session_locked(request.session_id):
        print(json.dumps({"session_locked": request.session_id}, sort_keys=True), flush=True)
        return _session_locked_response(guardrails_enabled)

    if guardrails_enabled:
        decision = check_input(
            request.message,
            request.session_id,
            history=request.history,
        )
        if not decision.allowed:
            anomaly = handle_guardrail_block(
                decision,
                request.session_id,
                request.message,
            )
            print(json.dumps({
                "guardrail_blocked": decision.event_payload(),
                "anomaly": anomaly,
            }, sort_keys=True), flush=True)
            return _blocked_response(decision, guardrails_enabled)

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip().strip("\"'").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")

    async def events() -> AsyncIterator[str]:
        """Yield token events and keep provider failures parseable by the client."""
        try:
            async for text in stream_response(
                request.message,
                request.session_id,
                guardrails_enabled=guardrails_enabled,
            ):
                yield f"data: {json.dumps(text)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # Keep the SSE stream parseable for the client.
            print(f"chat stream error: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            yield f"data: {json.dumps({'error': 'Unable to generate a response.'})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers=_sse_headers(guardrails_enabled),
    )
