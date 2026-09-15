"""FastAPI entrypoint for the streaming chatbot API."""

import json
import os
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .db import initialize_database
from .chain import stream_response


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


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    session_id: str = Field(min_length=1, max_length=128)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a lightweight liveness response for the API service."""
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream a session-aware assistant response as server-sent events."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip().strip("\"'").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")

    async def events() -> AsyncIterator[str]:
        """Yield token events and keep provider failures parseable by the client."""
        try:
            async for text in stream_response(request.message, request.session_id):
                yield f"data: {json.dumps(text)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # Keep the SSE stream parseable for the client.
            print(f"chat stream error: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            yield f"data: {json.dumps({'error': 'Unable to generate a response.'})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
