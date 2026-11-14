"""Token-efficient local RAG agent with support-ticket escalation."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from typing import AsyncIterator
import warnings

warnings.filterwarnings("ignore", message="`langchain-community` is being sunset.*", category=DeprecationWarning)

from chromadb.config import Settings
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolRuntime

from .db import (
    count_session_tickets,
    has_captured_lead,
    save_lead,
    save_ticket,
)
from .guardrails import (
    check_output,
    max_tickets_per_session,
    report_tool_bypass,
    resolve_guardrails_enabled,
    tool_error,
    tool_success,
    validate_lead_input,
    validate_ticket_input,
)

SYSTEM_PROMPT = "You are Dot, a concise business assistant. Give direct, factual answers based on provided context. Be professional, brief, and never offer long filler text. When escalating an unresolved problem or handling a human support request, interactively collect: 1. Full Name, 2. Email Address, 3. Company or Team name (optional), 4. Phone number (optional), and 5. A short summary of the core issue and urgency (Low, Medium, or High). Once the user provides this information, call capture_lead FIRST to store it in leads.db. Only after lead capture succeeds may you call create_support_ticket to log the formal support ticket in tickets.db."

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
api_key = os.getenv("ANTHROPIC_API_KEY", "").strip().strip("\"'").strip()
model_name = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5").strip() or "claude-haiku-5-5"


class HashEmbeddings(Embeddings):
    """Local deterministic embeddings requiring no external embedding API."""

    dimensions = 256

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dimensions
            vector[index] += 1.0
        magnitude = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / magnitude for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


knowledge_file = Path(__file__).with_name("knowledge") / "kb.txt"


@lru_cache(maxsize=1)
def get_retriever() -> VectorStoreRetriever:
    """Build and cache the in-memory knowledge index on first use."""
    documents = TextLoader(str(knowledge_file), encoding="utf-8").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(documents)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=HashEmbeddings(),
        collection_name="dot_local_knowledge",
        client_settings=Settings(anonymized_telemetry=False),
    )
    return vector_store.as_retriever(search_kwargs={"k": 2})


def _tool_runtime_settings(runtime: ToolRuntime) -> tuple[str, bool]:
    """Read the session id and guardrails mode injected by LangGraph."""
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    if not isinstance(configurable, dict):
        configurable = {}
    session_id = str(configurable.get("thread_id") or "").strip()
    raw_enabled = configurable.get("guardrails_enabled", True)
    if isinstance(raw_enabled, str):
        guardrails_enabled = raw_enabled.strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}
    else:
        guardrails_enabled = bool(raw_enabled)
    return session_id, guardrails_enabled


def capture_lead_for_session(
    session_id: str,
    *,
    name: str,
    email: str,
    company: str | None = None,
    phone: str | None = None,
    issue_summary: str | None = None,
    urgency: str = "medium",
    guardrails_enabled: bool = True,
) -> str:
    """Validate and persist a lead for a known session.

    The LangChain tool below delegates here, so the same deterministic checks
    also apply when the helper is called outside an LLM tool call (for tests or
    future API endpoints).
    """
    if guardrails_enabled:
        if not session_id:
            return tool_error(
                "missing_session",
                "Unable to verify the session for this tool call. Please retry the conversation.",
            )
        validation = validate_lead_input(
            name=name,
            email=email,
            company=company,
            phone=phone,
            issue_summary=issue_summary,
            urgency=urgency,
        )
        if not validation.ok:
            return tool_error(validation.code, validation.message, field=validation.field)
        lead_values = validation.values
    else:
        # Broken (prompt-only) demo mode intentionally skips deterministic checks.
        lead_values = {
            "name": name or "",
            "email": email or "",
            "company": company,
            "phone": phone,
            "issue_summary": issue_summary,
            "urgency": urgency or "medium",
        }

    try:
        lead_id = save_lead(
            name=lead_values["name"],
            email=lead_values["email"],
            company=lead_values.get("company"),
            phone=lead_values.get("phone"),
            issue_summary=lead_values.get("issue_summary"),
            urgency=lead_values.get("urgency") or "medium",
            session_id=session_id or None,
        )
    except Exception as exc:
        print(f"capture_lead storage error: {exc}", flush=True)
        return tool_error("storage_error", "Unable to save the lead right now. Please try again.")

    return tool_success(lead_id=lead_id, message="Lead captured successfully.")


def create_support_ticket_for_session(
    session_id: str,
    *,
    user_name: str,
    issue_description: str,
    email: str,
    priority: str = "medium",
    guardrails_enabled: bool = True,
) -> str:
    """Enforce ticket preconditions and validation for a known session."""
    if guardrails_enabled:
        if not session_id:
            return tool_error(
                "missing_session",
                "Unable to verify the session for this tool call. Please retry the conversation.",
            )
        try:
            lead_captured = has_captured_lead(session_id)
            existing_tickets = count_session_tickets(session_id) if lead_captured else 0
        except Exception as exc:
            print(f"ticket guardrail state error: {exc}", flush=True)
            return tool_error(
                "session_state_unavailable",
                "Unable to verify this session's support state right now. Please try again.",
            )
        if not lead_captured:
            report_tool_bypass(
                session_id,
                "lead_capture_required",
                "create_support_ticket was called before capture_lead.",
            )
            return tool_error(
                "lead_capture_required",
                "A support ticket cannot be created before a lead is captured. "
                "Call capture_lead with the customer's name, email, and issue summary first.",
            )
        limit = max_tickets_per_session()
        if existing_tickets >= limit:
            return tool_error(
                "ticket_limit_reached",
                f"This session has reached the limit of {limit} support tickets. "
                "Please continue with the existing ticket or contact support directly.",
                limit=limit,
                existing=existing_tickets,
            )
        validation = validate_ticket_input(
            user_name=user_name,
            issue_description=issue_description,
            email=email,
            priority=priority,
        )
        if not validation.ok:
            return tool_error(validation.code, validation.message, field=validation.field)
        ticket_values = validation.values
    else:
        # Broken (prompt-only) demo mode allows the model to skip lead capture.
        ticket_values = {
            "user_name": user_name or "",
            "issue_description": issue_description or "",
            "email": email or "",
            "priority": priority or "medium",
        }

    try:
        ticket_id = save_ticket(
            ticket_values["user_name"],
            ticket_values["issue_description"],
            ticket_values["priority"],
            ticket_values["email"],
            session_id=session_id or None,
        )
    except Exception as exc:
        print(f"create_support_ticket storage error: {exc}", flush=True)
        return tool_error("storage_error", "Unable to create the ticket right now. Please try again.")

    print(json.dumps({
        "ticket_id": ticket_id,
        "reporter": ticket_values["user_name"],
        "email": ticket_values["email"],
        "issue": ticket_values["issue_description"],
        "priority": ticket_values["priority"],
        "status": "open",
    }, sort_keys=True), flush=True)
    return tool_success(
        ticket_id=ticket_id,
        message=f"Support ticket {ticket_id} created successfully.",
    )


@tool
def capture_lead(
    runtime: ToolRuntime,
    name: str,
    email: str,
    company: str | None = None,
    phone: str | None = None,
    issue_summary: str | None = None,
    urgency: str = "medium",
) -> str:
    """Save customer contact and qualification details before creating a support ticket."""
    session_id, guardrails_enabled = _tool_runtime_settings(runtime)
    return capture_lead_for_session(
        session_id,
        name=name,
        email=email,
        company=company,
        phone=phone,
        issue_summary=issue_summary,
        urgency=urgency,
        guardrails_enabled=guardrails_enabled,
    )


@tool
def create_support_ticket(
    runtime: ToolRuntime,
    user_name: str,
    issue_description: str,
    email: str,
    priority: str = "medium",
) -> str:
    """Create an open support ticket for human follow-up."""
    session_id, guardrails_enabled = _tool_runtime_settings(runtime)
    return create_support_ticket_for_session(
        session_id,
        user_name=user_name,
        issue_description=issue_description,
        email=email,
        priority=priority,
        guardrails_enabled=guardrails_enabled,
    )


primary_model = ChatAnthropic(
    model=model_name,
    streaming=True,
    api_key=api_key or None,
    timeout=60,
    max_retries=3,
)
chat_model = primary_model.bind_tools([capture_lead, create_support_ticket])

agent = create_agent(
    model=chat_model,
    tools=[capture_lead, create_support_ticket],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
    name="dot_support_agent",
)


def retrieve_context(query: str) -> str:
    """Return the two most relevant knowledge-base chunks."""
    return "\n\n".join(document.page_content for document in get_retriever().invoke(query)[:2])


@lru_cache(maxsize=1)
def _knowledge_vectors() -> list[list[float]]:
    """Cache the local knowledge-base vectors used by the scope gate."""
    embedding_data = get_retriever().vectorstore.get(include=["embeddings"])
    embeddings = embedding_data.get("embeddings")
    if embeddings is None:
        return []
    return [[float(value) for value in vector] for vector in embeddings]


def get_topic_similarity(query: str) -> float:
    """Return the best cosine similarity between the query and knowledge chunks."""
    vectors = _knowledge_vectors()
    if not vectors:
        return 0.0
    query_vector = HashEmbeddings().embed_query(query)
    return max(
        sum(left * right for left, right in zip(query_vector, vector))
        for vector in vectors
    )


async def _iter_agent_text(message: str, session_id: str, guardrails_enabled: bool) -> AsyncIterator[str]:
    """Yield raw text chunks from the LangGraph agent."""
    context = retrieve_context(message)
    user_prompt = f"Context:\n{context}\n\nQuestion: {message}"
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config={
            "configurable": {
                "thread_id": session_id,
                "guardrails_enabled": guardrails_enabled,
            }
        },
        version="v2",
    ):
        if event["event"] != "on_chat_model_stream":
            continue
        content = getattr(event["data"].get("chunk"), "content", "")
        if isinstance(content, str) and content:
            yield content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    if block:
                        yield block
                elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    yield block["text"]


def _yield_in_chunks(text: str, chunk_size: int = 80) -> list[str]:
    """Split validated output into SSE-friendly pieces without changing text."""
    return [text[index:index + chunk_size] for index in range(0, len(text), chunk_size)]


async def stream_response(
    message: str,
    session_id: str,
    guardrails_enabled: bool | None = None,
) -> AsyncIterator[str]:
    """Stream assistant text with code-enforced execution and output guardrails."""
    # ``main.py`` passes an already-resolved effective mode. When called
    # directly, ``None`` means "use the server default".
    enabled = (
        bool(guardrails_enabled)
        if guardrails_enabled is not None
        else resolve_guardrails_enabled(None)
    )
    text_stream = _iter_agent_text(message, session_id, enabled)

    if not enabled:
        # Broken/prompt-only demo mode: preserve the original token-by-token stream.
        async for text in text_stream:
            yield text
        return

    # Layer 3 must inspect the complete response before any of it reaches the
    # browser. We then release the already-validated text in small SSE chunks.
    buffered: list[str] = []
    async for text in text_stream:
        buffered.append(text)

    generated = "".join(buffered)
    decision = check_output(generated, user_message=message)
    if not decision.allowed:
        print(json.dumps({"guardrail_blocked": decision.event_payload()}, sort_keys=True), flush=True)
        yield decision.message
        return

    for chunk in _yield_in_chunks(generated):
        yield chunk
