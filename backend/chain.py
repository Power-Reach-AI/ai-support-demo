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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

from .db import save_lead, save_ticket

SYSTEM_PROMPT = "You are Dot, a concise business assistant. Give direct, factual answers based on provided context. Be professional, brief, and never offer long filler text. When human escalation is needed, collect the user's full name, email address, company or team name, phone number if available, a short issue summary, and urgency as Low, Medium, or High. Call capture_lead FIRST after collecting these details. Only after capture_lead succeeds may you call create_support_ticket."

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
api_key = os.getenv("GEMINI_API_KEY", "").strip().strip("\"'").strip()


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


@tool
def capture_lead(
    name: str,
    email: str,
    company: str = None,
    phone: str = None,
    issue_summary: str = None,
    urgency: str = "medium",
) -> str:
    """Save customer contact and qualification details before ticket creation."""
    return save_lead(
        name=name,
        email=email,
        company=company,
        phone=phone,
        issue_summary=issue_summary,
        urgency=urgency,
    )


@tool
def create_support_ticket(user_name: str, issue_description: str, email: str, priority: str = "medium") -> str:
    """Create an open support ticket for human follow-up."""
    ticket_id = save_ticket(user_name, issue_description, priority, email)
    print(json.dumps({
        "ticket_id": ticket_id,
        "reporter": user_name,
        "email": email,
        "issue": issue_description,
        "priority": priority,
        "status": "open",
    }, sort_keys=True), flush=True)
    return f"Support ticket {ticket_id} created successfully."


llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    google_api_key=api_key or None,
    temperature=0.3,
    streaming=True,
)
chat_model = llm.bind_tools([capture_lead, create_support_ticket])

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


async def stream_response(message: str, session_id: str) -> AsyncIterator[str]:
    """Stream assistant text while the agent handles any ticket tool call."""
    context = retrieve_context(message)
    user_prompt = f"Context:\n{context}\n\nQuestion: {message}"
    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": user_prompt}]},
        config={"configurable": {"thread_id": session_id}},
        version="v2",
    ):
        if event["event"] != "on_chat_model_stream":
            continue
        content = getattr(event["data"].get("chunk"), "content", "")
        if isinstance(content, str) and content:
            yield content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    yield block["text"]
