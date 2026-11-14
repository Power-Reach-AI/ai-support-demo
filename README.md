# Northstar AI Chatbot Demo (Dot)

Customer-support and lead-qualification AI chatbot built with a **FastAPI + LangChain / LangGraph** backend and a **Next.js (App Router) + Tailwind CSS** frontend.

The backend retrieves local knowledge with deterministic embeddings, streams Anthropic Claude responses over Server-Sent Events (SSE), persists encrypted leads/tickets in SQLite, enforces 3-layer code guardrails, and escalates anomalous sessions to a human via webhook notifications and session locks.

---

## Architecture Overview

```text
Next.js Frontend (frontend/app/page.tsx)
   │  POST /api/chat (SSE stream)
   ▼
FastAPI Entrypoint (backend/main.py)
   ├─► Session Lock Check (db.is_session_locked) ──► Immediate HITL message (no LLM call)
   ├─► Layer 1 Input Guardrails (guardrails.check_input)
   │      └─► Blocked? ──► Log encrypted anomaly (db.save_anomaly)
   │                     └─► Lock session + Webhook alert (_notify_human)
   ▼
RAG & Tool Agent (backend/chain.py)
   ├─► Local RAG: TextLoader (kb.txt) -> Chroma + HashEmbeddings (top 2 chunks)
   ├─► LangGraph Agent: ChatAnthropic + InMemorySaver (session_id)
   ├─► Layer 2 Tool Guardrails:
   │      ├─► capture_lead() ──► validate_lead_input() ──► leads.db (encrypted)
   │      └─► create_support_ticket() ──► requires capture_lead + max ticket cap ──► tickets.db (encrypted)
   └─► Layer 3 Output Guardrails (guardrails.check_output)
          └─► Buffers & scans response before streaming SSE chunks
```

---

## Project Tree

```text
backend/
  knowledge/
    kb.txt               # Local knowledge base indexed into Chroma
  .env.example           # Environment variable template
  __init__.py
  chain.py               # RAG retriever, LangGraph agent, tools, and output buffering
  db.py                  # SQLite persistence (WAL mode) + Fernet field encryption
  guardrails.py          # 3-layer guardrails, rate limiter, anomaly escalation, webhook alerts
  main.py                # FastAPI app (/health, /api/chat, /admin/unlock/{session_id})
  requirements.txt
  test_guardrails.py     # 18 automated guardrail and encryption tests
frontend/
  .env.example
  app/
    globals.css
    layout.tsx           # RootLayout & metadata
    page.tsx             # Chat UI & sendMessage() SSE handler
  components/
    ChatInput.tsx        # Message input component
    ChatMessage.tsx      # Message bubble & Message type
  next.config.mjs
  next-env.d.ts
  package.json           # Next.js, React, Tailwind CSS, framer-motion, lucide-react
  postcss.config.mjs
  tailwind.config.ts
  tsconfig.json
```

---

## Installation & Running Locally

### 1. Backend (FastAPI)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
```

Edit `backend\.env` and set your Anthropic key (and optional settings below), then start the API server:

```powershell
uvicorn backend.main:app --reload --reload-dir backend --reload-include "*.py" --reload-include "*.env" --host 0.0.0.0 --port 8000
```

*(Including `--reload-include "*.env"` ensures `uvicorn` automatically reloads whenever you toggle flags in `backend/.env`.)*

* Health check: `http://localhost:8000/health` (`{"status": "ok"}`)
* Interactive API docs: `http://localhost:8000/docs`

### 2. Frontend (Next.js)

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open **`http://localhost:3000`** in your browser.

---

## Environment Variables (`backend/.env`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key used by `ChatAnthropic` and checked on `/api/chat`. |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Claude model ID used in `backend/chain.py`. |
| `DB_ENCRYPTION_KEY` | `dot-local-development-encryption-key` | Secret used to derive the 32-byte Fernet encryption key for SQLite. **Change in production.** |
| `GUARDRAILS_ENABLED` | `true` | Master switch for code-enforced guardrails (`true` = enforced, `false` = prompt-only demo mode). |
| `ALLOW_DEMO_TOGGLE` | `false` | When `true`, honors an optional `guardrails_enabled` boolean on `ChatRequest`. Keep `false` in production. |
| `ANOMALY_WEBHOOK_URL` | *(unset)* | Optional HTTP(S) webhook URL (e.g., Google Chat or Slack) that receives `{"text": "..."}` alerts on session lock. |
| `GUARDRAIL_MAX_MESSAGE_CHARS` | `1500` | Maximum allowed characters per user message. |
| `GUARDRAIL_MAX_HISTORY_TURNS` | `20` | Maximum allowed conversation history turns in `ChatRequest.history`. |
| `GUARDRAIL_MAX_HISTORY_CHARS` | `8000` | Maximum total characters across `ChatRequest.history`. |
| `GUARDRAIL_RATE_LIMIT_REQUESTS` | `20` | Max requests per `session_id` within the sliding window. |
| `GUARDRAIL_RATE_LIMIT_CHARS` | `30000` | Max characters per `session_id` within the sliding window. |
| `GUARDRAIL_RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding-window duration in seconds for rate limiting. |
| `GUARDRAIL_TOPIC_SIMILARITY_THRESHOLD` | `0.20` | Minimum `HashEmbeddings` cosine similarity required when intent regex does not already match. |
| `GUARDRAIL_MAX_TICKETS_PER_SESSION` | `2` | Maximum support tickets allowed per `session_id`. |

---

## How Guardrails Work (`backend/guardrails.py` & `backend/chain.py`)

Guardrails are enforced deterministically in Python code so the LLM cannot overlook or talk around them:

1. **Layer 1 — Input Guardrails (`check_input` in `main.py` before calling Claude):**
   * **Payload & Token-Drain Protection:** Rejects empty messages, messages exceeding `GUARDRAIL_MAX_MESSAGE_CHARS`, oversized `history`, and repetitive low-uniqueness payloads (`_looks_like_token_drain`).
   * **Rate Limiting:** Thread-safe per-`session_id` sliding-window limiter (`SlidingWindowRateLimiter`) capping both request count and character volume.
   * **Prompt-Injection & Payload Detection (`detect_prompt_injection`):** Normalizes NFKC Unicode, Cyrillic/confusable homoglyphs, HTML entities, URL encoding, zero-width characters, leetspeak, Base64, and ROT13 before matching instruction-override, system-prompt-extraction, jailbreak, XSS (`<script>`), and SQL-injection patterns.
   * **Topic / Scope Gate (`check_topic_scope`):** Allows greetings, follow-ups, and business/product/pricing/support/lead intents; explicitly blocks off-topic categories (general coding, creative writing, trivia, weather/sports, food/travel, finance trivia, entertainment, medical/legal/tax advice, homework, translation, resume writing); and falls back to `HashEmbeddings` cosine similarity against `backend/knowledge/kb.txt`.

2. **Layer 2 — Execution & Tool Guardrails (`chain.py`):**
   * **Strict Input Validation (`validate_lead_input`, `validate_ticket_input`):** Validates email syntax, phone length/characters (7–15 digits), `urgency`/`priority` (`low`, `medium`, `high`), and field length limits before any DB write.
   * **Lead-Before-Ticket Precondition:** `create_support_ticket_for_session()` checks SQLite (`has_captured_lead(session_id)`) and rejects ticket creation unless `capture_lead()` has already succeeded for that session. Calling `create_support_ticket` without lead capture records a `tool_bypass` anomaly.
   * **Ticket Cap:** Enforces `GUARDRAIL_MAX_TICKETS_PER_SESSION` (default `2`) per session.

3. **Layer 3 — Output Guardrails (`check_output` in `chain.py`):**
   * In guarded mode, `stream_response()` buffers the model's generated text and runs `check_output()` before yielding SSE chunks to the browser.
   * Blocks responses containing system-prompt markers, internal file paths/tracebacks, secret/API-key patterns (`sk-ant-`, `sk-`, `AKIA`, private keys, env variable names, `Bearer` tokens), or off-topic drift (fenced code blocks, creative story/poem openings, generic AI disclaimers), replacing them with a safe fallback message.

---

## Toggling Guardrails (On / Off)

### Turning Guardrails Off (Broken / Prompt-Only Demo Mode)
Set `GUARDRAILS_ENABLED=false` in `backend/.env` (or, if `ALLOW_DEMO_TOGGLE=true`, pass `"guardrails_enabled": false` in the `POST /api/chat` JSON body).

When disabled:
* Layer 1 input checks and Layer 3 output buffering are skipped (raw token-by-token streaming).
* Layer 2 skips deterministic lead-precondition and input-validation checks, allowing prompt-only behavior to be demonstrated.

### Turning Guardrails On (Enforced Mode)
Set `GUARDRAILS_ENABLED=true` in `backend/.env` (default) and keep `ALLOW_DEMO_TOGGLE=false` in production so callers cannot override enforcement via request JSON.

Every `/api/chat` response returns an `X-Guardrails-Enabled` (`true` / `false`) response header indicating the active mode.

---

## Human-in-the-Loop (HITL) Escalation & Notifications

When Layer 1 or Layer 2 blocks an anomalous request, `handle_guardrail_block()` in `backend/guardrails.py` handles escalation:

1. **Anomaly Logging:** Saves an encrypted record to the `anomalies` table in `backend/tickets.db`.
2. **Escalation Rules:**
   * **Immediate lock:** `prompt_injection` and `tool_bypass` violations lock the session immediately.
   * **Threshold lock:** `off_topic` and `rate_limit` violations lock the session once `>= 2` open anomalies accumulate for that `session_id`.
3. **Human Notification (`_notify_human`):** On a new lock, a background daemon thread sends a `POST` request with `{"text": "..."}` (containing `session_id`, `anomaly_type`, `rule`, `anomaly_id`, and excerpt) to `ANOMALY_WEBHOOK_URL` (compatible with Google Chat and Slack incoming webhooks). If `ANOMALY_WEBHOOK_URL` is unset, a warning is logged to stdout/stderr.
4. **Session Lock Behavior:** While `is_session_locked(session_id)` is true, `POST /api/chat` skips the LLM completely and immediately streams `SESSION_LOCKED_MESSAGE`.
5. **Human Unlock:** After reviewing the incident, unlock the session and mark its open anomalies `resolved`:
   ```powershell
   Invoke-RestMethod -Method Post "http://localhost:8000/admin/unlock/<session_id>"
   ```

---

## Database Persistence & Encryption (`backend/db.py`)

`initialize_database()` runs on FastAPI startup (`lifespan`) and creates two local SQLite databases configured with `PRAGMA journal_mode = WAL` and restricted file permissions (`0o600`):

* **`backend/tickets.db`**
  * `tickets`: `id`, `user_name`, `issue_description` *(encrypted)*, `priority`, `email` *(encrypted)*, `session_id`, `created_at`
  * `anomalies`: `id`, `session_id`, `anomaly_type`, `rule_triggered`, `excerpt` *(encrypted)*, `status` (`open` / `resolved`), `created_at`
  * `session_locks`: `session_id`, `locked_at`
* **`backend/leads.db`**
  * `leads`: `id`, `name`, `email` *(encrypted)*, `phone` *(encrypted)*, `company`, `issue_summary` *(encrypted)*, `urgency`, `created_at`
  * `guardrail_sessions`: `session_id`, `lead_captured`, `lead_id`, `lead_captured_at`, `updated_at`

### How Encryption Works
1. `_derive_fernet_key()` hashes `DB_ENCRYPTION_KEY` using **SHA-256** and URL-safe Base64-encodes the 32-byte digest into a Fernet key.
2. `encrypt_val()` encrypts sensitive text fields via `cryptography.fernet.Fernet` before inserting into SQLite.
3. `decrypt_val()` decrypts stored Fernet tokens when authorized inspection is needed.

---

## API Endpoints (`backend/main.py`)

| Method & Path | Description |
| :--- | :--- |
| `GET /health` | Lightweight liveness check returning `{"status": "ok"}`. |
| `POST /api/chat` | Accepts `{ "message", "session_id", "history", "guardrails_enabled" }` and streams SSE chunks (`data: ...`, `data: [DONE]`). |
| `POST /admin/unlock/{session_id}` | Removes the session lock and marks open anomalies for that `session_id` as `resolved`. |

---

## Running Tests

Run the 18 automated backend guardrail and encryption tests:

```powershell
.\.venv\Scripts\python.exe -m pytest backend/test_guardrails.py -v
```