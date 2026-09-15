# AI Support & Lead Capture Chatbot

AI Support & Lead Capture Chatbot is a production-minded customer-support demo for mid-sized businesses. It combines real-time streaming chat, a local retrieval-augmented knowledge base, autonomous lead qualification, and support-ticket escalation in a lightweight full-stack application.

The platform is designed to deliver:

- Real-time customer support with streamed responses.
- Local RAG with deterministic `HashEmbeddings`, requiring zero external embedding API costs.
- Autonomous two-step escalation: capture qualified contact details first, then create a formal support ticket.
- Dual SQLite persistence with Fernet field-level encryption for sensitive customer data.
- A polished floating “Dot” chat widget for client demonstrations.

Together, these provide real-time support, zero external embedding costs, autonomous qualification, and military-grade AES-based field-level protection for stored customer data.

## Architecture & Technology Stack

### Frontend

- Next.js with the App Router.
- React and TypeScript.
- Tailwind CSS for styling.
- Framer Motion for widget and message animations.
- Lucide React for interface icons.
- Server-Sent Events consumed through the browser `ReadableStream` API.

### Backend

- Python 3.10+.
- FastAPI and Uvicorn.
- LangChain and LangGraph.
- Google Gemini Flash through `langchain-google-genai`.
- Configurable model through `GEMINI_MODEL`, recommended as `gemini-2.0-flash`.
- Local deterministic `HashEmbeddings` and a Chroma-backed in-memory retrieval index.
- Local knowledge source: `backend/knowledge/kb.txt`.

### Persistence and Security

The application uses two SQLite databases:

- `backend/leads.db` — qualified customer contact and issue information.
- `backend/tickets.db` — formal support tickets.

Sensitive fields are encrypted individually with `cryptography.fernet.Fernet` before they are written to SQLite. This is field-level encryption at rest: database structure and non-sensitive metadata remain queryable, while customer PII and issue content are stored as ciphertext.

## Project Structure

```text
ai-support-demo/
├── backend/
│   ├── .env.example
│   ├── chain.py              # RAG agent, Gemini model, tools, and streaming
│   ├── db.py                 # SQLite persistence and Fernet encryption
│   ├── main.py               # FastAPI application and SSE endpoint
│   ├── requirements.txt
│   └── knowledge/
│       └── kb.txt            # Local business knowledge base
├── frontend/
│   ├── app/
│   │   └── page.tsx          # Dot widget host page
│   ├── components/
│   │   ├── ChatInput.tsx
│   │   └── ChatMessage.tsx
│   └── package.json
└── README.md
```

## Getting Started from Scratch

### Prerequisites

- Python 3.10 or newer.
- Node.js 18 or newer and npm.
- A Google Gemini API key.

### 1. Clone and enter the repository

```powershell
cd path\to\ai-support-demo
```

### 2. Create the backend virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Configure backend environment variables

Copy the example file:

```powershell
Copy-Item backend\.env.example backend\.env
```

Then edit `backend/.env`:

```dotenv
GEMINI_API_KEY=your-real-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash
DB_ENCRYPTION_KEY=replace-with-a-long-private-secret
```

Keep `backend/.env` private. Never commit it or share the encryption key. If the key is lost, encrypted database fields cannot be decrypted.

### 4. Initialize the databases

From the repository root:

```bash
python backend/db.py
```

Expected output:

```text
Initialized tickets.db: True
Initialized leads.db: True
```

This creates:

```text
backend/leads.db
backend/tickets.db
```

### 5. Start the backend

```bash
uvicorn backend.main:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

The streaming chat endpoint is:

```text
POST http://127.0.0.1:8000/api/chat
```

### 6. Start the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

Click the floating Dot button in the bottom-right corner to open the assistant.

## Security & Encryption Architecture

### Cryptographic specification

Sensitive values are encrypted with `cryptography.fernet.Fernet` before insertion into SQLite. Fernet provides:

- AES-128-CBC encryption.
- PKCS7 padding.
- HMAC-SHA256 authentication and tamper detection.
- A fresh token timestamp and initialization vector for every encryption operation.

The application derives the Fernet key from `DB_ENCRYPTION_KEY` using SHA-256 and URL-safe Base64 encoding:

```python
base64.urlsafe_b64encode(
    hashlib.sha256(secret.encode()).digest()
)
```

This produces the 32-byte key material required by Fernet. The same secret must be available whenever encrypted records are read.

### What is encrypted

In `leads.db`:

- Email address.
- Phone number, when provided.
- Issue summary, when provided.

In `tickets.db`:

- Issue description.
- Email address.

Names, company names, urgency, priority, IDs, and timestamps remain normal SQLite fields so the application can identify records and operate its workflow.

### Threat model and resistance to database exfiltration

If an unauthorized party copies `leads.db` or `tickets.db`, encrypted fields appear as Fernet tokens beginning with values such as `gAAAAAB...`. The database file alone does not provide the private `DB_ENCRYPTION_KEY`, so customer email addresses, phone numbers, and issue summaries cannot be read from the ciphertext.

Fernet authentication also detects modification: altered ciphertext fails integrity validation instead of silently decrypting to manipulated data.

The design does not claim that database encryption protects a compromised running server. An attacker who can read the running process environment or application memory may be able to obtain the key. Protect the host, environment files, deployment secrets, backups, and logs accordingly.

### SQL injection and filesystem safeguards

All database writes use parameterized SQLite statements with `?` placeholders. User-provided values are never interpolated into SQL strings.

Database files, WAL files, environment files, virtual environments, and local build artifacts are excluded through `.gitignore`. Do not manually override those protections when packaging or sharing the project.

## Owner/Admin Database Decryption Tool

The following owner-only utility opens the two databases and decrypts sensitive fields using the configured `DB_ENCRYPTION_KEY`. Save it as `backend/view_db.py` if you want a reusable admin command.

```python
"""Print decrypted leads and tickets for an authorized administrator."""

import sqlite3

from dotenv import load_dotenv
from tabulate import tabulate

from backend.db import (
    LEADS_DB_PATH,
    TICKETS_DB_PATH,
    decrypt_val,
)


def safe_decrypt(value: str | None) -> str:
    """Decrypt a nullable field while keeping the table readable."""
    if not value:
        return ""
    try:
        return decrypt_val(value)
    except Exception:
        return "[UNDECRYPTABLE]"


def show_leads() -> None:
    with sqlite3.connect(LEADS_DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT id, name, email, phone, company,
                   issue_summary, urgency, created_at
            FROM leads
            ORDER BY created_at DESC
            """
        ).fetchall()

    decrypted = [
        [
            lead_id,
            name,
            safe_decrypt(email),
            safe_decrypt(phone),
            company or "",
            safe_decrypt(issue_summary),
            urgency,
            created_at,
        ]
        for lead_id, name, email, phone, company, issue_summary, urgency, created_at in rows
    ]
    print("\nLEADS")
    print(tabulate(
        decrypted,
        headers=["ID", "Name", "Email", "Phone", "Company", "Issue", "Urgency", "Created"],
        tablefmt="simple",
    ))


def show_tickets() -> None:
    with sqlite3.connect(TICKETS_DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT id, user_name, issue_description, priority, email, created_at
            FROM tickets
            ORDER BY created_at DESC
            """
        ).fetchall()

    decrypted = [
        [
            ticket_id,
            user_name,
            safe_decrypt(issue_description),
            priority,
            safe_decrypt(email),
            created_at,
        ]
        for ticket_id, user_name, issue_description, priority, email, created_at in rows
    ]
    print("\nTICKETS")
    print(tabulate(
        decrypted,
        headers=["ID", "User", "Issue", "Priority", "Email", "Created"],
        tablefmt="simple",
    ))


if __name__ == "__main__":
    load_dotenv(dotenv_path="backend/.env")
    show_leads()
    show_tickets()
```

Install `tabulate` through `backend/requirements.txt`, then run from the repository root:

```bash
python -m backend.view_db
```

Alternatively, after saving the file, run it directly:

```bash
python backend/view_db.py
```

This tool is intended for authorized owners or administrators only. Do not expose it through the API or commit decrypted exports.

## Autonomous AI Agent Workflow

```text
User asks a complex question or requests human support
                │
                ▼
Dot searches the local knowledge base and attempts to help
                │
                ▼
If escalation is needed, Gemini collects:
Name · Email · Company/Team · Phone · Issue Summary · Urgency
                │
                ▼
capture_lead(...)
                │
                ▼
Encrypted contact and qualification data saved to leads.db
                │
                ▼
create_support_ticket(...)
                │
                ▼
Encrypted issue and email saved to tickets.db
                │
                ▼
Dot streams the ticket confirmation and ticket ID to the user
```

The order is intentional. The agent must successfully capture the lead before filing the formal ticket, ensuring support staff have contact and qualification context for follow-up.

## Operational Notes

- Keep `DB_ENCRYPTION_KEY` stable. Changing it prevents existing encrypted fields from being decrypted.
- Back up `leads.db`, `tickets.db`, and the encryption secret through separate secure channels.
- Do not commit `.env`, database files, database WAL files, or decrypted exports.
- Replace the local `backend/knowledge/kb.txt` with client-specific approved business content before a production demo.
- Use HTTPS, a secret manager, restricted filesystem permissions, authentication, rate limiting, and structured audit logging before exposing the service beyond localhost.
