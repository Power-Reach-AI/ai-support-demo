"""SQLite persistence with Fernet encryption for sensitive field values."""

import base64
import hashlib
import logging
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from cryptography.fernet import Fernet
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TICKETS_DB_PATH = BASE_DIR / "tickets.db"
LEADS_DB_PATH = BASE_DIR / "leads.db"
DATABASE_PATH = TICKETS_DB_PATH
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()

logger = logging.getLogger(__name__)


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a stable Fernet-compatible key from the configured secret."""
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())


_encryption_secret = os.getenv("DB_ENCRYPTION_KEY", "dot-local-development-encryption-key").strip()
_fernet = Fernet(_derive_fernet_key(_encryption_secret))


def encrypt_val(text: str) -> str:
    """Encrypt a sensitive text value for storage in SQLite."""
    return _fernet.encrypt(text.encode("utf-8")).decode("utf-8")


def decrypt_val(cipher_text: str) -> str:
    """Decrypt a Fernet-encrypted text value from SQLite."""
    return _fernet.decrypt(cipher_text.encode("utf-8")).decode("utf-8")


def _resolve_database_path(db_path: str | Path) -> Path:
    path = Path(db_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _restrict_permissions(database_path: Path) -> None:
    try:
        os.chmod(database_path, 0o600)
    except (OSError, NotImplementedError) as exc:
        logger.warning("Could not restrict permissions on %s: %s", database_path, exc)


def get_connection(db_path: str | Path = TICKETS_DB_PATH) -> sqlite3.Connection:
    """Open a standard SQLite connection configured for write-ahead logging."""
    database_path = _resolve_database_path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path))
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def _ensure_guardrail_sessions_table(connection: sqlite3.Connection) -> None:
    """Create the per-session tool guardrail state table if it is missing."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS guardrail_sessions (
            session_id TEXT PRIMARY KEY,
            lead_captured INTEGER NOT NULL DEFAULT 0,
            lead_id TEXT,
            lead_captured_at TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _ensure_anomaly_tables(connection: sqlite3.Connection) -> None:
    """Create the anomaly audit trail and session-lock tables if missing."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            anomaly_type TEXT NOT NULL,
            rule_triggered TEXT NOT NULL,
            excerpt TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_anomalies_session_status "
        "ON anomalies(session_id, status)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS session_locks (
            session_id TEXT PRIMARY KEY,
            locked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def initialize_database() -> None:
    """Create or migrate the tickets and leads databases on API startup."""
    with get_connection(TICKETS_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id TEXT PRIMARY KEY,
                user_name TEXT NOT NULL,
                issue_description TEXT NOT NULL,
                priority TEXT NOT NULL,
                email TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        ticket_columns = {row[1] for row in connection.execute("PRAGMA table_info(tickets)")}
        if "session_id" not in ticket_columns:
            connection.execute("ALTER TABLE tickets ADD COLUMN session_id TEXT")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_tickets_session_id ON tickets(session_id)"
        )
        _ensure_anomaly_tables(connection)
        connection.commit()
    _restrict_permissions(TICKETS_DB_PATH)

    with get_connection(LEADS_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT,
                company TEXT,
                issue_summary TEXT,
                urgency TEXT DEFAULT 'medium',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(leads)")}
        migrations = {
            "company": "ALTER TABLE leads ADD COLUMN company TEXT",
            "issue_summary": "ALTER TABLE leads ADD COLUMN issue_summary TEXT",
            "urgency": "ALTER TABLE leads ADD COLUMN urgency TEXT DEFAULT 'medium'",
        }
        for column, statement in migrations.items():
            if column not in existing_columns:
                connection.execute(statement)
        _ensure_guardrail_sessions_table(connection)
        connection.commit()
    _restrict_permissions(LEADS_DB_PATH)


def save_ticket(
    user_name: str,
    issue_description: str,
    priority: str,
    email: str,
    session_id: str | None = None,
) -> str:
    """Persist an open support ticket and return its generated identifier."""
    ticket_id = f"TKT-{uuid4().hex[:10].upper()}"
    with get_connection(TICKETS_DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO tickets (id, user_name, issue_description, priority, email, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                user_name,
                encrypt_val(issue_description),
                priority,
                encrypt_val(email),
                session_id,
            ),
        )
        connection.commit()
    return ticket_id


def save_lead(
    name: str,
    email: str,
    phone: str = None,
    company: str = None,
    issue_summary: str = None,
    urgency: str = "medium",
    session_id: str | None = None,
) -> str:
    """Persist encrypted lead contact and qualification details."""
    lead_id = f"LEAD-{uuid4().hex[:8].upper()}"
    with get_connection(LEADS_DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO leads (id, name, email, phone, company, issue_summary, urgency)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lead_id,
                name,
                encrypt_val(email),
                encrypt_val(phone) if phone else None,
                company,
                encrypt_val(issue_summary) if issue_summary else None,
                urgency,
            ),
        )
        if session_id:
            _ensure_guardrail_sessions_table(connection)
            connection.execute(
                """
                INSERT INTO guardrail_sessions
                    (session_id, lead_captured, lead_id, lead_captured_at, updated_at)
                VALUES (?, 1, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    lead_captured = 1,
                    lead_id = excluded.lead_id,
                    lead_captured_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, lead_id),
            )
        connection.commit()
    return lead_id


def record_lead_capture(session_id: str, lead_id: str) -> None:
    """Mark a session as having successfully captured a lead."""
    if not session_id:
        return
    with get_connection(LEADS_DB_PATH) as connection:
        _ensure_guardrail_sessions_table(connection)
        connection.execute(
            """
            INSERT INTO guardrail_sessions
                (session_id, lead_captured, lead_id, lead_captured_at, updated_at)
            VALUES (?, 1, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                lead_captured = 1,
                lead_id = excluded.lead_id,
                lead_captured_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, lead_id),
        )
        connection.commit()


def has_captured_lead(session_id: str) -> bool:
    """Return whether capture_lead has succeeded for this session."""
    if not session_id:
        return False
    with get_connection(LEADS_DB_PATH) as connection:
        _ensure_guardrail_sessions_table(connection)
        row = connection.execute(
            "SELECT lead_captured FROM guardrail_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return bool(row and row[0])


def get_guardrail_session(session_id: str) -> dict:
    """Return the persisted guardrail state for a session."""
    if not session_id:
        return {"session_id": session_id, "lead_captured": False, "lead_id": None}
    with get_connection(LEADS_DB_PATH) as connection:
        _ensure_guardrail_sessions_table(connection)
        row = connection.execute(
            "SELECT session_id, lead_captured, lead_id, lead_captured_at, updated_at "
            "FROM guardrail_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return {"session_id": session_id, "lead_captured": False, "lead_id": None}
    return {
        "session_id": row[0],
        "lead_captured": bool(row[1]),
        "lead_id": row[2],
        "lead_captured_at": row[3],
        "updated_at": row[4],
    }


def count_session_tickets(session_id: str) -> int:
    """Return how many tickets this session has already created."""
    if not session_id:
        return 0
    with get_connection(TICKETS_DB_PATH) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM tickets WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return int(row[0]) if row else 0


def save_anomaly(
    session_id: str,
    anomaly_type: str,
    rule_triggered: str,
    excerpt: str = "",
    status: str = "open",
) -> int:
    """Persist an anomaly with an encrypted excerpt and return its identifier."""
    clean_excerpt = str(excerpt or "").strip()[:500]
    with get_connection(TICKETS_DB_PATH) as connection:
        _ensure_anomaly_tables(connection)
        cursor = connection.execute(
            """
            INSERT INTO anomalies
                (session_id, anomaly_type, rule_triggered, excerpt, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                anomaly_type,
                rule_triggered,
                encrypt_val(clean_excerpt) if clean_excerpt else None,
                status,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def count_session_anomalies(
    session_id: str,
    anomaly_types: tuple[str, ...] | None = None,
) -> int:
    """Return the number of unresolved anomalies for a session."""
    if not session_id:
        return 0
    query = "SELECT COUNT(*) FROM anomalies WHERE session_id = ? AND status = 'open'"
    params: list[str] = [session_id]
    if anomaly_types:
        placeholders = ", ".join("?" for _ in anomaly_types)
        query += f" AND anomaly_type IN ({placeholders})"
        params.extend(anomaly_types)
    with get_connection(TICKETS_DB_PATH) as connection:
        _ensure_anomaly_tables(connection)
        row = connection.execute(query, params).fetchone()
    return int(row[0]) if row else 0


def is_session_locked(session_id: str) -> bool:
    """Return whether a session has been escalated and locked for HITL review."""
    if not session_id:
        return False
    with get_connection(TICKETS_DB_PATH) as connection:
        _ensure_anomaly_tables(connection)
        row = connection.execute(
            "SELECT 1 FROM session_locks WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return row is not None


def lock_session(session_id: str) -> bool:
    """Lock a session and return whether this call created the lock."""
    if not session_id:
        return False
    with get_connection(TICKETS_DB_PATH) as connection:
        _ensure_anomaly_tables(connection)
        cursor = connection.execute(
            "INSERT OR IGNORE INTO session_locks (session_id) VALUES (?)",
            (session_id,),
        )
        connection.commit()
        return cursor.rowcount > 0


def unlock_session(session_id: str) -> bool:
    """Unlock a reviewed session and resolve its open anomalies."""
    if not session_id:
        return False
    with get_connection(TICKETS_DB_PATH) as connection:
        _ensure_anomaly_tables(connection)
        cursor = connection.execute(
            "DELETE FROM session_locks WHERE session_id = ?",
            (session_id,),
        )
        connection.execute(
            "UPDATE anomalies SET status = 'resolved' "
            "WHERE session_id = ? AND status = 'open'",
            (session_id,),
        )
        connection.commit()
        return cursor.rowcount > 0


if __name__ == "__main__":
    initialize_database()
    print("Initialized tickets.db:", TICKETS_DB_PATH.exists())
    print("Initialized leads.db:", LEADS_DB_PATH.exists())
