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
TICKETS_DB_PATH = BASE_DIR / "tickets.db"
LEADS_DB_PATH = BASE_DIR / "leads.db"
DATABASE_PATH = TICKETS_DB_PATH
ENV_PATH = Path(__file__).resolve().parent / ".env"

load_dotenv(dotenv_path=ENV_PATH)
load_dotenv()

logger = logging.getLogger(__name__)


def _derive_fernet_key(secret: str) -> bytes:
    """Derive a stable Fernet-compatible key from a configured secret."""
    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())


_encryption_secret = os.getenv("DB_ENCRYPTION_KEY", "dot-local-development-encryption-key").strip()
_fernet = Fernet(_derive_fernet_key(_encryption_secret))


def encrypt_val(text: str) -> str:
    """Encrypt a sensitive text value for storage in SQLite."""
    return _fernet.encrypt(text.encode("utf-8")).decode("utf-8")


def decrypt_val(cipher_text: str) -> str:
    """Decrypt a Fernet-encrypted text value from SQLite."""
    return _fernet.decrypt(cipher_text.encode("utf-8")).decode("utf-8")


def get_connection(db_path: str | Path = TICKETS_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection configured for write-ahead logging."""
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path))
    try:
        connection.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError as exc:
        logger.warning(
            "WAL mode unavailable for %s; continuing with SQLite default journal mode: %s",
            database_path,
            exc,
        )
    return connection


def _restrict_permissions(database_path: Path) -> None:
    """Restrict database files to the application user where supported."""
    try:
        os.chmod(database_path, 0o600)
    except (OSError, NotImplementedError) as exc:
        logger.warning("Could not restrict permissions on %s: %s", database_path, exc)


def initialize_database() -> None:
    """Create both databases and their tables on startup."""
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
        connection.commit()
    _restrict_permissions(LEADS_DB_PATH)

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
        connection.commit()
    _restrict_permissions(TICKETS_DB_PATH)


def save_ticket(user_name: str, issue_description: str, priority: str, email: str) -> str:
    """Persist an open support ticket and return its generated identifier."""
    ticket_id = f"TKT-{uuid4().hex[:10].upper()}"
    with get_connection(TICKETS_DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO tickets (id, user_name, issue_description, priority, email)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ticket_id, user_name, encrypt_val(issue_description), priority, encrypt_val(email)),
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
        connection.commit()
    return lead_id


if __name__ == "__main__":
    initialize_database()
    print("Initialized tickets.db:", TICKETS_DB_PATH.exists())
    print("Initialized leads.db:", LEADS_DB_PATH.exists())
