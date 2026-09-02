"""SQLite persistence helpers for support ticket escalation."""

from pathlib import Path
import sqlite3
from uuid import uuid4

DATABASE_PATH = Path(__file__).with_name("tickets.db")


def initialize_database() -> None:
    """Create the support ticket table when the API starts."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                reporter TEXT,
                email TEXT,
                issue TEXT,
                priority TEXT,
                status TEXT
            )
            """
        )
        columns = {row[1] for row in connection.execute("PRAGMA table_info(support_tickets)")}
        if "email" not in columns:
            connection.execute("ALTER TABLE support_tickets ADD COLUMN email TEXT")


def save_ticket(user_name: str, issue_description: str, priority: str, email: str) -> str:
    """Persist an open support ticket and return its generated identifier."""
    ticket_id = f"TKT-{uuid4().hex[:10].upper()}"
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO support_tickets (ticket_id, reporter, email, issue, priority, status)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticket_id, user_name, email, issue_description, priority, "open"),
        )
    return ticket_id
