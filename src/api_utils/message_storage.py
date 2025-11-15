"""
Message Storage Service
Explicit message persistence independent of LangGraph checkpoints.
This ensures conversation history is preserved across updates and checkpoint failures.
"""
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DB_PATH = Path("sessions_metadata.db")


def get_message_db():
    """Get database connection and ensure schema exists"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_type TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            metadata TEXT DEFAULT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
    """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_session_messages
        ON conversation_messages(session_id, created_at)
    """
    )

    conn.commit()
    return conn


def save_message(session_id: str, message_type: str, content: str, metadata: Dict[str, Any] = None) -> bool:
    """
    Save a message to explicit storage.

    Args:
        session_id: Session ID
        message_type: 'human' or 'ai'
        content: Message content
        metadata: Optional metadata (plots, models, etc.)

    Returns:
        True if saved successfully
    """
    if not session_id or not message_type or not content:
        logger.warning("Skipping save: missing required fields")
        return False

    try:
        conn = get_message_db()

        import json

        metadata_json = json.dumps(metadata) if metadata else None

        conn.execute(
            """INSERT INTO conversation_messages
               (session_id, message_type, content, created_at, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, message_type, content, datetime.now().isoformat(), metadata_json),
        )
        conn.commit()
        conn.close()

        logger.debug(f"Saved {message_type} message for session {session_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to save message: {e}")
        return False


def load_messages(session_id: str) -> List[Dict[str, Any]]:
    """
    Load all messages for a session from explicit storage.

    Args:
        session_id: Session ID

    Returns:
        List of message dicts with type, content, timestamp, metadata
    """
    try:
        conn = get_message_db()
        cursor = conn.execute(
            """SELECT message_type, content, created_at, metadata
               FROM conversation_messages
               WHERE session_id = ?
               ORDER BY created_at ASC""",
            (session_id,),
        )

        messages = []
        for row in cursor.fetchall():
            message_type, content, created_at, metadata_json = row

            message = {"type": message_type, "content": content, "timestamp": created_at}

            if metadata_json:
                try:
                    import json

                    message["metadata"] = json.loads(metadata_json)
                except:
                    pass

            messages.append(message)

        conn.close()
        logger.info(f"Loaded {len(messages)} messages for session {session_id}")
        return messages

    except Exception as e:
        logger.error(f"Failed to load messages: {e}")
        return []


def delete_session_messages(session_id: str) -> bool:
    """
    Delete all messages for a session.

    Args:
        session_id: Session ID

    Returns:
        True if deleted successfully
    """
    try:
        conn = get_message_db()
        conn.execute("DELETE FROM conversation_messages WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

        logger.info(f"Deleted messages for session {session_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to delete messages: {e}")
        return False


def get_message_count(session_id: str) -> int:
    """Get total message count for a session"""
    try:
        conn = get_message_db()
        cursor = conn.execute("SELECT COUNT(*) FROM conversation_messages WHERE session_id = ?", (session_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.error(f"Failed to get message count: {e}")
        return 0


def cleanup_old_messages(days_old: int = 30) -> int:
    """
    Cleanup messages older than specified days for sessions that no longer exist.

    Args:
        days_old: Delete messages older than this many days

    Returns:
        Number of messages deleted
    """
    try:
        from datetime import timedelta

        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()

        conn = get_message_db()

        cursor = conn.execute(
            """DELETE FROM conversation_messages
               WHERE created_at < ?
               AND session_id NOT IN (SELECT session_id FROM sessions)""",
            (cutoff_date,),
        )

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Cleaned up {deleted_count} old messages")
        return deleted_count

    except Exception as e:
        logger.error(f"Failed to cleanup old messages: {e}")
        return 0
