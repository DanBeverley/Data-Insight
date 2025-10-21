import uuid
import pickle
import sqlite3
import logging
from fastapi import APIRouter, HTTPException
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])

def get_session_db():
    db_path = Path("sessions_metadata.db")
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT,
            last_updated TEXT
        )
    """)
    conn.commit()
    return conn


@router.get("")
async def get_sessions():
    from ..api import session_store, GLOBAL_SESSION_ID

    sessions = []
    db_conn = get_session_db()

    try:
        cursor = db_conn.execute("SELECT session_id, title, created_at, last_updated FROM sessions ORDER BY created_at DESC")
        db_sessions = cursor.fetchall()

        for row in db_sessions:
            session_id, title, created_at, last_updated = row
            if session_id == GLOBAL_SESSION_ID:
                continue

            if session_id not in session_store:
                session_store[session_id] = {
                    "session_id": session_id,
                    "created_at": datetime.fromisoformat(created_at) if created_at else datetime.now()
                }

            sessions.append({
                "id": session_id,
                "title": title or "New Chat",
                "created_at": created_at or datetime.now().isoformat()
            })
    except Exception as e:
        logger.error(f"Failed to load sessions from database: {e}")
    finally:
        db_conn.close()

    return sessions


@router.post("/new")
async def create_new_session():
    from ..api import session_store
    from ..api_utils.session_management import clean_checkpointer_state

    new_session_id = str(uuid.uuid4())
    created_at = datetime.now()

    session_store[new_session_id] = {
        "session_id": new_session_id,
        "created_at": created_at
    }

    db_conn = get_session_db()
    try:
        db_conn.execute(
            "INSERT INTO sessions (session_id, title, created_at, last_updated) VALUES (?, ?, ?, ?)",
            (new_session_id, "New Chat", created_at.isoformat(), created_at.isoformat())
        )
        db_conn.commit()
        logger.info(f"Created new session: {new_session_id}")
    except Exception as e:
        logger.error(f"Failed to persist new session: {e}")
    finally:
        db_conn.close()

    clean_checkpointer_state(new_session_id, "new session")

    return {"session_id": new_session_id}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    from ..api import session_store, agent_sessions
    from ..api_utils.session_management import clean_checkpointer_state

    if session_id in session_store:
        del session_store[session_id]
    if session_id in agent_sessions:
        del agent_sessions[session_id]

    db_conn = get_session_db()
    try:
        db_conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        db_conn.commit()
        logger.info(f"Deleted session: {session_id}")
    except Exception as e:
        logger.error(f"Failed to delete session from database: {e}")
    finally:
        db_conn.close()

    clean_checkpointer_state(session_id, "delete session")

    return {"success": True}


@router.put("/{session_id}/rename")
async def rename_session(session_id: str, request: dict):
    try:
        new_title = request.get("title", "").strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")

        db_conn = get_session_db()
        try:
            db_conn.execute(
                "UPDATE sessions SET title = ?, last_updated = ? WHERE session_id = ?",
                (new_title, datetime.now().isoformat(), session_id)
            )
            db_conn.commit()
            logger.info(f"Renamed session {session_id} to '{new_title}'")
        except Exception as e:
            logger.error(f"Failed to rename session in database: {e}")
            raise
        finally:
            db_conn.close()

        return {"success": True, "title": new_title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename session: {str(e)}")


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str):
    from ..api import session_store

    if session_id not in session_store:
        db_conn = get_session_db()
        try:
            cursor = db_conn.execute(
                "SELECT session_id, created_at FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                session_id_db, created_at = row
                session_store[session_id] = {
                    "session_id": session_id_db,
                    "created_at": datetime.fromisoformat(created_at) if created_at else datetime.now()
                }
                logger.info(f"Restored session {session_id} to session_store from database")
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to check session in database: {e}")
            raise HTTPException(status_code=404, detail="Session not found")
        finally:
            db_conn.close()

    messages = []

    try:
        from data_scientist_chatbot.app.core.graph_builder import memory
        if memory:
            cursor = memory.conn.execute(
                "SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                checkpoint_data = pickle.loads(row[0])
                logger.info(f"Checkpoint data keys: {checkpoint_data.keys()}")
                if 'channel_values' in checkpoint_data and 'messages' in checkpoint_data['channel_values']:
                    msg_count = len(checkpoint_data['channel_values']['messages'])
                    logger.info(f"Found {msg_count} messages in checkpoint for session {session_id}")
                    for msg in checkpoint_data['channel_values']['messages']:
                        if hasattr(msg, 'type') and msg.type in ['human', 'ai']:
                            messages.append({
                                "type": msg.type,
                                "content": str(msg.content),
                                "timestamp": datetime.now().isoformat()
                            })
                    logger.info(f"Returning {len(messages)} messages for session {session_id}")
                else:
                    logger.warning(f"No messages found in checkpoint for session {session_id}")
            else:
                logger.warning(f"No checkpoint found for session {session_id}")
        else:
            logger.warning(f"Memory not initialized")
    except Exception as e:
        logger.error(f"Error loading messages from checkpointer: {e}", exc_info=True)

    return messages
