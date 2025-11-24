import re
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import HTTPException


async def clean_checkpointer_state(session_id: str, operation: str = "new session") -> bool:
    try:
        from data_scientist_chatbot.app.core.graph_builder import DB_PATH
        import aiosqlite

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
            await db.commit()
            print(f"DEBUG: Cleaned checkpointer state for {operation} {session_id}")
            return True
    except Exception as e:
        print(f"WARNING: Failed to clean checkpointer state for {operation} {session_id}: {e}")
        return False


def get_or_create_agent_session(session_id: str, session_agents: Dict[str, Any], create_enhanced_agent_executor):
    import logging
    import asyncio

    if session_id not in session_agents:
        logging.info(f"[SESSION] Creating new agent for session: {session_id}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                logging.warning(f"[SESSION] Event loop running, skipping async checkpoint cleanup for {session_id}")
            else:
                logging.info(f"[SESSION] Cleaning checkpointer state for {session_id}")
                loop.run_until_complete(clean_checkpointer_state(session_id))
        except Exception as e:
            logging.warning(f"[SESSION] Could not clean checkpointer state: {e}")

        session_agents[session_id] = create_enhanced_agent_executor(session_id)
        logging.info(f"[SESSION] Agent created for session: {session_id}")
    else:
        logging.debug(f"[SESSION] Reusing existing agent for session: {session_id}")

    return session_agents[session_id]


def validate_session(session_id: str, session_store: Dict[str, Any]) -> Dict[str, Any]:
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    return session_store[session_id]


def validate_session_id(session_id: str) -> bool:
    if not session_id or not isinstance(session_id, str):
        return False

    if len(session_id) > 128:
        return False

    pattern = r"^[a-zA-Z0-9_-]+$"
    if not re.match(pattern, session_id):
        return False

    dangerous_patterns = ["..", "/", "\\", "<", ">", "DROP", "UNION", "script", "SELECT", "INSERT", "DELETE", "UPDATE"]
    return not any(pattern.lower() in session_id.lower() for pattern in dangerous_patterns)


def create_new_session() -> Dict[str, str]:
    import builtins

    new_session_id = str(uuid.uuid4())
    created_at = datetime.now()

    if not hasattr(builtins, "_session_store"):
        builtins._session_store = {}

    builtins._session_store[new_session_id] = {
        "session_id": new_session_id,
        "created_at": created_at,
        "thinking_mode": False,
        "web_search_enabled": False,
    }

    clean_checkpointer_state(new_session_id, "new session")

    return {"session_id": new_session_id}


def clear_session(session_id: str) -> bool:
    import builtins

    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
        del builtins._session_store[session_id]

    if hasattr(builtins, "_persistent_sandboxes") and session_id in builtins._persistent_sandboxes:
        try:
            sandbox = builtins._persistent_sandboxes[session_id]
            sandbox.close()
            del builtins._persistent_sandboxes[session_id]
        except Exception:
            pass

    clean_checkpointer_state(session_id, "clear session")

    return True
