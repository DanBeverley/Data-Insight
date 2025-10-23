import re
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import HTTPException


def clean_checkpointer_state(session_id: str, operation: str = "new session") -> bool:
    try:
        from data_scientist_chatbot.app.core.graph_builder import memory

        if memory:
            memory.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
            memory.conn.commit()
            print(f"DEBUG: Cleaned checkpointer state for {operation} {session_id}")
            return True
        return False
    except Exception as e:
        print(f"WARNING: Failed to clean checkpointer state for {operation} {session_id}: {e}")
        return False


def get_or_create_agent_session(session_id: str, session_agents: Dict[str, Any], create_enhanced_agent_executor):
    if session_id not in session_agents:
        clean_checkpointer_state(session_id)
        session_agents[session_id] = create_enhanced_agent_executor(session_id)
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

    builtins._session_store[new_session_id] = {"session_id": new_session_id, "created_at": created_at}

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
