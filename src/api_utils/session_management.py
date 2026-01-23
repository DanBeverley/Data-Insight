import re
import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import HTTPException
import logging


# Singleton for in-memory session storage
class SessionDataManager:
    _instance = None
    _store = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionDataManager, cls).__new__(cls)
            # Initialize builtins._session_store if not present
            import builtins

            if not hasattr(builtins, "_session_store"):
                builtins._session_store = {}
        return cls._instance

    @property
    def store(self):
        import builtins

        if not hasattr(builtins, "_session_store"):
            builtins._session_store = {}
        return builtins._session_store

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.store.get(session_id)

    def create_session(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.store:
            self.store[session_id] = {
                "session_id": session_id,
                "created_at": datetime.now(),
                "thinking_mode": False,
                "web_search_enabled": False,
            }
        return self.store[session_id]

    def delete_session(self, session_id: str):
        if session_id in self.store:
            del self.store[session_id]

    def set_session(self, session_id: str, data: Dict[str, Any]):
        """Set or update session data. Creates session if it doesn't exist."""
        if session_id not in self.store:
            self.create_session(session_id)
        self.store[session_id].update(data)


session_data_manager = SessionDataManager()


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


def clear_transient_agent_state(session_id: str, reason: str = "new data loaded") -> bool:
    """Clear transient execution state (artifacts, insights) while preserving conversation.

    Called when new data is loaded to prevent stale analysis from polluting new requests.
    """
    try:
        from data_scientist_chatbot.app.core.graph_builder import get_checkpointer
        import asyncio

        checkpointer = get_checkpointer()
        if not checkpointer:
            logging.debug(f"[SESSION] No checkpointer available for {session_id}")
            return True

        config = {"configurable": {"thread_id": session_id}}

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                current_state = asyncio.run_coroutine_threadsafe(checkpointer.aget(config), loop).result(timeout=2)
            else:
                current_state = loop.run_until_complete(checkpointer.aget(config))
        except Exception:
            current_state = None

        if current_state and hasattr(current_state, "values"):
            state_values = current_state.values
            cleared_fields = []

            if state_values.get("artifacts"):
                state_values["artifacts"] = []
                cleared_fields.append("artifacts")
            if state_values.get("agent_insights"):
                state_values["agent_insights"] = []
                cleared_fields.append("insights")
            if state_values.get("execution_result"):
                state_values["execution_result"] = None
                cleared_fields.append("execution_result")

            if cleared_fields:
                logging.info(f"[SESSION] Cleared transient state for {session_id} ({reason}): {cleared_fields}")
                return True

        logging.debug(f"[SESSION] No transient state to clear for {session_id}")
        return True

    except Exception as e:
        logging.warning(f"[SESSION] Could not clear transient state for {session_id}: {e}")
        return False


def get_or_create_agent_session(session_id: str, session_agents: Dict[str, Any], create_enhanced_agent_executor):
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
    new_session_id = str(uuid.uuid4())

    session_data_manager.create_session(new_session_id)
    clean_checkpointer_state(new_session_id, "new session")

    return {"session_id": new_session_id}


def clear_session(session_id: str) -> bool:
    session_data_manager.delete_session(session_id)

    # Handle sandbox cleanup if needed (requires access to sandbox store)
    # For now, we focus on the session data

    clean_checkpointer_state(session_id, "clear session")

    return True
