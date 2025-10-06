from typing import Optional, Dict, Any
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
