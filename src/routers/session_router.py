import uuid
import pickle
from fastapi import APIRouter, HTTPException
from datetime import datetime

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
async def get_sessions():
    from ..api import session_store, agent_sessions, GLOBAL_SESSION_ID

    sessions = []
    for session_id, session_data in session_store.items():
        if session_id == GLOBAL_SESSION_ID:
            continue

        session_title = "New Chat"

        session_titles = getattr(rename_session, 'session_titles', {})
        if session_id in session_titles:
            session_title = session_titles[session_id]
        elif session_id in agent_sessions:
            try:
                agent = agent_sessions[session_id]
                if hasattr(agent, 'state') and agent.state and 'messages' in agent.state:
                    messages = agent.state['messages']
                    if messages:
                        first_user_msg = next((msg for msg in messages if hasattr(msg, 'type') and msg.type == 'human'), None)
                        if first_user_msg:
                            session_title = str(first_user_msg.content)[:50]
            except:
                pass

        sessions.append({
            "id": session_id,
            "title": session_title,
            "created_at": session_data.get("created_at", datetime.now()).isoformat()
        })

    return sorted(sessions, key=lambda x: x["created_at"], reverse=True)


@router.post("/new")
async def create_new_session():
    from ..api import session_store
    from ..api_utils.session_management import clean_checkpointer_state

    new_session_id = str(uuid.uuid4())
    session_store[new_session_id] = {
        "session_id": new_session_id,
        "created_at": datetime.now()
    }

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

    clean_checkpointer_state(session_id, "delete session")

    return {"success": True}


@router.put("/{session_id}/rename")
async def rename_session(session_id: str, request: dict):
    try:
        new_title = request.get("title", "").strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")

        if not hasattr(rename_session, 'session_titles'):
            rename_session.session_titles = {}

        rename_session.session_titles[session_id] = new_title

        return {"success": True, "title": new_title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename session: {str(e)}")


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str):
    from ..api import session_store

    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

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
                if 'channel_values' in checkpoint_data and 'messages' in checkpoint_data['channel_values']:
                    for msg in checkpoint_data['channel_values']['messages']:
                        if hasattr(msg, 'type') and msg.type in ['human', 'ai']:
                            messages.append({
                                "type": msg.type,
                                "content": str(msg.content)[:500],
                                "timestamp": datetime.now().isoformat()
                            })
    except Exception as e:
        print(f"Error loading messages from checkpointer: {e}")

    return messages
