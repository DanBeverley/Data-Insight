import uuid
import pickle
import sqlite3
import logging
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from pathlib import Path
from ..auth.dependencies import get_current_user, get_optional_current_user
from ..database.models import User
from typing import Optional

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def get_session_db():
    db_path = Path("sessions_metadata.db")
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at TEXT,
            last_updated TEXT
        )
    """
    )

    try:
        conn.execute("SELECT user_id FROM sessions LIMIT 1")
    except sqlite3.OperationalError:
        logger.info("Migrating sessions table to add user_id column")
        conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT DEFAULT 'anonymous'")
        conn.commit()

    conn.commit()
    return conn


@router.get("")
async def get_sessions(current_user: Optional[User] = Depends(get_optional_current_user)):
    from ..api import session_store, GLOBAL_SESSION_ID

    user_id = current_user.id if current_user else "anonymous"
    sessions = []
    db_conn = get_session_db()

    try:
        cursor = db_conn.execute(
            "SELECT session_id, title, created_at, last_updated, user_id FROM sessions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        db_sessions = cursor.fetchall()

        for row in db_sessions:
            session_id, title, created_at, last_updated, user_id = row
            if session_id == GLOBAL_SESSION_ID:
                continue

            if session_id not in session_store:
                from ..api_utils.session_persistence import session_data_store

                persisted_data = session_data_store.load_session_data(session_id)

                if persisted_data:
                    session_store[session_id] = persisted_data
                    logger.info(f"Restored session {session_id} from persistent storage")
                else:
                    session_store[session_id] = {
                        "session_id": session_id,
                        "user_id": user_id,
                        "created_at": datetime.fromisoformat(created_at) if created_at else datetime.now(),
                    }

            sessions.append(
                {"id": session_id, "title": title or "New Chat", "created_at": created_at or datetime.now().isoformat()}
            )
    except Exception as e:
        logger.error(f"Failed to load sessions from database: {e}")
    finally:
        db_conn.close()

    return sessions


@router.post("/new")
async def create_new_session():
    from ..api import session_store

    user_id = "anonymous"
    new_session_id = str(uuid.uuid4())
    created_at = datetime.now()

    session_store[new_session_id] = {"session_id": new_session_id, "user_id": user_id, "created_at": created_at}

    db_conn = get_session_db()
    try:
        db_conn.execute(
            "INSERT INTO sessions (session_id, user_id, title, created_at, last_updated) VALUES (?, ?, ?, ?, ?)",
            (new_session_id, user_id, "New Chat", created_at.isoformat(), created_at.isoformat()),
        )
        db_conn.commit()
        logger.info(f"Created new session: {new_session_id} for user: {user_id}")
    except Exception as e:
        logger.error(f"Failed to persist new session: {e}")
    finally:
        db_conn.close()

    return {"session_id": new_session_id}


@router.delete("/{session_id}")
async def delete_session(session_id: str, current_user: Optional[User] = Depends(get_optional_current_user)):
    user_id = current_user.id if current_user else "anonymous"
    from ..api import session_store, agent_sessions
    from ..api_utils.session_management import clean_checkpointer_state
    from ..api_utils.message_storage import delete_session_messages

    db_conn = get_session_db()
    try:
        cursor = db_conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        if row[0] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this session")
    finally:
        db_conn.close()

    if session_id in session_store:
        del session_store[session_id]
    if session_id in agent_sessions:
        del agent_sessions[session_id]

    delete_session_messages(session_id)

    from ..api_utils.session_persistence import session_data_store

    session_data_store.delete_session_data(session_id)

    db_conn = get_session_db()
    try:
        db_conn.execute("DELETE FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        db_conn.commit()
        logger.info(f"Deleted session: {session_id}")
    except Exception as e:
        logger.error(f"Failed to delete session from database: {e}")
    finally:
        db_conn.close()

    await clean_checkpointer_state(session_id, "delete session")

    return {"success": True}


@router.put("/{session_id}/rename")
async def rename_session(
    session_id: str, request: dict, current_user: Optional[User] = Depends(get_optional_current_user)
):
    try:
        new_title = request.get("title", "").strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")

        user_id = current_user.id if current_user else "anonymous"

        db_conn = get_session_db()
        try:
            db_conn.execute(
                "UPDATE sessions SET title = ?, last_updated = ? WHERE session_id = ? AND user_id = ?",
                (new_title, datetime.now().isoformat(), session_id, user_id),
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


@router.post("/{session_id}/auto-name")
async def auto_name_session(
    session_id: str, request: dict, current_user: Optional[User] = Depends(get_optional_current_user)
):
    try:
        from langchain_ollama import ChatOllama
        import os

        user_id = current_user.id if current_user else "anonymous"
        user_message = request.get("user_message", "").strip()
        agent_response = request.get("agent_response", "").strip()

        if not user_message:
            raise HTTPException(status_code=400, detail="User message is required for auto-naming")

        ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        llm = ChatOllama(
            model="gpt-oss:20b-cloud", base_url=ollama_base_url, temperature=0.7, format="", options={"num_predict": 60}
        )

        if agent_response and len(agent_response) > 50:
            agent_preview = agent_response[:500] if len(agent_response) > 500 else agent_response
            prompt = f"""Generate a concise, descriptive title (4-8 words) for a data science conversation.

User asked: "{user_message}"

Agent explained: "{agent_preview}"

Create a title describing the TOPIC or CAPABILITY discussed, not the user's question.

Good examples:
- "Supported Data File Formats"
- "Dataset Format Compatibility Overview"
- "Sales Trend Analysis"
- "Customer Churn Prediction Model"

Bad examples:
- "hello what dataset formats can" (too literal, incomplete)
- "what sort of dataset" (copying user question)

Return ONLY the title. No quotes. No periods."""
        else:
            prompt = f"""Generate a descriptive title (4-8 words) for: "{user_message}"

Focus on the topic, not the question. Return ONLY the title. No quotes. No periods."""

        response = llm.invoke(prompt)
        title = response.content.strip().strip('"').strip("'")

        if len(title) > 60:
            title = title[:57] + "..."

        db_conn = get_session_db()
        try:
            db_conn.execute(
                "UPDATE sessions SET title = ?, last_updated = ? WHERE session_id = ? AND user_id = ?",
                (title, datetime.now().isoformat(), session_id, user_id),
            )
            db_conn.commit()
            logger.info(f"Auto-named session {session_id} to '{title}'")
        except Exception as e:
            logger.error(f"Failed to auto-name session in database: {e}")
            raise
        finally:
            db_conn.close()

        return {"success": True, "title": title}
    except Exception as e:
        logger.error(f"Failed to auto-name session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to auto-name session: {str(e)}")


@router.get("/{session_id}/messages")
async def get_session_messages(session_id: str, current_user: Optional[User] = Depends(get_optional_current_user)):
    user_id = current_user.id if current_user else "anonymous"
    from ..api import session_store
    from ..api_utils.message_storage import load_messages

    if session_id not in session_store:
        db_conn = get_session_db()
        try:
            cursor = db_conn.execute(
                "SELECT session_id, user_id, created_at FROM sessions WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            )
            row = cursor.fetchone()
            if row:
                session_id_db, user_id, created_at = row
                session_store[session_id] = {
                    "session_id": session_id_db,
                    "user_id": user_id,
                    "created_at": datetime.fromisoformat(created_at) if created_at else datetime.now(),
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

    messages = load_messages(session_id)
    if messages:
        logger.info(f"Loaded {len(messages)} messages from explicit storage for session {session_id}")
        return messages

    logger.info(f"No messages in explicit storage, checking checkpoint for session {session_id}")

    try:
        from data_scientist_chatbot.app.core.graph_builder import DB_PATH
        import aiosqlite

        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                "SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row:
                try:
                    if not row[0] or len(row[0]) < 10:
                        logger.warning(f"Checkpoint data too small or empty for session {session_id}")
                        return messages

                    checkpoint_data = pickle.loads(row[0])
                    logger.info(f"Checkpoint data keys: {checkpoint_data.keys()}")
                except (pickle.UnpicklingError, EOFError, AttributeError) as pickle_error:
                    logger.error(f"Corrupted checkpoint data for session {session_id}: {pickle_error}")
                    logger.info("Attempting to clean corrupted checkpoint...")
                    try:
                        await db.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
                        await db.commit()
                        logger.info(f"Cleaned corrupted checkpoint for session {session_id}")
                    except Exception as cleanup_error:
                        logger.error(f"Failed to clean checkpoint: {cleanup_error}")
                    return messages
                except Exception as e:
                    logger.error(f"Unexpected error loading checkpoint for session {session_id}: {e}")
                    return messages
                if "channel_values" in checkpoint_data and "messages" in checkpoint_data["channel_values"]:
                    msg_count = len(checkpoint_data["channel_values"]["messages"])
                    logger.info(f"Found {msg_count} messages in checkpoint for session {session_id}")
                    for msg in checkpoint_data["channel_values"]["messages"]:
                        if hasattr(msg, "type") and msg.type in ["human", "ai"]:
                            messages.append(
                                {"type": msg.type, "content": str(msg.content), "timestamp": datetime.now().isoformat()}
                            )
                    logger.info(f"Returning {len(messages)} messages for session {session_id}")
                else:
                    logger.warning(f"No messages found in checkpoint for session {session_id}")
            else:
                logger.warning(f"No checkpoint found for session {session_id}")
    except Exception as e:
        logger.error(f"Error loading messages from checkpointer: {e}", exc_info=True)

    return messages


@router.get("/{session_id}/export")
async def export_session(
    session_id: str, format: str = "markdown", current_user: Optional[User] = Depends(get_optional_current_user)
):
    user_id = current_user.id if current_user else "anonymous"
    from ..api_utils.message_storage import load_messages
    from fastapi.responses import Response
    import markdown

    messages = load_messages(session_id)

    if not messages:
        try:
            from data_scientist_chatbot.app.core.graph_builder import DB_PATH
            import aiosqlite

            async with aiosqlite.connect(DB_PATH) as db:
                cursor = await db.execute(
                    "SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                    (session_id,),
                )
                row = await cursor.fetchone()
                if row:
                    checkpoint_data = pickle.loads(row[0])
                    if "channel_values" in checkpoint_data and "messages" in checkpoint_data["channel_values"]:
                        for msg in checkpoint_data["channel_values"]["messages"]:
                            if hasattr(msg, "type") and msg.type in ["human", "ai"]:
                                messages.append(
                                    {
                                        "type": msg.type,
                                        "content": str(msg.content),
                                        "timestamp": datetime.now().isoformat(),
                                    }
                                )
        except Exception as e:
            logger.error(f"Error loading messages for export: {e}")

    if not messages:
        raise HTTPException(status_code=404, detail="No messages found in session")

    db_conn = get_session_db()
    try:
        cursor = db_conn.execute(
            "SELECT title FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id)
        )
        row = cursor.fetchone()
        session_title = row[0] if row and row[0] else "Conversation"
    except Exception:
        session_title = "Conversation"
    finally:
        db_conn.close()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if format == "markdown":
        content = f"# {session_title}\n\n"
        content += f"*Exported on {timestamp}*\n\n"
        content += "---\n\n"

        for msg in messages:
            role = "User" if msg["type"] == "human" else "Assistant"
            content += f"## {role}\n\n"
            content += f"{msg['content']}\n\n"
            content += "---\n\n"

        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.md"},
        )

    elif format == "json":
        export_data = {"session_id": session_id, "title": session_title, "exported_at": timestamp, "messages": messages}

        return Response(
            content=json.dumps(export_data, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.json"},
        )

    elif format == "html":
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{session_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 0.5rem;
        }}
        .meta {{
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 2rem;
        }}
        .message {{
            margin: 1.5rem 0;
            padding: 1rem;
            border-radius: 8px;
        }}
        .user {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-left: 4px solid #667eea;
        }}
        .assistant {{
            background: rgba(0, 0, 0, 0.02);
            border-left: 4px solid #764ba2;
        }}
        .role {{
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #667eea;
        }}
        .assistant .role {{
            color: #764ba2;
        }}
        .content {{
            line-height: 1.6;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{session_title}</h1>
        <div class="meta">Exported on {timestamp}</div>
"""

        for msg in messages:
            role = "User" if msg["type"] == "human" else "Assistant"
            role_class = "user" if msg["type"] == "human" else "assistant"
            html_content += f"""
        <div class="message {role_class}">
            <div class="role">{role}</div>
            <div class="content">{msg['content']}</div>
        </div>
"""

        html_content += """
    </div>
</body>
</html>
"""

        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.html"},
        )

    elif format == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_LEFT
            from io import BytesIO

            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=24,
                textColor="#667eea",
                spaceAfter=12,
            )

            meta_style = ParagraphStyle(
                "Meta",
                parent=styles["Normal"],
                fontSize=10,
                textColor="#666666",
                spaceAfter=20,
            )

            user_style = ParagraphStyle(
                "User",
                parent=styles["Normal"],
                fontSize=11,
                leftIndent=20,
                spaceAfter=12,
                textColor="#667eea",
            )

            assistant_style = ParagraphStyle(
                "Assistant",
                parent=styles["Normal"],
                fontSize=11,
                leftIndent=20,
                spaceAfter=12,
                textColor="#764ba2",
            )

            story = []
            story.append(Paragraph(session_title, title_style))
            story.append(Paragraph(f"Exported on {timestamp}", meta_style))
            story.append(Spacer(1, 0.2 * inch))

            for msg in messages:
                role = "User" if msg["type"] == "human" else "Assistant"
                style = user_style if msg["type"] == "human" else assistant_style

                story.append(Paragraph(f"<b>{role}:</b>", style))
                story.append(Paragraph(msg["content"].replace("\n", "<br/>"), style))
                story.append(Spacer(1, 0.3 * inch))

            doc.build(story)
            buffer.seek(0)

            return Response(
                content=buffer.getvalue(),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=conversation_{session_id[:8]}.pdf"},
            )
        except ImportError:
            raise HTTPException(
                status_code=501, detail="PDF export requires reportlab package. Install with: pip install reportlab"
            )

    else:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format: {format}. Supported: markdown, json, html, pdf"
        )


@router.post("/{session_id}/revert/{message_index}")
async def revert_to_message(
    session_id: str, message_index: int, current_user: Optional[User] = Depends(get_optional_current_user)
):
    """
    Revert conversation to a specific message index.
    Deletes all messages and checkpoints after the specified index.

    Example: If at message #14 and revert to #3, deletes messages 4-14
    """
    from ..api_utils.message_storage import load_messages, save_messages

    user_id = current_user.id if current_user else "anonymous"

    try:
        messages = load_messages(session_id)

        if not messages:
            try:
                from data_scientist_chatbot.app.core.graph_builder import DB_PATH
                import aiosqlite

                async with aiosqlite.connect(DB_PATH) as db:
                    cursor = await db.execute(
                        "SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                        (session_id,),
                    )
                    row = await cursor.fetchone()
                    if row:
                        checkpoint_data = pickle.loads(row[0])
                        if "channel_values" in checkpoint_data and "messages" in checkpoint_data["channel_values"]:
                            for msg in checkpoint_data["channel_values"]["messages"]:
                                if hasattr(msg, "type") and msg.type in ["human", "ai"]:
                                    messages.append(
                                        {
                                            "type": msg.type,
                                            "content": str(msg.content),
                                            "timestamp": datetime.now().isoformat(),
                                        }
                                    )
            except Exception as e:
                logger.error(f"Error loading messages for revert: {e}")

        if not messages or message_index < 0 or message_index >= len(messages):
            raise HTTPException(status_code=400, detail="Invalid message index")

        db_conn = get_session_db()
        try:
            cursor = db_conn.execute("SELECT user_id FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()

            if row:
                if row[0] != user_id:
                    raise HTTPException(status_code=403, detail="Unauthorized")
            else:
                logger.warning(f"Session {session_id} not in DB, creating entry for user {user_id}")
                db_conn.execute(
                    "INSERT INTO sessions (session_id, user_id, created_at, last_updated) VALUES (?, ?, ?, ?)",
                    (session_id, user_id, datetime.now().isoformat(), datetime.now().isoformat()),
                )
                db_conn.commit()
        finally:
            db_conn.close()

        kept_messages = messages[: message_index + 1]

        save_messages(session_id, kept_messages)

        try:
            from data_scientist_chatbot.app.core.graph_builder import DB_PATH
            import aiosqlite

            async with aiosqlite.connect(DB_PATH) as db:
                cursor = await db.execute(
                    "SELECT checkpoint_id, checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id ASC",
                    (session_id,),
                )
                checkpoints = await cursor.fetchall()

                target_checkpoint_id = None

                for idx, (checkpoint_id, checkpoint_blob) in enumerate(checkpoints):
                    try:
                        checkpoint_data = pickle.loads(checkpoint_blob)
                        if "channel_values" in checkpoint_data and "messages" in checkpoint_data["channel_values"]:
                            checkpoint_msg_count = sum(
                                1
                                for msg in checkpoint_data["channel_values"]["messages"]
                                if hasattr(msg, "type") and msg.type in ["human", "ai"]
                            )

                            if checkpoint_msg_count == message_index + 1:
                                target_checkpoint_id = checkpoint_id
                                break
                            elif checkpoint_msg_count > message_index + 1:
                                if idx > 0:
                                    target_checkpoint_id = checkpoints[idx - 1][0]
                                break
                    except Exception as e:
                        logger.error(f"Error parsing checkpoint: {e}")
                        continue

                if target_checkpoint_id:
                    await db.execute(
                        "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id > ?",
                        (session_id, target_checkpoint_id),
                    )
                    await db.commit()
                    logger.info(f"Deleted checkpoints after ID {target_checkpoint_id} for session {session_id}")
        except Exception as e:
            logger.error(f"Error reverting checkpoints: {e}")

        return {
            "success": True,
            "session_id": session_id,
            "reverted_to_index": message_index,
            "remaining_messages": len(kept_messages),
            "message": f"Reverted to message #{message_index + 1}, deleted {len(messages) - len(kept_messages)} messages",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reverting session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to revert session: {str(e)}")


@router.get("/search")
async def search_conversations(q: str, current_user: Optional[User] = Depends(get_optional_current_user)):
    """
    Search through user's conversations

    Args:
        q: Search query string

    Returns:
        List of matching sessions with highlighted message excerpts
    """
    if not q or len(q.strip()) < 2:
        return {"results": [], "query": q}

    query = q.strip().lower()
    results = []

    db_conn = get_session_db()
    try:
        cursor = db_conn.execute(
            "SELECT session_id, title, created_at FROM sessions WHERE user_id = ? ORDER BY last_updated DESC",
            (user_id,),
        )
        sessions = cursor.fetchall()

        from ..api_utils.message_storage import load_messages

        for session_id, title, created_at in sessions:
            from data_scientist_chatbot.app.core.graph_builder import DB_PATH
            import aiosqlite

            messages = load_messages(session_id)

            if not messages:
                try:
                    async with aiosqlite.connect(DB_PATH) as db:
                        cursor = await db.execute(
                            "SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                            (session_id,),
                        )
                        row = await cursor.fetchone()
                        if row:
                            checkpoint_data = pickle.loads(row[0])
                            if "channel_values" in checkpoint_data and "messages" in checkpoint_data["channel_values"]:
                                for msg in checkpoint_data["channel_values"]["messages"]:
                                    if hasattr(msg, "type") and msg.type in ["human", "ai"]:
                                        messages.append(
                                            {
                                                "type": msg.type,
                                                "content": str(msg.content),
                                                "timestamp": datetime.now().isoformat(),
                                            }
                                        )
                except Exception as e:
                    logger.error(f"Error loading messages for search: {e}")

            matching_messages = []
            for idx, msg in enumerate(messages):
                content_lower = msg["content"].lower()
                if query in content_lower:
                    query_pos = content_lower.find(query)
                    start = max(0, query_pos - 50)
                    end = min(len(msg["content"]), query_pos + len(query) + 50)

                    excerpt = msg["content"][start:end]
                    if start > 0:
                        excerpt = "..." + excerpt
                    if end < len(msg["content"]):
                        excerpt = excerpt + "..."

                    matching_messages.append(
                        {
                            "message_index": idx,
                            "type": msg["type"],
                            "excerpt": excerpt,
                            "timestamp": msg.get("timestamp"),
                        }
                    )

            if matching_messages:
                results.append(
                    {
                        "session_id": session_id,
                        "title": title or "Untitled Conversation",
                        "created_at": created_at,
                        "match_count": len(matching_messages),
                        "matches": matching_messages[:3],
                    }
                )

        results.sort(key=lambda x: x["match_count"], reverse=True)

        return {"results": results[:20], "query": q, "total_results": len(results)}

    except Exception as e:
        logger.error(f"Error searching conversations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    finally:
        db_conn.close()
