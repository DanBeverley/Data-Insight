import io
import json
import logging
import time
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from langchain_core.messages import ToolMessage
import uuid
import re

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastAPIIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

try:
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    chatbot_path = project_root / "data_scientist_chatbot"
    app_path = chatbot_path / "app"
    sys.path.insert(0, str(chatbot_path))
    sys.path.insert(0, str(app_path))
    from data_scientist_chatbot.app.core.graph_builder import create_enhanced_agent_executor

    create_agent_executor = create_enhanced_agent_executor
    CHAT_AVAILABLE = True
    print("Chat functionality loaded successfully")
except Exception as e:
    CHAT_AVAILABLE = False
    create_agent_executor = None
    print(f"WARNING: Chat functionality not available: {e}")
    import traceback

    traceback.print_exc()
from .common.data_ingestion import ingest_data, ingest_from_url
from .data_quality.validator import DataQualityValidator
from .intelligence.data_profiler import IntelligentDataProfiler
from .api_utils.session_management import clean_checkpointer_state, get_or_create_agent_session, validate_session
from .api_utils.agent_response import extract_plot_urls, extract_agent_response
from .api_utils.data_ingestion import (
    process_dataframe_ingestion,
    create_intelligence_summary,
    load_dataframe_to_sandbox,
)
from .api_utils.helpers import (
    convert_pandas_output_to_html,
    create_agent_input,
    run_agent_task,
    create_workflow_status_context,
)
from .api_utils.models import (
    DataIngestionRequest,
    ProfilingRequest,
    FeatureRecommendationRequest,
    ChatMessage,
    ChatResponse,
    AgentChatRequest,
)
from .api_utils.upload_handler import enhance_with_agent_profile, load_data_to_agent_sandbox
from .api_utils.artifact_handler import handle_artifact_download
from .api_utils.streaming_service import stream_agent_chat
from .routers import data_router, session_router, auth_router

try:
    from .database.service import get_database_service
    from .database.connection import get_database_manager
    from .learning.persistent_storage import PersistentMetaDatabase

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

GLOBAL_SESSION_ID = "persistent_app_session"
session_agents = {}
agent_sessions = session_agents
AGENT_AVAILABLE = CHAT_AVAILABLE
async_tasks = {}
task_executor = None
performance_monitor = None
status_agent_runnable = None


def get_session_agent(session_id: str):
    if CHAT_AVAILABLE:
        return get_or_create_agent_session(session_id, session_agents, create_enhanced_agent_executor)
    return session_agents.get(session_id)


SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_AVAILABLE and SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        integrations=[
            FastAPIIntegration(),
            SqlalchemyIntegration(),
        ],
        traces_sample_rate=0.1,
        profiles_sample_rate=0.1,
        environment=os.getenv("ENVIRONMENT", "development"),
    )
    print("Sentry monitoring enabled")
elif SENTRY_DSN and not SENTRY_AVAILABLE:
    print("WARNING: Sentry DSN provided but sentry-sdk not installed")
else:
    print("INFO: Sentry monitoring disabled (no SENTRY_DSN configured)")

app = FastAPI(title="DnA", description="", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.on_event("startup")
async def startup_event():
    """Initialize database schema on application startup"""
    try:
        from .database.migrations import initialize_database_schema
        from .database.service import get_database_service

        db_service = get_database_service()
        if db_service:
            success = initialize_database_schema(db_service.db)
            if success:
                logger.info("Database schema initialized successfully")
            else:
                logger.warning("WARNING: Database schema initialization had issues")
        else:
            logger.warning("WARNING: Database service not available")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")

    try:
        from data_scientist_chatbot.app.core.graph_builder import perform_checkpoint_maintenance

        logger.info("Validating checkpoint system...")
        result = perform_checkpoint_maintenance(force_cleanup=False)

        if result.get("status") == "healthy":
            logger.info(f"Checkpoint system healthy: {result.get('stats', {})}")
        elif result.get("status") == "degraded":
            logger.warning(f"Checkpoint system degraded, auto-cleaned: {result.get('operations', [])}")
        elif result.get("status") == "unavailable":
            logger.info("Checkpoint system not configured")
        else:
            logger.error(f"Checkpoint system error: {result.get('message', 'Unknown error')}")
    except Exception as e:
        logger.warning(f"Checkpoint validation skipped: {e}")


logger = logging.getLogger(__name__)


static_dir = Path(__file__).parent.parent / "static"


@app.get("/static/script.js")
async def get_script():
    """Serve JavaScript file with correct MIME type."""
    script_path = static_dir / "script.js"
    return FileResponse(script_path, media_type="application/javascript")


@app.get("/static/styles.css")
async def get_styles():
    """Serve CSS file with correct MIME type."""
    css_path = static_dir / "styles.css"
    return FileResponse(css_path, media_type="text/css")


app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(auth_router)
app.include_router(data_router)
app.include_router(session_router)

session_store = {}

import builtins

builtins._session_store = session_store

data_profiler = IntelligentDataProfiler()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logging.info(f"WebSocket connected for session {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logging.info(f"WebSocket disconnected for session {session_id}")

    async def send_progress(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(message)
            except Exception as e:
                logging.error(f"Error sending progress to {session_id}: {e}")
                self.disconnect(session_id)


manager = ConnectionManager()


async def send_profiling_update(session_id: str, percent: int, message: str):
    if session_id in manager.active_connections:
        await manager.send_personal_message(
            json.dumps({"type": "progress", "percent": percent, "message": message}), session_id
        )


async def run_profiling_background(df: pd.DataFrame, session_id: str, filename: str):
    try:
        from src.intelligence.hybrid_data_profiler import generate_dataset_profile_for_agent
        import builtins
        from .api_utils.session_persistence import session_data_store

        def progress_callback(percent: int, message: str):
            session_store[session_id]["profiling_message"] = message
            if session_id in manager.active_connections:
                try:
                    asyncio.create_task(
                        manager.send_progress(session_id, {"type": "progress", "percent": percent, "message": message})
                    )
                except:
                    pass

        loop = asyncio.get_event_loop()

        data_profile = await loop.run_in_executor(
            None,
            lambda: generate_dataset_profile_for_agent(
                df,
                context={"filename": filename, "upload_session": session_id},
                config={"progress_callback": progress_callback},
            ),
        )

        session_data = {"dataframe": df, "data_profile": data_profile, "filename": filename}
        builtins._session_store[session_id] = session_data
        session_data_store.save_session_data(session_id, session_data)

        session_store[session_id]["profiling_status"] = "completed"
        session_store[session_id]["profiling_message"] = "Profiling complete"

        if session_id in manager.active_connections:
            await manager.send_progress(
                session_id,
                {
                    "type": "complete",
                    "profile": {
                        "quality_score": round(data_profile.quality_assessment.get("overall_score", 0), 1)
                        if data_profile.quality_assessment.get("overall_score")
                        else None,
                        "anomalies_detected": data_profile.anomaly_detection.get("summary", {}).get(
                            "total_anomalies", 0
                        ),
                        "profiling_time": round(data_profile.profile_metadata.get("profiling_duration", 0), 2),
                    },
                },
            )

        logging.info(f"Background profiling completed for session {session_id}")
    except Exception as e:
        logging.error(f"Background profiling failed for session {session_id}: {e}")
        session_store[session_id]["profiling_status"] = "failed"
        session_store[session_id]["profiling_error"] = str(e)
        session_store[session_id]["profiling_message"] = f"Profiling failed: {str(e)}"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat-first landing interface."""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/api/agent/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """Handle chat messages with the AI agent"""
    try:
        from .api_utils.message_storage import save_message

        session_id = chat_request.session_id or str(uuid.uuid4())
        agent = get_session_agent(session_id)

        if not agent:
            if not CHAT_AVAILABLE:
                return ChatResponse(
                    status="error",
                    response="I apologize, but the chat functionality is currently unavailable. This appears to be due to missing dependencies (like dotenv, langchain, etc.). Please run 'pip install -r requirements.txt' to install all required packages.",
                    plots=[],
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to initialize chat agent for this session")

        save_message(session_id, "human", chat_request.message)

        input_state = {"messages": [("user", chat_request.message)], "session_id": session_id, "python_executions": 0}

        final_response_content = ""
        all_plot_urls = []

        try:
            print(f"FastAPI DEBUG: Starting agent stream for session {session_id}")
            for chunk in agent.stream(
                input_state, config={"configurable": {"thread_id": session_id}, "recursion_limit": 10}
            ):
                print(f"FastAPI DEBUG: Received chunk with keys: {list(chunk.keys())}")

                if "action" in chunk:
                    print("FastAPI DEBUG: Processing tool execution")
                    tool_messages = chunk["action"]["messages"]
                    print(f"FastAPI DEBUG: Found {len(tool_messages)} tool messages")

                    for i, tool_msg in enumerate(tool_messages):
                        print(f"FastAPI DEBUG: Tool message {i} type: {type(tool_msg)}")
                        print(f"FastAPI DEBUG: Tool message {i} attributes: {dir(tool_msg)}")

                        # Check different possible attribute names for tool identification
                        tool_name = (
                            getattr(tool_msg, "name", None)
                            or getattr(tool_msg, "tool_name", None)
                            or getattr(tool_msg, "type", None)
                        )
                        print(f"FastAPI DEBUG: Tool message {i} identifier: {tool_name}")

                        content = str(getattr(tool_msg, "content", ""))
                        print(f"FastAPI DEBUG: Tool message {i} content preview: {content[:100]}...")

                        # Always check for PLOT_SAVED regardless of message type
                        if "PLOT_SAVED:" in content:
                            print("FastAPI DEBUG: Found PLOT_SAVED in content!")
                            plot_saved_matches = re.findall(r"PLOT_SAVED:([^\n\r\s]+\.png)", content)
                            print(f"FastAPI DEBUG: Plot saved matches: {plot_saved_matches}")

                            for plot_file in plot_saved_matches:
                                url = f"/static/plots/{plot_file}"
                                if url not in all_plot_urls:
                                    all_plot_urls.append(url)
                                    print(f"FastAPI DEBUG: Added plot URL: {url}")

                        # Also check for any PNG files mentioned
                        if ".png" in content:
                            print("FastAPI DEBUG: Found .png in content!")
                            plot_files = re.findall(r"([a-zA-Z0-9_\-]+\.png)", content)
                            print(f"FastAPI DEBUG: All PNG files found: {plot_files}")

                            for pf in plot_files:
                                if "plot" in pf:  # Only plot files
                                    url = f"/static/plots/{pf}"
                                    if url not in all_plot_urls:
                                        all_plot_urls.append(url)
                                        print(f"FastAPI DEBUG: Added plot URL from PNG: {url}")

                if "agent" in chunk:
                    print("FastAPI DEBUG: Processing agent response")
                    final_message = chunk["agent"]["messages"][-1]
                    if final_message:
                        final_response_content = final_message.content
                        print(f"FastAPI DEBUG: Final response: {final_response_content[:100]}...")

            if not final_response_content:
                final_state = agent.invoke(
                    input_state, config={"configurable": {"thread_id": session_id}, "recursion_limit": 10}
                )
                final_response_content = final_state["messages"][-1].content

        except Exception as agent_error:
            print(f"Agent streaming error: {str(agent_error)}")
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Agent error: {str(agent_error)}")

        metadata = {"plots": all_plot_urls} if all_plot_urls else None
        save_message(session_id, "ai", final_response_content, metadata=metadata)

        return ChatResponse(status="success", response=final_response_content, plots=all_plot_urls)

    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/api/upload")
async def upload_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enable_profiling: bool = Form(True),
    session_id: str = Form(...),
):
    """Upload and validate dataset with progressive profiling."""
    try:
        if not session_id or not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")

        contents = await file.read()

        import builtins

        if not hasattr(builtins, "_session_store"):
            builtins._session_store = {}

        if file.filename.endswith(".zip"):
            from pathlib import Path
            import tempfile
            from src.api_utils.dataset_manifest import handle_file_upload

            temp_file = Path(tempfile.gettempdir()) / f"upload_{session_id}_{file.filename}"
            with open(temp_file, "wb") as f:
                f.write(contents)

            result = handle_file_upload(temp_file, file.filename, session_id)

            if not result["success"]:
                raise HTTPException(status_code=400, detail=result.get("error", "Failed to process ZIP file"))

            builtins._session_store[session_id] = {
                "dataset_manifest": result["manifest"],
                "dataset_type": "multi_file",
                "upload_filename": file.filename,
            }

            temp_file.unlink()

            return {
                "status": "success",
                "session_id": session_id,
                "agent_session_id": session_id,
                "message": f"Uploaded archive with {result['manifest']['total_files']} files",
                "dataset_type": "multi_file",
                "manifest_summary": {
                    "total_files": result["manifest"]["total_files"],
                    "total_size_mb": result["manifest"]["total_size_mb"],
                    "file_types": result["manifest"].get("file_types_summary", {}),
                },
            }

        df = ingest_data(io.BytesIO(contents), filename=file.filename)

        if df is None:
            raise HTTPException(status_code=400, detail="Failed to parse uploaded file")

        response_data = process_dataframe_ingestion(
            df, session_id, {"filename": file.filename}, False, data_profiler, session_store
        )
        session_id = response_data["session_id"]

        load_data_to_agent_sandbox(df, session_id, session_agents, create_enhanced_agent_executor, response_data)

        if enable_profiling:
            response_data["profiling_status"] = "pending"
            response_data[
                "message"
            ] = f"Dataset uploaded successfully - {len(df)} rows, {len(df.columns)} columns. Profiling in progress..."
            background_tasks.add_task(run_profiling_background, df.copy(), session_id, file.filename)
        else:
            response_data["profiling_status"] = "disabled"

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@app.post("/api/privacy/consent")
async def handle_privacy_consent(request: dict):
    """Handle user consent for privacy protection"""
    try:
        session_id = request.get("session_id")
        apply_protection = request.get("apply_protection", False)

        if not session_id or session_id not in session_store:
            raise HTTPException(status_code=404, detail="Session not found")

        import builtins

        if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
            session_data = builtins._session_store[session_id]

            if apply_protection:
                # Apply privacy protection using PrivacyEngine
                from security.privacy_engine import PrivacyEngine

                privacy_engine = PrivacyEngine()

                original_df = session_data["dataframe"]
                protected_df, transformations = privacy_engine.apply_k_anonymity(
                    original_df, privacy_engine._auto_detect_quasi_identifiers(original_df)
                )

                # Update stored dataframe with protected version
                session_data["dataframe"] = protected_df
                session_data["privacy_applied"] = True
                session_data["privacy_transformations"] = transformations

                # Update session store as well
                session_store[session_id]["dataframe"] = protected_df

                return {
                    "status": "success",
                    "message": "Privacy protection applied successfully",
                    "protection_applied": True,
                }
            else:
                # Mark that user declined protection
                session_data["privacy_applied"] = False
                return {
                    "status": "success",
                    "message": "Continuing without privacy protection",
                    "protection_applied": False,
                }

        raise HTTPException(status_code=404, detail="Session data not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Privacy consent error: {str(e)}")


@app.post("/api/ingest-url")
async def ingest_from_url_endpoint(request: DataIngestionRequest):
    """Ingest data from URL with intelligent profiling."""
    try:
        df = ingest_from_url(request.url, request.data_type)

        if df is None:
            raise HTTPException(status_code=400, detail="Failed to fetch or parse data from URL")

        response_data = process_dataframe_ingestion(
            df,
            request.session_id,
            {"source": request.url, "source_url": request.url},
            request.enable_profiling,
            data_profiler,
            session_store,
        )
        session_id = response_data["session_id"]
        import builtins

        if not hasattr(builtins, "_session_store"):
            builtins._session_store = {}
        builtins._session_store[session_id] = {"dataframe": df}

        try:
            dataset_content = f"""
Dataset Summary:\n- Shape: {df.shape[0]} rows x {df.shape[1]} columns\n- Columns: {', '.join(df.columns[:30])}{'...' if len(df.columns) > 30 else ''}\n- Dtypes counts: {pd.Series(df.dtypes.astype(str)).value_counts().to_dict()}
"""
        except Exception as e:
            logging.warning(f"Failed to add dataset summary to RAG (URL): {e}")
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting from URL: {str(e)}")


@app.get("/favicon.ico")
async def favicon():
    """Simple favicon to prevent 404 errors."""
    return {"status": "no favicon"}


@app.post("/api/agent/chat")
async def agent_chat(request: AgentChatRequest):
    """Chat with AI data science agent using existing session system."""
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent service not available")

    try:
        from .api_utils.message_storage import save_message

        session_id = request.session_id
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID required")

        # Create session in session_store if it doesn't exist
        if session_id not in session_store:
            session_store[session_id] = {"session_id": session_id, "created_at": datetime.now()}

        # Initialize enhanced agent for this session if needed
        if create_enhanced_agent_executor is None:
            raise HTTPException(status_code=503, detail="Agent service not available")

        agent = get_or_create_agent_session(session_id, agent_sessions, create_enhanced_agent_executor)

        # Data is already loaded in the stateful sandbox, just use the user's message
        user_message = request.message

        save_message(session_id, "human", user_message)

        # Check for long-running tasks (model building, complex analysis)
        is_long_task = any(
            keyword in request.message.lower()
            for keyword in ["build model", "train model", "complex analysis", "deep analysis", "correlation analysis"]
        )

        if request.async_execution or is_long_task:
            # Run asynchronously
            task_id = str(uuid.uuid4())

            # Submit task to thread pool
            future = task_executor.submit(run_agent_task, agent, user_message, session_id)
            async_tasks[task_id] = {
                "future": future,
                "status": "running",
                "session_id": session_id,
                "created_at": time.time(),
            }

            if performance_monitor:
                performance_monitor.record_metric(
                    session_id=session_id, metric_name="async_task_created", value=1.0, context={"task_id": task_id}
                )

            return {
                "status": "async",
                "task_id": task_id,
                "message": "Task started. Use /api/agent/task-status to check progress.",
                "session_id": session_id,
            }
        else:
            # Run synchronously
            print(f"DEBUG: Using agent type: {type(agent)}")
            response = agent.invoke(create_agent_input(user_message, session_id))

            # Extract plots and response from messages
            messages = response.get("messages", [])
            content, plots = extract_agent_response(messages, recent_count=3)

            print(f"DEBUG: Final response content: {content[:100]}")
            print(f"DEBUG: Response has {len(plots)} plots")

            metadata = {"plots": plots} if plots else None
            save_message(session_id, "ai", content, metadata=metadata)

            return {"status": "success", "response": content, "session_id": session_id, "plots": plots}
    except Exception as e:
        return {"status": "error", "detail": f"Agent chat error: {str(e)}"}


@app.post("/api/agent/cancel/{session_id}")
async def cancel_agent_task(session_id: str):
    """
    Cancel an in-progress agent task.
    Marks the session as cancelled so streaming can gracefully stop.
    """
    if not DATABASE_AVAILABLE:
        return {"success": False, "detail": "Database not available"}

    try:
        from .database.models import CancelledTask
        from .database.connection import get_database_manager

        db_manager = get_database_manager()
        db = db_manager.db

        cancelled_task = CancelledTask(session_id=session_id, cancelled_at=datetime.now(), reason="user_requested")

        db.add(cancelled_task)
        db.commit()

        logging.info(f"Marked session {session_id} as cancelled")
        return {"success": True, "session_id": session_id}

    except Exception as e:
        logging.error(f"Failed to cancel task: {e}")
        return {"success": False, "detail": str(e)}


@app.get("/api/agent/chat-stream")
async def agent_chat_stream(
    message: str, session_id: str, web_search_enabled: bool = False, token_streaming: bool = True
):
    """Stream agent chat responses with dynamic status updates."""
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent service not available")

    if session_id not in session_store:
        raise HTTPException(
            status_code=404, detail="Session not found. Please create a session first using /api/sessions/new"
        )

    from fastapi.responses import StreamingResponse

    async def generate_events():
        global status_agent_runnable

        session_store[session_id]["web_search_enabled"] = web_search_enabled
        session_store[session_id]["token_streaming"] = token_streaming

        if create_enhanced_agent_executor is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent service not available'})}\n\n"
            return

        agent = get_or_create_agent_session(session_id, agent_sessions, create_enhanced_agent_executor)

        if status_agent_runnable is None:
            try:
                from data_scientist_chatbot.app.core.agent_factory import create_status_agent

                status_agent_runnable = create_status_agent()
                print(f"INFO: Status agent initialized successfully")
            except Exception as e:
                print(f"ERROR: Failed to initialize status agent: {e}")
                import traceback

                print(traceback.format_exc())

        async for event_data in stream_agent_chat(
            message, session_id, agent, session_store, status_agent_runnable, agent_sessions
        ):
            yield event_data

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )


@app.get("/api/agent/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Check status of async agent task."""
    if task_id not in async_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task_info = async_tasks[task_id]
    future = task_info["future"]

    if future.done():
        try:
            result = future.result()
            if result["success"]:
                async_tasks[task_id]["status"] = "completed"
                return {"status": "completed", "task_id": task_id, "result": result["response"]}
            else:
                async_tasks[task_id]["status"] = "failed"
                return {"status": "failed", "task_id": task_id, "error": result["error"]}
        except Exception as e:
            async_tasks[task_id]["status"] = "failed"
            return {"status": "failed", "task_id": task_id, "error": str(e)}
    else:
        return {"status": "running", "task_id": task_id, "message": "Task is still processing..."}


@app.get("/api/agent/artifacts/{session_id}/{filename}")
async def get_agent_artifact(session_id: str, filename: str):
    """Serve generated files (plots, reports) from agent execution."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        # Check in generated_plots directory with session-specific naming
        artifact_path = Path(__file__).parent.parent / "data_scientist_chatbot" / "generated_plots" / filename

        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")

        # Determine media type
        if filename.endswith(".png"):
            media_type = "image/png"
        elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
            media_type = "image/jpeg"
        elif filename.endswith(".html"):
            media_type = "text/html"
        elif filename.endswith(".csv"):
            media_type = "text/csv"
        else:
            media_type = "application/octet-stream"

        return FileResponse(artifact_path, media_type=media_type, filename=filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving artifact: {str(e)}")


@app.websocket("/ws/profiling/{session_id}")
async def websocket_profiling_progress(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time profiling progress updates"""
    await manager.connect(websocket, session_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)


@app.get("/api/profiling-status/{session_id}")
async def get_profiling_status(session_id: str):
    """Polling endpoint for profiling status (fallback for non-WebSocket clients)"""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = session_store[session_id]
    profiling_status = session_data.get("profiling_status", "unknown")
    profiling_message = session_data.get("profiling_message", "Processing...")

    if profiling_status == "completed" and "intelligence_summary" in session_data:
        return {"status": "completed", "message": profiling_message, "data": session_data["intelligence_summary"]}
    elif profiling_status == "failed":
        return {
            "status": "failed",
            "message": profiling_message,
            "error": session_data.get("profiling_error", "Unknown error"),
        }
    else:
        return {"status": profiling_status, "message": profiling_message}


@app.get("/api/health")
async def health_check():
    """Basic health check endpoint - returns 200 if app is running"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/ready")
async def readiness_check():
    """
    Readiness check endpoint for K8s/orchestration.
    Verifies all dependencies are available before accepting traffic.
    """
    ready_status = {"status": "ready", "timestamp": datetime.now().isoformat(), "checks": {}}
    all_ready = True

    if DATABASE_AVAILABLE:
        try:
            db_manager = get_database_manager()
            db_health = db_manager.execute_health_check()
            ready_status["checks"]["database"] = "ready" if db_health else "not_ready"
            if not db_health:
                all_ready = False
        except Exception as e:
            ready_status["checks"]["database"] = f"not_ready: {str(e)}"
            all_ready = False
    else:
        ready_status["checks"]["database"] = "not_configured"

    ready_status["checks"]["agent_service"] = "ready" if AGENT_AVAILABLE else "not_ready"
    if not AGENT_AVAILABLE:
        all_ready = False

    if AGENT_AVAILABLE and performance_monitor:
        try:
            cache_stats = performance_monitor.cache.get_stats()
            ready_status["checks"]["performance_monitoring"] = "ready"
            ready_status["cache_utilization"] = f"{cache_stats['size']}/{cache_stats['max_size']}"
            ready_status["active_agent_sessions"] = len(agent_sessions)
            ready_status["thread_pool_workers"] = task_executor._max_workers if task_executor else 0
        except:
            ready_status["checks"]["performance_monitoring"] = "degraded"
    else:
        ready_status["checks"]["performance_monitoring"] = "not_configured"

    ready_status["status"] = "ready" if all_ready else "not_ready"

    status_code = 200 if all_ready else 503
    return JSONResponse(content=ready_status, status_code=status_code)


@app.get("/api/checkpoint/health")
async def checkpoint_health():
    try:
        from data_scientist_chatbot.app.core.graph_builder import perform_checkpoint_maintenance

        result = perform_checkpoint_maintenance(force_cleanup=False)
        status_code = 200 if result.get("status") in ["healthy", "unavailable"] else 503

        return JSONResponse(content=result, status_code=status_code)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/checkpoint/maintenance")
async def trigger_checkpoint_maintenance():
    try:
        from data_scientist_chatbot.app.core.graph_builder import perform_checkpoint_maintenance

        result = perform_checkpoint_maintenance(force_cleanup=True)

        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/checkpoint/cleanup")
async def cleanup_old_checkpoints_endpoint(retention_days: int = 7):
    try:
        from data_scientist_chatbot.app.core.graph_builder import memory, cleanup_old_checkpoints

        if not memory:
            return JSONResponse(
                content={"status": "unavailable", "message": "Checkpoint system not available"}, status_code=503
            )

        deleted = cleanup_old_checkpoints(memory.conn, retention_days)

        return JSONResponse(
            content={"status": "success", "deleted_checkpoints": deleted, "retention_days": retention_days},
            status_code=200,
        )
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
