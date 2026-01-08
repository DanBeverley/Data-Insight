import io
import json
import logging
import time
import os
from pathlib import Path
import sys
import asyncio
from dotenv import load_dotenv

# Set Windows event loop policy for Playwright compatibility
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load environment variables
load_dotenv()

from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager, contextmanager
import pandas as pd
import joblib
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form,
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    Request,
)
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
from .api_utils.dataset_service import (
    process_dataframe_ingestion,
    create_intelligence_summary,
    load_dataframe_to_sandbox,
)
from .api_utils.helpers import (
    convert_pandas_output_to_html,
    create_agent_input,
    run_agent_task,
)
from data_scientist_chatbot.app.core.state_manager import create_workflow_status_context
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
from .routers import (
    data_router,
    session_router,
    auth_router,
    hierarchical_upload_router,
    report_router,
    alert_router,
    connection_router,
    notification_router,
)
from .auth.dependencies import get_optional_current_user
from .database.models import User

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if CHAT_AVAILABLE:
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
            from data_scientist_chatbot.app.core.graph_builder import (
                set_checkpointer,
                DB_PATH,
                perform_checkpoint_maintenance,
            )

            async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
                set_checkpointer(checkpointer)
                logging.info("AsyncSqliteSaver initialized successfully")

                result = await perform_checkpoint_maintenance(force_cleanup=False)
                if result.get("status") == "healthy":
                    logging.info(f"Checkpoint system healthy: {result.get('stats', {})}")
                elif result.get("status") == "degraded":
                    logging.warning(f"Checkpoint system degraded, auto-cleaned: {result.get('operations', [])}")

                try:
                    from data_scientist_chatbot.app.agent import warmup_models_parallel

                    logging.info("Starting parallel model warmup...")
                    await warmup_models_parallel()
                    logging.info("Model warmup completed successfully")

                    # Health check loop disabled to prevent 429 rate limiting
                    # from data_scientist_chatbot.app.core.model_manager import ModelManager
                    # model_manager = ModelManager()
                    # model_manager.start_health_check_loop()
                except Exception as warmup_err:
                    logging.warning(f"Model warmup failed (non-critical): {warmup_err}")

                try:
                    from src.scheduler.service import get_alert_scheduler

                    scheduler = get_alert_scheduler()
                    scheduler.start()
                    logging.info("Alert scheduler started successfully")
                except Exception as scheduler_err:
                    logging.warning(f"Alert scheduler failed (non-critical): {scheduler_err}")

                yield
        except Exception as e:
            logging.warning(f"Checkpointer initialization failed: {e}")
            yield
    else:
        yield


app = FastAPI(title="DnA", description="", version="2.0.0", lifespan=lifespan)

# Mount static assets
import os

static_dir = Path("static")
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.sessions import SessionMiddleware

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SECRET_KEY", "change-this-secret-key-in-production"),
    max_age=3600,
)

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

from src.routers.report_router import router as report_router

app.include_router(report_router)

from src.routers.image_router import router as image_router

app.include_router(image_router)


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


logger = logging.getLogger(__name__)


# Context manager to suppress profiling logs during background execution
@contextmanager
def suppress_profiling_logs():
    """Temporarily suppress noisy profiling logs to prevent interference with agent execution"""
    loggers_to_suppress = ["pandas_profiling", "ydata_profiling", "matplotlib", "numba", "PIL", "fontTools"]
    old_levels = {name: logging.getLogger(name).getEffectiveLevel() for name in loggers_to_suppress}

    for name in loggers_to_suppress:
        logging.getLogger(name).setLevel(logging.ERROR)

    try:
        yield
    finally:
        for name, level in old_levels.items():
            logging.getLogger(name).setLevel(level)


static_dir = Path(__file__).parent.parent / "static"


# Mount static files for React SPA
app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount reports directory
reports_dir = Path("data/reports")
reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

app.include_router(auth_router)
app.include_router(data_router)
app.include_router(session_router)
app.include_router(hierarchical_upload_router)
app.include_router(report_router)
app.include_router(alert_router)
app.include_router(connection_router)
app.include_router(notification_router)

from src.api_utils.session_management import session_data_manager

session_store = session_data_manager.store

# builtins._session_store is now managed by session_data_manager

# Client heartbeat tracking for long-running operations
session_heartbeats: Dict[str, float] = {}

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

        # Sample large DataFrames to reduce CPU contention with agent execution
        df_for_profiling = df.sample(n=5000, random_state=42) if len(df) > 5000 else df
        if len(df) > 5000:
            logging.info(f"[PROFILING] Sampled {len(df)} rows down to 5000 for efficient profiling")

        loop = asyncio.get_event_loop()

        def profiled_execution():
            """Execute profiling with suppressed logs to prevent interference"""
            with suppress_profiling_logs():
                return generate_dataset_profile_for_agent(
                    df_for_profiling,
                    context={"filename": filename, "upload_session": session_id},
                    config={"progress_callback": progress_callback, "minimal": True, "correlations": None},
                )

        # Add timeout for profiling (5 minutes max)
        try:
            data_profile = await asyncio.wait_for(
                loop.run_in_executor(None, profiled_execution), timeout=300  # 5 minutes
            )
        except asyncio.TimeoutError:
            logging.error(f"[PROFILING] Timeout after 5 minutes for session {session_id}")
            session_store[session_id]["profiling_status"] = "timeout"
            session_store[session_id]["profiling_message"] = "Profiling timed out - dataset may be too large"
            return

        # Update existing session data instead of overwriting
        if session_id not in session_store:
            session_store[session_id] = {}

        session_store[session_id].update({"dataframe": df, "data_profile": data_profile, "filename": filename})

        # Save the complete updated session data
        session_data_store.save_session_data(session_id, session_store[session_id])
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
    """Serve the React SPA."""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.get("/favicon.png")
async def get_favicon():
    """Serve favicon."""
    favicon_path = Path(__file__).parent.parent / "static" / "favicon.png"
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/png")
    return {"status": "no favicon"}


@app.post("/api/agent/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage, background_tasks: BackgroundTasks):
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
            for chunk in agent.stream(
                input_state, config={"configurable": {"thread_id": session_id}, "recursion_limit": 10}
            ):
                if "action" in chunk:
                    tool_messages = chunk["action"]["messages"]
                    for tool_msg in tool_messages:
                        content = str(getattr(tool_msg, "content", ""))

                        # Check for PLOT_SAVED
                        if "PLOT_SAVED:" in content:
                            plot_saved_matches = re.findall(r"PLOT_SAVED:([^\n\r\s]+\.png)", content)
                            for plot_file in plot_saved_matches:
                                url = f"/static/plots/{plot_file}"
                                if url not in all_plot_urls:
                                    all_plot_urls.append(url)

                        # Check for .png files
                        if ".png" in content:
                            plot_files = re.findall(r"([a-zA-Z0-9_\-]+\.png)", content)
                            for pf in plot_files:
                                if "plot" in pf:
                                    url = f"/static/plots/{pf}"
                                    if url not in all_plot_urls:
                                        all_plot_urls.append(url)

                if "agent" in chunk:
                    final_message = chunk["agent"]["messages"][-1]
                    if final_message:
                        final_response_content = final_message.content

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

        # Trigger auto-naming if needed
        try:
            from .routers.session_router import get_session_db, auto_name_session

            db_conn = get_session_db()
            cursor = db_conn.execute("SELECT title FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            current_title = row[0] if row else "New Chat"
            db_conn.close()

            if current_title == "New Chat" and len(final_response_content) > 20:
                # We need to wrap the async call for background tasks
                async def run_auto_name(sid, msg, resp):
                    try:
                        await auto_name_session(sid, {"user_message": msg, "agent_response": resp}, None)
                    except Exception as e:
                        print(f"Auto-naming failed: {e}")

                background_tasks.add_task(run_auto_name, session_id, chat_request.message, final_response_content)
        except Exception as e:
            print(f"Auto-naming check failed: {e}")

        return ChatResponse(status="success", response=final_response_content, plots=all_plot_urls)

    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/api/upload")
async def upload_data(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    enable_profiling: bool = Form(True),
    session_id: str = Form(...),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Upload and validate multiple datasets with progressive profiling."""
    try:
        if not session_id or not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")

        user_id = current_user.id if current_user else "anonymous"

        if session_id in session_store:
            session_owner = session_store[session_id].get("user_id")
            if session_owner and session_owner != user_id and session_owner != "anonymous":
                raise HTTPException(status_code=403, detail="Not authorized to access this session")
        else:
            session_store[session_id] = {"user_id": user_id}

        uploaded_details = []

        # Initialize datasets dict if not present
        if "datasets" not in session_store[session_id]:
            session_store[session_id]["datasets"] = {}

        for file in files:
            # Check file size (limit to 100MB to prevent memory issues)
            MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
            contents = await file.read()

            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} too large ({len(contents) / 1024 / 1024:.1f}MB). Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB",
                )

            if file.filename.endswith(".zip"):
                import tempfile
                from src.api_utils.dataset_manifest import handle_file_upload

                temp_file = Path(tempfile.gettempdir()) / f"upload_{session_id}_{file.filename}"
                with open(temp_file, "wb") as f:
                    f.write(contents)

                result = handle_file_upload(temp_file, file.filename, session_id)

                if not result["success"]:
                    # Log error but maybe continue with other files? For now raise to be safe
                    raise HTTPException(status_code=400, detail=result.get("error", "Failed to process ZIP file"))

                session_store[session_id].update(
                    {
                        "dataset_manifest": result["manifest"],
                        "dataset_type": "multi_file",
                        "upload_filename": file.filename,
                    }
                )

                temp_file.unlink()

                uploaded_details.append(
                    {"filename": file.filename, "type": "archive", "file_count": result["manifest"]["total_files"]}
                )
                continue

            IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
            file_ext = Path(file.filename).suffix.lower()

            if file_ext in IMAGE_EXTENSIONS:
                try:
                    from src.services.image_analyzer import ImageAnalyzer
                    import pandas as pd

                    upload_dir = Path("data/uploads") / session_id / "images"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    image_path = upload_dir / file.filename

                    with open(image_path, "wb") as f:
                        f.write(contents)

                    logging.info(f"[IMAGE] Analyzing uploaded image: {file.filename}")

                    analyzer = ImageAnalyzer()
                    extraction = await analyzer.extract_data(
                        image_path=str(image_path), analysis_type="auto", session_id=session_id
                    )

                    if extraction.get("rows") and extraction.get("columns"):
                        extraction = analyzer.validate_and_clean_data(extraction)
                        df = pd.DataFrame(extraction["rows"])

                        if list(df.columns) != extraction["columns"]:
                            df.columns = extraction["columns"][: len(df.columns)]

                        csv_filename = Path(file.filename).stem + "_extracted.csv"
                        csv_path = Path("data/uploads") / session_id / csv_filename
                        df.to_csv(csv_path, index=False)

                        logging.info(
                            f"[IMAGE] Extracted {len(df)} rows, {len(df.columns)} columns from {file.filename}"
                        )

                        dataset_name = csv_filename
                        session_store[session_id]["datasets"][dataset_name] = df

                        if "dataframe" not in session_store[session_id]:
                            session_store[session_id]["dataframe"] = df
                            session_store[session_id]["filename"] = csv_filename
                            session_store[session_id]["dataset_path"] = str(csv_path.absolute())
                            session_store[session_id]["source_image"] = file.filename

                        from src.api_utils.session_persistence import session_data_store

                        session_data_store.save_session_data(session_id, session_store[session_id])

                        response_data = process_dataframe_ingestion(
                            df, session_id, {"filename": csv_filename}, False, data_profiler, session_store
                        )
                        load_data_to_agent_sandbox(
                            df, session_id, session_agents, create_enhanced_agent_executor, response_data
                        )

                        if enable_profiling:
                            await run_profiling_background(df.copy(), session_id, csv_filename)

                        uploaded_details.append(
                            {
                                "filename": file.filename,
                                "type": "image",
                                "extracted_to": csv_filename,
                                "rows": len(df),
                                "columns": len(df.columns),
                                "confidence": extraction.get("confidence", 0),
                                "profiling": "started" if enable_profiling else "disabled",
                            }
                        )
                        continue
                    else:
                        logging.warning(f"[IMAGE] No data extracted from {file.filename}")
                        uploaded_details.append(
                            {
                                "filename": file.filename,
                                "type": "image",
                                "error": "No data extracted",
                                "notes": extraction.get("notes"),
                            }
                        )
                        continue
                except Exception as e:
                    logging.error(f"[IMAGE] Failed to analyze {file.filename}: {e}")
                    uploaded_details.append({"filename": file.filename, "type": "image", "error": str(e)})
                    continue

            df = ingest_data(io.BytesIO(contents), filename=file.filename)

            if df is None:
                logging.warning(f"Failed to parse uploaded file: {file.filename}")
                continue

            # Store in datasets dict
            dataset_name = file.filename
            session_store[session_id]["datasets"][dataset_name] = df

            # Set as primary dataframe if it's the first one or only one
            if "dataframe" not in session_store[session_id]:
                session_store[session_id]["dataframe"] = df
                session_store[session_id]["filename"] = file.filename

                # Save file to disk for report generator
                try:
                    upload_dir = Path("data/uploads") / session_id
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    file_path = upload_dir / file.filename

                    # Reset cursor to beginning of file
                    file.file.seek(0)
                    with open(file_path, "wb") as f:
                        f.write(await file.read())

                    session_store[session_id]["dataset_path"] = str(file_path.absolute())
                    logging.info(f"Saved dataset to {file_path}")
                    logging.info(
                        f"DEBUG: session_store[{session_id}] keys after save: {list(session_store[session_id].keys())}"
                    )
                    logging.info(f"DEBUG: dataset_path in store: {session_store[session_id].get('dataset_path')}")
                except Exception as e:
                    logging.error(f"Failed to save dataset to disk: {e}")

            # Persist session data
            from src.api_utils.session_persistence import session_data_store

            session_data_store.save_session_data(session_id, session_store[session_id])

            # Process ingestion for this specific dataframe
            # Note: process_dataframe_ingestion updates global session_store too
            response_data = process_dataframe_ingestion(
                df, session_id, {"filename": file.filename}, False, data_profiler, session_store
            )

            try:
                load_data_to_agent_sandbox(
                    df, session_id, session_agents, create_enhanced_agent_executor, response_data
                )
            except Exception as sandbox_error:
                logging.error(f"Failed to load data to agent sandbox (non-critical): {sandbox_error}")
                # We continue since the file is uploaded to the session/disk successfully

            if enable_profiling:
                # [INTELLIGENCE UPGRADE] Await profiling to ensure context is ready for Planner
                # We prioritize "Smart" behavior over "Fast Upload"
                logging.info(f"Starting synchronous profiling for {file.filename}...")
                await run_profiling_background(df.copy(), session_id, file.filename)
                logging.info(f"Profiling completed for {file.filename}")

            uploaded_details.append(
                {
                    "filename": file.filename,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "profiling": "started" if enable_profiling else "disabled",
                }
            )

        datasets_dict = session_store[session_id].get("datasets", {})
        if len(datasets_dict) > 1:
            try:
                from src.intelligence.dataset_clusterer import DatasetClusterer

                logging.info(f"[ORCHESTRATOR] Processing {len(datasets_dict)} datasets for session {session_id}")
                clusterer = DatasetClusterer()
                result = clusterer.process_datasets(datasets_dict)

                session_store[session_id]["dataset_registry"] = {
                    k: {"filename": v.filename, "rows": v.row_count, "columns": list(v.columns), "domain": v.domain}
                    for k, v in result.registry.items()
                }
                session_store[session_id]["clusters"] = [
                    {
                        "name": c.name,
                        "datasets": c.datasets,
                        "relationship": c.relationship,
                        "merge_strategy": c.merge_strategy,
                    }
                    for c in result.clusters
                ]
                session_store[session_id]["dataframe"] = result.unified_dataframe
                session_store[session_id]["unified_context"] = result.unified_context

                from data_scientist_chatbot.app.tools import refresh_sandbox_data

                refresh_sandbox_data(session_id, result.unified_dataframe)

                logging.info(
                    f"[ORCHESTRATOR] Unified {len(datasets_dict)} datasets into {result.unified_dataframe.shape}"
                )
            except Exception as e:
                logging.error(f"[ORCHESTRATOR] Clustering failed: {e}")

        from src.api_utils.session_persistence import session_data_store

        session_data_store.save_session_data(session_id, session_store[session_id])

        return {
            "status": "success",
            "session_id": session_id,
            "message": f"Uploaded {len(uploaded_details)} files successfully",
            "files": uploaded_details,
            "primary_dataset": session_store[session_id].get("filename"),
            "unified_shape": session_store[session_id].get("dataframe").shape
            if session_store[session_id].get("dataframe") is not None
            else None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


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
        session_id = response_data["session_id"]

        if session_id not in session_store:
            session_store[session_id] = {}
        session_store[session_id].update({"dataframe": df})

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


@app.post("/api/heartbeat/{session_id}")
async def client_heartbeat(session_id: str):
    """
    Client heartbeat to indicate the session is still active.
    Frontend should call this every 30 seconds during long operations.
    """
    session_heartbeats[session_id] = time.time()
    return {"success": True, "session_id": session_id, "timestamp": session_heartbeats[session_id]}


@app.get("/api/heartbeat/{session_id}")
async def get_heartbeat_status(session_id: str):
    """Check when the last heartbeat was received for a session."""
    last_heartbeat = session_heartbeats.get(session_id)
    if last_heartbeat:
        elapsed = time.time() - last_heartbeat
        return {
            "session_id": session_id,
            "last_heartbeat": last_heartbeat,
            "elapsed_seconds": elapsed,
            "active": elapsed < 60,
        }
    return {"session_id": session_id, "last_heartbeat": None, "active": False}


@app.get("/api/agent/chat-stream")
async def agent_chat_stream(
    message: str,
    session_id: str,
    web_search_mode: bool = False,
    search_provider: str = "duckduckgo",
    search_api_key: Optional[str] = None,
    search_url: Optional[str] = None,
    token_streaming: bool = True,
    thinking_mode: bool = False,
    regenerate: bool = False,
    message_id: str = None,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Stream agent chat responses with dynamic status updates and optional thinking mode."""
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent service not available")

    user_id = current_user.id if current_user else "anonymous"

    if session_id not in session_store:
        from src.routers.session_router import get_session_db

        db_conn = get_session_db()
        try:
            cursor = db_conn.execute(
                "SELECT session_id, user_id, created_at FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                session_owner = row[1]
                if (
                    session_owner
                    and session_owner != "anonymous"
                    and user_id != "anonymous"
                    and session_owner != user_id
                ):
                    raise HTTPException(status_code=403, detail="Not authorized to access this session")
                session_store[session_id] = {
                    "session_id": row[0],
                    "user_id": row[1],
                    "created_at": datetime.fromisoformat(row[2]) if row[2] else datetime.now(),
                }
                logger.info(f"Loaded session {session_id} from database")
            else:
                session_store[session_id] = {"session_id": session_id, "user_id": user_id}
                logger.info(f"Created new session {session_id} for user {user_id}")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to load session from database: {e}")
            raise HTTPException(status_code=500, detail="Failed to load session")
        finally:
            db_conn.close()
    else:
        session_owner = session_store[session_id].get("user_id")
        if session_owner and session_owner != "anonymous" and user_id != "anonymous" and session_owner != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to access this session")

    from fastapi.responses import StreamingResponse

    async def generate_events():
        session_store[session_id]["web_search_mode"] = web_search_mode
        session_store[session_id]["token_streaming"] = token_streaming
        session_store[session_id]["thinking_mode"] = thinking_mode
        if web_search_mode:
            session_store[session_id]["search_config"] = {
                "provider": search_provider,
                "api_key": search_api_key or "",
                "url": search_url or "",
            }

        if create_enhanced_agent_executor is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent service not available'})}\n\n"
            return

        agent = get_or_create_agent_session(session_id, agent_sessions, create_enhanced_agent_executor)

        full_response = ""

        async for event_data in stream_agent_chat(
            message, session_id, agent, session_store, agent_sessions, regenerate, message_id
        ):
            yield event_data
            try:
                if event_data.startswith("data: "):
                    data = json.loads(event_data[6:])
                    if data.get("type") == "final_response":
                        full_response = data.get("response", "")
            except Exception:
                pass

        if full_response and len(full_response) > 20:

            async def run_auto_name_inline(sid, msg, resp):
                try:
                    from src.routers.session_router import get_session_db
                    from data_scientist_chatbot.app.core.agent_factory import create_brain_agent

                    db_conn = get_session_db()
                    try:
                        cursor = db_conn.execute(
                            "SELECT title, user_id, is_manually_named FROM sessions WHERE session_id = ?", (sid,)
                        )
                        row = cursor.fetchone()
                        current_title = row[0] if row else None
                        user_id = row[1] if row else "anonymous"
                        is_manually_named = row[2] if row and len(row) > 2 else 0
                        print(
                            f"DEBUG: Auto-name check - Session: {sid}, Title: {current_title}, User: {user_id}, Manual: {is_manually_named}"
                        )
                    finally:
                        db_conn.close()

                    if is_manually_named:
                        print(f"INFO: Skipping auto-name for session {sid} (manually named)")
                        return

                    if not current_title or current_title == "New Chat":
                        llm = create_brain_agent()
                        if resp and len(resp) > 50:
                            agent_preview = resp[:500] if len(resp) > 500 else resp
                            prompt = f"""Generate a concise, descriptive title (4-8 words) for a data science conversation.

User asked: "{msg}"

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
                            prompt = f"""Generate a descriptive title (4-8 words) for: "{msg}"

Focus on the topic, not the question. Return ONLY the title. No quotes. No periods."""

                        response = await llm.ainvoke(prompt)
                        title = response.content.strip().strip('"').strip("'")

                        if len(title) > 60:
                            title = title[:57] + "..."

                        db_conn = get_session_db()
                        try:
                            db_conn.execute(
                                "UPDATE sessions SET title = ?, last_updated = ? WHERE session_id = ? AND user_id = ?",
                                (title, datetime.now().isoformat(), sid, user_id),
                            )
                            db_conn.commit()
                            print(f"INFO: Auto-named session {sid} to '{title}'")
                        finally:
                            db_conn.close()
                except Exception as e:
                    print(f"ERROR: Auto-naming failed: {e}")

            asyncio.create_task(run_auto_name_inline(session_id, message, full_response))

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

        result = await perform_checkpoint_maintenance(force_cleanup=False)
        status_code = 200 if result.get("status") in ["healthy", "unavailable"] else 503

        return JSONResponse(content=result, status_code=status_code)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/checkpoint/maintenance")
async def trigger_checkpoint_maintenance():
    try:
        from data_scientist_chatbot.app.core.graph_builder import perform_checkpoint_maintenance

        result = await perform_checkpoint_maintenance(force_cleanup=True)

        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/checkpoint/cleanup")
async def cleanup_old_checkpoints_endpoint(retention_days: int = 7):
    try:
        from data_scientist_chatbot.app.core.graph_builder import perform_checkpoint_maintenance

        result = await perform_checkpoint_maintenance(force_cleanup=True)

        return JSONResponse(
            content=result,
            status_code=200,
        )
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve the Single Page Application (SPA)"""
    if full_path.startswith("api"):
        raise HTTPException(status_code=404, detail="Not found")

    # Check if file exists in static folder (e.g. favicon.ico)
    static_file = Path("static") / full_path
    if static_file.is_file():
        return FileResponse(static_file)

    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
