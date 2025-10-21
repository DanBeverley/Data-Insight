import io
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd
import joblib
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import ToolMessage
import uuid
import re
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
    print("✅ Chat functionality loaded successfully")
except Exception as e:
    CHAT_AVAILABLE = False
    create_agent_executor = None
    print(f"⚠️  Chat functionality not available: {e}")
    import traceback
    traceback.print_exc()
from .common.data_ingestion import ingest_data, ingest_from_url
from .data_quality.validator import DataQualityValidator
from .intelligence.data_profiler import IntelligentDataProfiler
from .visualization.data_explorer import IntelligentDataExplorer, VisualizationRequest
from .api_utils.session_management import clean_checkpointer_state, get_or_create_agent_session, validate_session
from .api_utils.agent_response import extract_plot_urls, extract_agent_response
from .api_utils.data_ingestion import process_dataframe_ingestion, create_intelligence_summary, load_dataframe_to_sandbox
from .api_utils.helpers import convert_pandas_output_to_html, create_agent_input, run_agent_task, create_workflow_status_context
from .api_utils.models import DataIngestionRequest, ProfilingRequest, FeatureRecommendationRequest, ChatMessage, ChatResponse, AgentChatRequest
from .api_utils.upload_handler import enhance_with_agent_profile, load_data_to_agent_sandbox
from .api_utils.artifact_handler import handle_artifact_download
from .api_utils.streaming_service import stream_agent_chat
from .routers import data_router, session_router
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

app = FastAPI(
    title="DnA",
    description="",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
data_explorer = IntelligentDataExplorer()
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

app.include_router(data_router)
app.include_router(session_router)

session_store = {}

import builtins
builtins._session_store = session_store

data_profiler = IntelligentDataProfiler()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat-first landing interface."""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())

@app.post("/api/agent/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """Handle chat messages with the AI agent"""
    try:
        session_id = chat_request.session_id or str(uuid.uuid4())
        agent = get_session_agent(session_id)
        
        if not agent:
            if not CHAT_AVAILABLE:
                return ChatResponse(
                    status="error",
                    response="I apologize, but the chat functionality is currently unavailable. This appears to be due to missing dependencies (like dotenv, langchain, etc.). Please run 'pip install -r requirements.txt' to install all required packages.",
                    plots=[]
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to initialize chat agent for this session")
        
        input_state = {
            "messages": [("user", chat_request.message)],
            "session_id": session_id,
            "python_executions": 0
        }
        
        final_response_content = ""
        all_plot_urls = []
        
        try:
            print(f"FastAPI DEBUG: Starting agent stream for session {session_id}")
            for chunk in agent.stream(input_state, config={"configurable": {"thread_id": session_id}, "recursion_limit": 10}):
                print(f"FastAPI DEBUG: Received chunk with keys: {list(chunk.keys())}")
                
                if "action" in chunk:
                    print("FastAPI DEBUG: Processing tool execution")
                    tool_messages = chunk["action"]["messages"]
                    print(f"FastAPI DEBUG: Found {len(tool_messages)} tool messages")
                    
                    for i, tool_msg in enumerate(tool_messages):
                        print(f"FastAPI DEBUG: Tool message {i} type: {type(tool_msg)}")
                        print(f"FastAPI DEBUG: Tool message {i} attributes: {dir(tool_msg)}")
                        
                        # Check different possible attribute names for tool identification
                        tool_name = getattr(tool_msg, 'name', None) or getattr(tool_msg, 'tool_name', None) or getattr(tool_msg, 'type', None)
                        print(f"FastAPI DEBUG: Tool message {i} identifier: {tool_name}")
                        
                        content = str(getattr(tool_msg, 'content', ''))
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
                                    print(f"FastAPI DEBUG: ✅ Added plot URL: {url}")
                        
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
                                        print(f"FastAPI DEBUG: ✅ Added plot URL from PNG: {url}")
                                        
                if "agent" in chunk:
                    print("FastAPI DEBUG: Processing agent response")
                    final_message = chunk["agent"]["messages"][-1]
                    if final_message:
                        final_response_content = final_message.content
                        print(f"FastAPI DEBUG: Final response: {final_response_content[:100]}...")
            
            if not final_response_content:
                final_state = agent.invoke(input_state, config={"configurable": {"thread_id": session_id}, "recursion_limit": 10})
                final_response_content = final_state["messages"][-1].content
                
        except Exception as agent_error:
            print(f"Agent streaming error: {str(agent_error)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Agent error: {str(agent_error)}")
        
        return ChatResponse(
            status="success",
            response=final_response_content,
            plots=all_plot_urls
        )
        
    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...), enable_profiling: bool = Form(True), session_id: str = Form(...)):
    """Upload and validate dataset with intelligent profiling."""
    try:
        if not session_id or not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")

        contents = await file.read()
        df = ingest_data(io.BytesIO(contents), filename=file.filename)

        if df is None:
            raise HTTPException(status_code=400, detail="Failed to parse uploaded file")

        response_data = process_dataframe_ingestion(
            df, session_id, {"filename": file.filename}, enable_profiling, data_profiler, session_store
        )
        session_id = response_data["session_id"]

        # Enhance with agent profile and load to sandbox
        import builtins
        if not hasattr(builtins, '_session_store'):
            builtins._session_store = {}

        enhance_with_agent_profile(df, session_id, file.filename, response_data)

        if AGENT_AVAILABLE:
            load_data_to_agent_sandbox(df, session_id, agent_sessions, create_enhanced_agent_executor, response_data)

        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.post("/api/privacy/consent")
async def handle_privacy_consent(request: dict):
    """Handle user consent for privacy protection"""
    try:
        session_id = request.get('session_id')
        apply_protection = request.get('apply_protection', False)

        if not session_id or session_id not in session_store:
            raise HTTPException(status_code=404, detail="Session not found")

        import builtins
        if hasattr(builtins, '_session_store') and session_id in builtins._session_store:
            session_data = builtins._session_store[session_id]

            if apply_protection:
                # Apply privacy protection using PrivacyEngine
                from security.privacy_engine import PrivacyEngine
                privacy_engine = PrivacyEngine()

                original_df = session_data['dataframe']
                protected_df, transformations = privacy_engine.apply_k_anonymity(
                    original_df,
                    privacy_engine._auto_detect_quasi_identifiers(original_df)
                )

                # Update stored dataframe with protected version
                session_data['dataframe'] = protected_df
                session_data['privacy_applied'] = True
                session_data['privacy_transformations'] = transformations

                # Update session store as well
                session_store[session_id]['dataframe'] = protected_df

                return {
                    "status": "success",
                    "message": "Privacy protection applied successfully",
                    "protection_applied": True
                }
            else:
                # Mark that user declined protection
                session_data['privacy_applied'] = False
                return {
                    "status": "success",
                    "message": "Continuing without privacy protection",
                    "protection_applied": False
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
            df, request.session_id, {"source": request.url, "source_url": request.url},
            request.enable_profiling, data_profiler, session_store
        )
        session_id = response_data["session_id"]
        import builtins
        if not hasattr(builtins, '_session_store'):
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
        
        # Check for long-running tasks (model building, complex analysis)
        is_long_task = any(keyword in request.message.lower() for keyword in 
                         ['build model', 'train model', 'complex analysis', 'deep analysis', 'correlation analysis'])
        
        if request.async_execution or is_long_task:
            # Run asynchronously
            task_id = str(uuid.uuid4())
            
            # Submit task to thread pool
            future = task_executor.submit(run_agent_task, agent, user_message, session_id)
            async_tasks[task_id] = {
                "future": future,
                "status": "running",
                "session_id": session_id,
                "created_at": time.time()
            }
            
            if performance_monitor:
                performance_monitor.record_metric(
                    session_id=session_id,
                    metric_name="async_task_created",
                    value=1.0,
                    context={"task_id": task_id}
                )
            
            return {
                "status": "async",
                "task_id": task_id,
                "message": "Task started. Use /api/agent/task-status to check progress.",
                "session_id": session_id
            }
        else:
            # Run synchronously
            print(f"DEBUG: Using agent type: {type(agent)}")
            response = agent.invoke(create_agent_input(user_message, session_id))
            
            # Extract plots and response from messages
            messages = response.get('messages', [])
            content, plots = extract_agent_response(messages, recent_count=3)

            print(f"DEBUG: Final response content: {content[:100]}")
            print(f"DEBUG: Response has {len(plots)} plots")

            return {
                "status": "success",
                "response": content,
                "session_id": session_id,
                "plots": plots
            }
    except Exception as e:
        return {"status": "error", "detail": f"Agent chat error: {str(e)}"}

@app.get("/api/agent/chat-stream")
async def agent_chat_stream(message: str, session_id: str, web_search_enabled: bool = False):
    """Stream agent chat responses with dynamic status updates."""
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent service not available")

    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found. Please create a session first using /api/sessions/new")

    from fastapi.responses import StreamingResponse

    async def generate_events():
        global status_agent_runnable

        session_store[session_id]["web_search_enabled"] = web_search_enabled

        if create_enhanced_agent_executor is None:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Agent service not available'})}\n\n"
            return

        agent = get_or_create_agent_session(session_id, agent_sessions, create_enhanced_agent_executor)

        if status_agent_runnable is None:
            try:
                from data_scientist_chatbot.app.core.agent_factory import create_status_agent
                status_agent_runnable = create_status_agent()
            except Exception as e:
                print(f"WARNING: Failed to initialize status agent: {e}")

        async for event_data in stream_agent_chat(message, session_id, agent, session_store, status_agent_runnable):
            yield event_data

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
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
                return {
                    "status": "completed",
                    "task_id": task_id,
                    "result": result["response"]
                }
            else:
                async_tasks[task_id]["status"] = "failed"
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": result["error"]
                }
        except Exception as e:
            async_tasks[task_id]["status"] = "failed"
            return {
                "status": "failed",
                "task_id": task_id,
                "error": str(e)
            }
    else:
        return {
            "status": "running",
            "task_id": task_id,
            "message": "Task is still processing..."
        }

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
        if filename.endswith('.png'):
            media_type = "image/png"
        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            media_type = "image/jpeg"
        elif filename.endswith('.html'):
            media_type = "text/html"
        elif filename.endswith('.csv'):
            media_type = "text/csv"
        else:
            media_type = "application/octet-stream"
            
        return FileResponse(
            artifact_path, 
            media_type=media_type,
            filename=filename
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving artifact: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Enhanced application health check endpoint with performance monitoring."""
    health_status = {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    if DATABASE_AVAILABLE:
        try:
            db_manager = get_database_manager()
            db_health = db_manager.execute_health_check()
            health_status["database"] = "healthy" if db_health else "degraded"
        except:
            health_status["database"] = "unavailable"
    else:
        health_status["database"] = "not_configured"
    
    health_status["agent_service"] = "available" if AGENT_AVAILABLE else "unavailable"
    
    # Add performance monitoring status
    if AGENT_AVAILABLE and performance_monitor:
        try:
            cache_stats = performance_monitor.cache.get_stats()
            health_status["performance_monitoring"] = "active"
            health_status["cache_utilization"] = f"{cache_stats['size']}/{cache_stats['max_size']}"
            health_status["active_agent_sessions"] = len(agent_sessions)
            health_status["thread_pool_workers"] = task_executor._max_workers if task_executor else 0
        except:
            health_status["performance_monitoring"] = "degraded"
    else:
        health_status["performance_monitoring"] = "unavailable"
    
    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
