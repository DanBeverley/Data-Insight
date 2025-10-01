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

def convert_pandas_output_to_html(output_text):
    if not output_text or not isinstance(output_text, str):
        return output_text
    lines = output_text.strip().split('\n')
    shape_line = next((line for line in lines if 'Shape:' in line), '')
    columns_line = next((line for line in lines if 'Columns:' in line), '')
    
    if shape_line and columns_line:
        columns_text = columns_line.split(': ')[1] if ': ' in columns_line else 'N/A'
        return f"📊 Dataset loaded successfully!\n{shape_line}\nColumns: {columns_text}"
    return "📊 Dataset loaded successfully!"
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
    if CHAT_AVAILABLE and session_id not in session_agents:
        # Ensure clean checkpointer state for new agent session
        try:
            from data_scientist_chatbot.app.core.graph_builder import memory
            if memory:
                memory.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
                memory.conn.commit()
                print(f"DEBUG: Cleaned checkpointer state for new agent session {session_id}")
        except Exception as e:
            print(f"WARNING: Failed to clean checkpointer state for agent session {session_id}: {e}")

        session_agents[session_id] = create_enhanced_agent_executor(session_id)
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

def create_intelligence_summary(intelligence_profile: dict) -> dict:
    """Extract intelligence summary from profiling results"""
    column_profiles = intelligence_profile.get('column_profiles', {})
    domain_analysis = intelligence_profile.get('domain_analysis', {})

    semantic_types = {col: profile.semantic_type.value
                    for col, profile in column_profiles.items()}

    detected_domains = domain_analysis.get('detected_domains', [])
    primary_domain = detected_domains[0] if detected_domains else None

    return {
        "semantic_types": semantic_types,
        "primary_domain": primary_domain.get('domain') if primary_domain else 'unknown',
        "domain_confidence": primary_domain.get('confidence', 0) if primary_domain else 0,
        "domain_analysis": domain_analysis,
        "key_insights": intelligence_profile.get('overall_recommendations', [])[:3] if intelligence_profile.get('overall_recommendations') else [],
        "relationships_found": len(intelligence_profile.get('relationship_analysis', {}).get('relationships', [])),
        "profiling_completed": True
    }
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

class DataIngestionRequest(BaseModel):
    url: str
    data_type: str = "csv"
    enable_profiling: bool = True
    session_id: Optional[str] = None
session_store = {}

import builtins
builtins._session_store = session_store

data_profiler = IntelligentDataProfiler()

class ProfilingRequest(BaseModel):
    deep_analysis: bool = True
    include_relationships: bool = True
    include_domain_detection: bool = True

class FeatureRecommendationRequest(BaseModel):
    target_column: Optional[str] = None
    max_recommendations: int = 10
    priority_filter: Optional[str] = None  # 'high', 'medium', 'low'

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    status: str
    response: str
    plots: List[str] = []

class AgentChatRequest(BaseModel):
    message: str
    session_id: str
    async_execution: bool = False

def create_agent_input(message: str, session_id: str) -> Dict[str, Any]:
    """Create input state for agent execution"""
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id
    }

def run_agent_task(agent, message: str, session_id: str) -> Dict[str, Any]:
    """Run agent task in background thread"""
    try:
        response = agent.invoke(create_agent_input(message, session_id))
        messages = response.get('messages', [])
        final_message = messages[-1] if messages else None
        content = final_message.content if final_message else "Task completed"

        return {
            "success": True,
            "response": content,
            "session_id": session_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }

def create_workflow_status_context(workflow_context: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """Create context for status agent"""
    return {
        "user_goal": workflow_context.get("user_goal", ""),
        "current_agent": workflow_context.get("current_agent", ""),
        "current_action": workflow_context.get("current_action", ""),
        "event_type": event.get("event", ""),
        "node_name": event.get("name", "")
    }

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
async def upload_data(file: UploadFile = File(...), enable_profiling: bool = Form(True), session_id: str = Form(None)):
    """Upload and validate dataset with intelligent profiling."""
    try:
        contents = await file.read()
        df = ingest_data(io.BytesIO(contents), filename=file.filename)

        if df is None:
            raise HTTPException(status_code=400, detail="Failed to parse uploaded file")

        # Validate data quality
        validator = DataQualityValidator(df)
        validation_report = validator.validate()

        # Use provided session ID or create new one
        session_id = session_id or str(uuid.uuid4())
        session_data = {
            "dataframe": df,
            "validation_report": validation_report,
            "filename": file.filename,
            "created_at": datetime.now().isoformat()
        }
        
        response_data = {
            "session_id": session_id,
            "status": "success", 
            "filename": file.filename,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "validation": {
                "is_valid": validation_report.is_valid,
                "issues": [
                    {
                        "type": check.name,
                        "description": check.message,
                        "severity": "high" if "missing" in check.message.lower() or "null" in check.message.lower() else "medium",
                        "affected_columns": list(check.details.keys()) if check.details else [],
                        "details": check.details,
                        "passed": check.passed
                    }
                    for check in validation_report.checks if not check.passed
                ]
            }
        }
        
        if enable_profiling:
            try:
                intelligence_profile = data_profiler.profile_dataset(df)
                
                # Extract key insights for response
                intelligence_summary = create_intelligence_summary(intelligence_profile)
                response_data["intelligence_summary"] = intelligence_summary
                session_data["intelligence_profile"] = intelligence_profile
                
            except Exception as prof_e:
                response_data["intelligence_summary"] = {
                    "profiling_completed": False,
                    "profiling_error": str(prof_e),
                    "key_insights": [],
                    "semantic_types": {},
                    "primary_domain": "unknown",
                    "domain_confidence": 0,
                    "domain_analysis": {"detected_domains": []},
                    "relationships_found": 0
                }
        
        session_store[session_id] = session_data

        # Store data in builtins for agent access
        import builtins
        if not hasattr(builtins, '_session_store'):
            builtins._session_store = {}
        try:
            from intelligence.hybrid_data_profiler import generate_dataset_profile_for_agent
            data_profile = generate_dataset_profile_for_agent(df, context={'filename': file.filename, 'upload_session': session_id})

            # Check for PII detection in profile metadata
            pii_findings = data_profile.profile_metadata.get('pii_detection')
            if pii_findings and pii_findings.privacy_score < 0.7:
                response_data["pii_detection"] = {
                    "pii_detected": True,
                    "risk_level": pii_findings.risk_level,
                    "privacy_score": round(pii_findings.privacy_score, 2),
                    "reidentification_risk": round(pii_findings.reidentification_risk, 2),
                    "recommendations": pii_findings.recommendations,
                    "requires_consent": True,
                    "message": f"Privacy concerns detected (Risk Level: {pii_findings.risk_level}). Would you like to apply privacy protection before analysis?"
                }

            # Add profiling summary for enhanced upload message
            response_data["profiling_summary"] = {
                "quality_score": round(data_profile.quality_assessment.get('overall_score', 0), 1) if data_profile.quality_assessment.get('overall_score') else None,
                "anomalies_detected": data_profile.anomaly_detection.get('summary', {}).get('total_anomalies', 0),
                "profiling_time": round(data_profile.profile_metadata.get('profiling_duration', 0), 2)
            }

            builtins._session_store[session_id] = {
                'dataframe': df,
                'data_profile': data_profile,
                'filename': file.filename
            }

            # Capture to knowledge graph
            try:
                from knowledge_graph.service import SessionDataStorage
                storage = SessionDataStorage()
                storage.add_session(session_id, {
                    'dataset_info': {
                        'filename': file.filename,
                        'shape': df.shape,
                        'columns': df.columns.tolist(),
                        'dtypes': df.dtypes.to_dict(),
                        'domain': data_profile.dataset_insights.detected_domains
                    }
                })
            except Exception as e:
                logging.warning(f"Failed to store in knowledge graph: {e}")
        except Exception as e:
            logging.warning(f"Failed to generate dataset profile: {e}")
            builtins._session_store[session_id] = {"dataframe": df}

        # Initialize agent session and load data into sandbox if available  
        if AGENT_AVAILABLE:
            try:
                from data_scientist_chatbot.app.tools import execute_python_in_sandbox
                
                if session_id not in agent_sessions:
                    agent_sessions[session_id] = create_enhanced_agent_executor(session_id)
                
                # Convert dataframe to CSV string for direct transfer to sandbox
                csv_data = df.to_csv(index=False)
                
                # Load data directly into the stateful sandbox
                load_code = f"""
                            import pandas as pd
                            import numpy as np
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            from io import StringIO

                            # Create dataframe from CSV data
                            csv_data = '''{csv_data}'''
                            df = pd.read_csv(StringIO(csv_data))
                            print(f"Dataset loaded successfully!")
                            print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
                            print(f"Columns: {{list(df.columns)}}")
                            print("First 5 rows:")
                            print(df.head())
                            """
                
                print(f"DEBUG: Loading data into sandbox for session {session_id}")
                print(f"DEBUG: Load code length: {len(load_code)} chars")
                load_result = execute_python_in_sandbox(load_code, session_id)
                print(f"DEBUG: Sandbox load result success: {load_result['success']}")
                if not load_result['success']:
                    print(f"DEBUG: Sandbox load error: {load_result.get('stderr', 'No error details')}")
                if load_result["success"]:
                    clean_output = load_result['stdout']
                    print(f"DEBUG: Sandbox load stdout: {clean_output[:200]}...")
                    html_output = convert_pandas_output_to_html(clean_output)
                    enhanced_message = f"\n\n{html_output}"

                    # Add profiling summary if available
                    profile_summary = response_data.get("profiling_summary", {})
                    if profile_summary.get("quality_score"):
                        enhanced_message += f"\n📊 Quality Score: {profile_summary['quality_score']}/100"
                    if profile_summary.get("anomalies_detected"):
                        enhanced_message += f"\n⚠️ Detected {profile_summary['anomalies_detected']} anomalies"
                    if profile_summary.get("profiling_time"):
                        enhanced_message += f"\n⏱️ Analysis completed in {profile_summary['profiling_time']}s"

                    response_data["agent_analysis"] = enhanced_message
                    response_data["agent_session_id"] = session_id

                    save_code = f"""
                                # Save dataset to common filenames that agents might use
                                df.to_csv('dataset.csv', index=False)
                                df.to_csv('data.csv', index=False)
                                df.to_csv('ds.csv', index=False)
                                print("Dataset saved as dataset.csv, data.csv, and ds.csv")
                                """
                    save_result = execute_python_in_sandbox(save_code, session_id)
                    print(f"DEBUG: Dataset save result: {save_result['success']}")
                else:
                    error_msg = load_result.get('stderr', 'Unknown error')
                    print(f"ERROR: Sandbox data loading failed: {error_msg}")
                    response_data["agent_analysis"] = f"⚠️ Data loaded but with issues: {error_msg}"
                    response_data["agent_session_id"] = session_id 
            except Exception as agent_e:
                logging.warning(f"Agent data loading failed: {agent_e}")
                response_data["agent_analysis"] = None

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
        
        # Validate data quality
        validator = DataQualityValidator(df)
        validation_report = validator.validate()
        
        # Use provided session ID or create new one
        session_id = request.session_id or str(uuid.uuid4())
        session_data = {
            "dataframe": df,
            "validation_report": validation_report,
            "source_url": request.url,
            "created_at": datetime.now().isoformat()
        }
        
        response_data = {
            "session_id": session_id,
            "status": "success",
            "source": request.url,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "validation": {
                "is_valid": validation_report.is_valid,
                "issues": [
                    {
                        "type": check.name,
                        "description": check.message,
                        "severity": "high" if "missing" in check.message.lower() or "null" in check.message.lower() else "medium",
                        "affected_columns": list(check.details.keys()) if check.details else [],
                        "details": check.details,
                        "passed": check.passed
                    }
                    for check in validation_report.checks if not check.passed
                ]
            }
        }
        
        if request.enable_profiling:
            try:
                intelligence_profile = data_profiler.profile_dataset(df)
                
                # Extract key insights for response
                intelligence_summary = create_intelligence_summary(intelligence_profile)
                response_data["intelligence_summary"] = intelligence_summary
                session_data["intelligence_profile"] = intelligence_profile
                
            except Exception as prof_e:
                response_data["intelligence_summary"] = {
                    "profiling_completed": False,
                    "profiling_error": str(prof_e),
                    "key_insights": [],
                    "semantic_types": {},
                    "primary_domain": "unknown",
                    "domain_confidence": 0,
                    "domain_analysis": {"detected_domains": []},
                    "relationships_found": 0
                }
        
        session_store[session_id] = session_data
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

@app.get("/api/data/{session_id}/preview")
async def get_data_preview(session_id: str, rows: int = 10):
    """Get data preview for a session."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = session_store[session_id]["dataframe"]
    preview_data = df.head(rows).to_dict('records')
    
    return {
        "data": preview_data,
        "shape": df.shape,
        "columns": df.columns.tolist()
    }

@app.post("/api/data/{session_id}/eda")
async def generate_eda(session_id: str):
    """Generate comprehensive EDA report."""
    if session_id not in session_store:
        return {"status": "error", "detail": "Session not found"}
    
    try:
        df = session_store[session_id]["dataframe"]
        
        eda_report = {
            "basic_info": {
                "shape": list(df.shape),
                "columns": df.columns.tolist(),
                "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
                "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
                "missing_percentage": {k: float(v) for k, v in (df.isnull().sum() / len(df) * 100).to_dict().items()}
            },
            "numeric_summary": {},
            "categorical_summary": {},
            "correlations": {}
        }
        
        # Numeric analysis
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            desc = df[numeric_cols].describe()
            eda_report["numeric_summary"] = {col: {k: float(v) for k, v in desc[col].to_dict().items()} for col in numeric_cols}
            
            # Simple correlation matrix
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                eda_report["correlations"] = {
                    col1: {col2: float(corr_matrix.loc[col1, col2]) if not pd.isna(corr_matrix.loc[col1, col2]) else 0.0 
                           for col2 in numeric_cols} 
                    for col1 in numeric_cols
                }
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:5]
        for col in categorical_cols:
            value_counts = df[col].value_counts().head()
            eda_report["categorical_summary"][col] = {str(k): int(v) for k, v in value_counts.to_dict().items()}
        
        session_store[session_id]["eda_report"] = eda_report
        
        return {
            "status": "success", 
            "report": eda_report,
            "message": "EDA report generated successfully"
        }
    
    except Exception as e:
        return {"status": "error", "detail": f"Error generating EDA: {str(e)}"}



@app.get("/api/data/{session_id}/download/{artifact_type}")
async def download_artifact(session_id: str, artifact_type: str):
    """Download processing artifacts including intelligence reports."""
    if session_id not in session_store:
        return {"status": "error", "detail": "Session not found"}
    
    session_data = session_store[session_id]
    
    try:
        print(f"🔧 Download request: session_id={session_id}, artifact_type={artifact_type}")
        print(f"🔧 Session data keys: {list(session_data.keys())}")
        
        if artifact_type == "data":
            # Check unified pipeline result for processed data
            processed_data = None
            if "processed_data" in session_data:
                processed_data = session_data["processed_data"]
            elif "pipeline_result" in session_data:
                pipeline_result = session_data["pipeline_result"]
                if isinstance(pipeline_result, dict) and "final_data" in pipeline_result:
                    processed_data = pipeline_result["final_data"]
                elif isinstance(pipeline_result, dict) and "processed_data" in pipeline_result:
                    processed_data = pipeline_result["processed_data"]
            
            if processed_data is None:
                return {"status": "error", "detail": "Processed data not found"}
            
            df = processed_data
            if session_data.get("aligned_target") is not None:
                df = df.copy()
                df["target"] = session_data["aligned_target"]
            
            csv_data = df.to_csv(index=False)
            
            return Response(
                content=csv_data.encode(),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=processed_data_{session_id}.csv"}
            )
        
        elif artifact_type == "pipeline":
            if "pipeline_result" not in session_data:
                raise HTTPException(status_code=404, detail="Pipeline not found")
            
            try:
                # Get the pipeline from unified pipeline result
                pipeline_result = session_data["pipeline_result"]
                
                # Try to get pipeline from result
                pipeline = None
                if isinstance(pipeline_result, dict):
                    if 'pipeline' in pipeline_result:
                        pipeline = pipeline_result['pipeline']
                    elif 'final_pipeline' in pipeline_result:
                        pipeline = pipeline_result['final_pipeline']
                
                if pipeline is None:
                    # If no pipeline found, create a placeholder
                    placeholder_data = {
                        "session_id": session_id,
                        "message": "Pipeline object not available - processing may have used a different approach",
                        "pipeline_result_keys": list(pipeline_result.keys()) if isinstance(pipeline_result, dict) else str(type(pipeline_result)),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    return Response(
                        content=json.dumps(placeholder_data, indent=2).encode(),
                        media_type="application/json",
                        headers={"Content-Disposition": f"attachment; filename=pipeline_info_{session_id}.json"}
                    )
                
                try:
                    # Use a temporary file approach instead of BytesIO to avoid memory issues
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
                        joblib.dump(pipeline, tmp_file.name)
                        
                        # Read the file back as bytes
                        with open(tmp_file.name, 'rb') as f:
                            pipeline_bytes = f.read()
                        
                        # Clean up temp file
                        os.unlink(tmp_file.name)
                        
                        return Response(
                            content=pipeline_bytes,
                            media_type="application/octet-stream",
                            headers={
                                "Content-Disposition": f"attachment; filename=pipeline_{session_id}.joblib",
                                "Content-Length": str(len(pipeline_bytes))
                            }
                        )
                except Exception as serialize_error:
                    # Return pipeline info as JSON instead
                    pipeline_info = {
                        "session_id": session_id,
                        "pipeline_type": str(type(pipeline)),
                        "pipeline_string": str(pipeline)[:1000] + "..." if len(str(pipeline)) > 1000 else str(pipeline),
                        "serialization_error": str(serialize_error),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    return Response(
                        content=json.dumps(pipeline_info, indent=2).encode(),
                        media_type="application/json",
                        headers={"Content-Disposition": f"attachment; filename=pipeline_info_{session_id}.json"}
                    )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Pipeline serialization failed: {str(e)}")
        
        elif artifact_type == "lineage":
            if "pipeline_result" not in session_data:
                raise HTTPException(status_code=404, detail="Lineage report not found")
            
            try:
                pipeline_result = session_data["pipeline_result"]
                
                # Try different ways to get lineage data
                lineage = None
                if isinstance(pipeline_result, dict):
                    if 'lineage_report' in pipeline_result:
                        lineage = pipeline_result['lineage_report']
                    elif 'lineage' in pipeline_result:
                        lineage = pipeline_result['lineage']
                    elif 'execution_summary' in pipeline_result:
                        # Generate lineage from execution summary
                        execution_summary = pipeline_result['execution_summary']
                        lineage = {
                            "pipeline_execution": "completed",
                            "timestamp": datetime.now().isoformat(),
                            "total_stages": execution_summary.get('total_stages', 0),
                            "successful_stages": execution_summary.get('successful_stages', 0),
                            "execution_time": execution_summary.get('total_time', 0),
                            "session_id": session_id,
                            "stages_executed": list(pipeline_result.get('results', {}).keys())
                        }
                    else:
                        # Generate basic lineage from available data
                        lineage = {
                            "pipeline_execution": "completed",
                            "timestamp": datetime.now().isoformat(),
                            "steps_executed": list(pipeline_result.keys()) if isinstance(pipeline_result, dict) else [],
                            "session_id": session_id
                        }
                
                lineage_json = json.dumps(lineage, indent=2, default=str)
                
                return Response(
                    content=lineage_json.encode(),
                    media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename=lineage_report_{session_id}.json"}
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Lineage report generation failed: {str(e)}")
        
        elif artifact_type == "intelligence":
            if "intelligence_profile" not in session_data:
                raise HTTPException(status_code=404, detail="Intelligence profile not found")
            
            intelligence_profile = session_data["intelligence_profile"]
            
            # Convert to JSON-serializable format
            serializable_profile = {}
            
            # Convert column profiles
            if 'column_profiles' in intelligence_profile:
                serializable_profile['column_profiles'] = {}
                for col, profile in intelligence_profile['column_profiles'].items():
                    serializable_profile['column_profiles'][col] = {
                        'semantic_type': profile.semantic_type.value,
                        'confidence': profile.confidence,
                        'evidence': profile.evidence,
                        'recommendations': profile.recommendations
                    }
            
            for key in ['domain_analysis', 'relationship_analysis', 'overall_recommendations']:
                if key in intelligence_profile:
                    serializable_profile[key] = intelligence_profile[key]
            
            intelligence_json = json.dumps(serializable_profile, indent=2, default=str)
            
            return Response(
                content=intelligence_json.encode(),
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=intelligence_report_{session_id}.json"}
            )
        
        elif artifact_type == "pipeline-metadata":
            if "pipeline_result" not in session_data:
                raise HTTPException(status_code=404, detail="Pipeline metadata not found")
            
            try:
                pipeline_result = session_data["pipeline_result"]
                print(f"🔧 Pipeline result type: {type(pipeline_result)}")
                print(f"🔧 Pipeline result keys: {list(pipeline_result.keys()) if isinstance(pipeline_result, dict) else 'Not a dict'}")
                
                # Create a safe, serializable version of the metadata
                safe_metadata = {}
                
                if isinstance(pipeline_result, dict):
                    for key, value in pipeline_result.items():
                        try:
                            print(f"🔧 Processing key: {key}, value type: {type(value)}")
                            
                            if key in ['pipeline', 'final_pipeline']:
                                safe_metadata[key] = f"<Pipeline object of type {type(value)}>"
                            elif key in ['dataframe', 'processed_data', 'final_data', 'enhanced_data']:
                                if hasattr(value, 'shape'):
                                    safe_metadata[key] = f"<DataFrame with shape {value.shape}>"
                                else:
                                    safe_metadata[key] = f"<Data object of type {type(value)}>"
                            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
                                safe_metadata[key] = str(value)[:500] + "..." if len(str(value)) > 500 else str(value)
                            else:
                                try:
                                    json.dumps(value) 
                                    safe_metadata[key] = value
                                except:
                                    safe_metadata[key] = str(value)[:500] + "..." if len(str(value)) > 500 else str(value)
                        except Exception as key_error:
                            print(f"❌ Error processing key {key}: {str(key_error)}")
                            safe_metadata[key] = f"<Error processing key: {str(key_error)}>"
                else:
                    safe_metadata = {
                        "pipeline_result_type": str(type(pipeline_result)),
                        "pipeline_result_str": str(pipeline_result)[:1000] + "..." if len(str(pipeline_result)) > 1000 else str(pipeline_result)
                    }
                
                safe_metadata["session_id"] = session_id
                safe_metadata["export_timestamp"] = datetime.now().isoformat()
                
                metadata_json = json.dumps(safe_metadata, indent=2, default=str)
                
                return Response(
                    content=metadata_json.encode(),
                    media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename=pipeline_metadata_{session_id}.json"}
                )
            except Exception as e:
                print(f"❌ Metadata serialization error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Metadata export failed: {str(e)}")
        
        elif artifact_type == "enhanced-data":
            enhanced_df = None
            
            if "enhanced_data" in session_data:
                enhanced_df = session_data["enhanced_data"]
            elif "pipeline_result" in session_data:
                pipeline_result = session_data["pipeline_result"]
                if isinstance(pipeline_result, dict):
                    if "enhanced_data" in pipeline_result:
                        enhanced_df = pipeline_result["enhanced_data"]
                    elif "final_data" in pipeline_result:
                        enhanced_df = pipeline_result["final_data"] 
                    elif "processed_data" in pipeline_result:
                        enhanced_df = pipeline_result["processed_data"]
            
            if enhanced_df is None:
                # Fallback to regular processed data
                if "processed_data" in session_data:
                    enhanced_df = session_data["processed_data"]
                else:
                    raise HTTPException(status_code=404, detail="Enhanced data not found")
            
            try:
                csv_data = enhanced_df.to_csv(index=False)
                
                return Response(
                    content=csv_data.encode(),
                    media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=enhanced_data_{session_id}.csv"}
                )
            except Exception as e:
                print(f"❌ Enhanced data download error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Enhanced data export failed: {str(e)}")
        
        else:
            raise HTTPException(status_code=400, detail="Invalid artifact type")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

@app.post("/api/data/{session_id}/profile")
async def generate_intelligence_profile(session_id: str, request: ProfilingRequest):
    """Generate comprehensive intelligent data profile on-demand."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        df = session_store[session_id]["dataframe"]
        
        # Generate or retrieve intelligence profile
        if "intelligence_profile" in session_store[session_id] and not request.deep_analysis:
            intelligence_profile = session_store[session_id]["intelligence_profile"]
        else:
            intelligence_profile = data_profiler.profile_dataset(df)
            session_store[session_id]["intelligence_profile"] = intelligence_profile
        
        # Filter response based on request parameters
        response_data = {}
        
        # Always include column profiles
        column_profiles = intelligence_profile.get('column_profiles', {})
        response_data['column_profiles'] = {
            col: {
                'semantic_type': profile.semantic_type.value,
                'confidence': profile.confidence,
                'evidence': profile.evidence,
                'recommendations': profile.recommendations
            }
            for col, profile in column_profiles.items()
        }
        
        # Include domain analysis if requested
        if request.include_domain_detection:
            domain_analysis = intelligence_profile.get('domain_analysis', {})
            response_data['domain_analysis'] = domain_analysis
        
        # Include relationship analysis if requested
        if request.include_relationships:
            relationship_analysis = intelligence_profile.get('relationship_analysis', {})
            response_data['relationship_analysis'] = relationship_analysis
        
        # Always include overall recommendations
        response_data['overall_recommendations'] = intelligence_profile.get('overall_recommendations', [])
        
        # Add profiling metadata
        response_data['profiling_metadata'] = {
            'deep_analysis': request.deep_analysis,
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'profile_timestamp': datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "intelligence_profile": response_data,
            "session_id": session_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling error: {str(e)}")

@app.get("/api/data/{session_id}/feature-recommendations")
async def get_feature_recommendations(session_id: str, 
                                    target_column: Optional[str] = None,
                                    max_recommendations: int = 10,
                                    priority_filter: Optional[str] = None):
    """Get AI-powered feature engineering recommendations."""
    if session_id not in session_store:
        return {"status": "error", "detail": "Session not found"}
    
    try:
        df = session_store[session_id]["dataframe"]
        
        if "intelligence_profile" not in session_store[session_id]:
            intelligence_profile = data_profiler.profile_dataset(df)
            session_store[session_id]["intelligence_profile"] = intelligence_profile
        else:
            intelligence_profile = session_store[session_id]["intelligence_profile"]
        
        from .intelligence.feature_intelligence import AdvancedFeatureIntelligence
        feature_intelligence = AdvancedFeatureIntelligence()
        
        fe_analysis = feature_intelligence.analyze_feature_engineering_opportunities(
            df, intelligence_profile, target_column
        )
        
        recommendations = fe_analysis.get('feature_engineering_recommendations', [])
        
        if priority_filter:
            recommendations = [rec for rec in recommendations if rec.priority == priority_filter]
        
        recommendations = recommendations[:max_recommendations]
        
        serializable_recommendations = []
        for rec in recommendations:
            serializable_recommendations.append({
                'feature_type': rec.feature_type,
                'priority': rec.priority,
                'description': rec.description,
                'implementation': rec.implementation,
                'expected_benefit': rec.expected_benefit,
                'computational_cost': rec.computational_cost
            })
        
        session_store[session_id]["feature_analysis"] = fe_analysis
        
        return {
            "status": "success",
            "recommendations": serializable_recommendations,
            "feature_selection_strategy": fe_analysis.get('feature_selection_strategy', {}),
            "scaling_strategy": fe_analysis.get('scaling_strategy', {}),
            "implementation_pipeline": fe_analysis.get('implementation_pipeline', []),
            "total_recommendations": len(serializable_recommendations),
            "recommendation_metadata": {
                "target_column": target_column,
                "priority_filter": priority_filter,
                "generated_at": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        return {"status": "error", "detail": f"Feature recommendation error: {str(e)}"}

@app.post("/api/data/{session_id}/apply-features")
async def apply_feature_recommendations(session_id: str, request: FeatureRecommendationRequest):
    """Apply selected feature engineering recommendations."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        df = session_store[session_id]["dataframe"]
        
        # Get stored feature analysis
        if "feature_analysis" not in session_store[session_id]:
            raise HTTPException(status_code=400, detail="No feature analysis found. Generate recommendations first.")
        
        feature_analysis = session_store[session_id]["feature_analysis"]
        
        # Apply automated feature engineering
        from .feature_generation.auto_fe import AutoFeatureEngineer
        auto_fe = AutoFeatureEngineer()
        
        enhanced_df = auto_fe.engineer_features(df, feature_analysis)
        
        # Store enhanced data
        session_store[session_id]["enhanced_data"] = enhanced_df
        
        # Calculate feature engineering impact
        impact_summary = {
            "original_features": len(df.columns),
            "engineered_features": len(enhanced_df.columns),
            "features_added": len(enhanced_df.columns) - len(df.columns),
            "feature_expansion_ratio": len(enhanced_df.columns) / len(df.columns),
            "memory_impact": enhanced_df.memory_usage().sum() / df.memory_usage().sum()
        }
        
        return {
            "status": "success",
            "message": "Feature engineering applied successfully",
            "impact_summary": impact_summary,
            "enhanced_data_shape": enhanced_df.shape,
            "new_columns": [col for col in enhanced_df.columns if col not in df.columns],
            "artifacts": {
                "enhanced_data": f"/api/data/{session_id}/download/enhanced-data"
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature application error: {str(e)}")

@app.get("/api/data/{session_id}/relationship-graph")
async def get_relationship_graph(session_id: str):
    """Get interactive relationship graph data for visualization."""
    if session_id not in session_store:
        return {"status": "error", "detail": "Session not found"}
    
    try:
        session_data = session_store[session_id]
        intelligence_profile = session_data.get("intelligence_profile")
        
        if not intelligence_profile:
            return {"status": "error", "detail": "Intelligence profile required. Run profiling first."}
        
        relationship_analysis = intelligence_profile.get('relationship_analysis', {})
        relationships = relationship_analysis.get('relationships', [])
        
        if not relationships:
            return {
                "status": "success",
                "message": "No relationships found in the data",
                "graph_data": {
                    "nodes": [],
                    "edges": [],
                    "metadata": {"total_relationships": 0}
                }
            }
        
        from .intelligence.relationship_discovery import RelationshipDiscovery
        relationship_discovery = RelationshipDiscovery()
        
        relationship_objects = []
        for rel_dict in relationships:
            class MockRelationship:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            relationship_objects.append(MockRelationship(**rel_dict))
        
        graph_data = relationship_discovery.generate_relationship_graph(relationship_objects)
        
        # Add additional metadata for visualization
        relationship_types = [rel.get('relationship_type', 'unknown') for rel in relationships]
        type_counts = {}
        for rel_type in relationship_types:
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        
        enhanced_graph = {
            "nodes": graph_data.get('nodes', []),
            "edges": graph_data.get('edges', []),
            "metadata": {
                "total_relationships": len(relationships),
                "relationship_type_counts": type_counts,
                "total_nodes": len(graph_data.get('nodes', [])),
                "graph_generated_at": datetime.now().isoformat()
            },
            "visualization_config": {
                "node_size_range": [20, 60],
                "edge_width_range": [1, 5],
                "color_scheme": {
                    "primary_key": "#ff6b6b",
                    "foreign_key": "#4ecdc4", 
                    "correlation": "#45b7d1",
                    "categorical": "#96ceb4",
                    "temporal": "#ffeaa7",
                    "default": "#ddd"
                }
            }
        }
        
        return {
            "status": "success",
            "graph_data": enhanced_graph
        }
    
    except Exception as e:
        return {"status": "error", "detail": f"Relationship graph error: {str(e)}"}

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
        if session_id not in agent_sessions:
            if create_enhanced_agent_executor is None:
                raise HTTPException(status_code=503, detail="Agent service not available")

            # Ensure clean checkpointer state for new agent session
            try:
                from data_scientist_chatbot.app.core.graph_builder import memory
                if memory:
                    memory.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
                    memory.conn.commit()
                    print(f"DEBUG: Cleaned checkpointer state for chat agent session {session_id}")
            except Exception as e:
                print(f"WARNING: Failed to clean checkpointer state for chat agent session {session_id}: {e}")

            agent_sessions[session_id] = create_enhanced_agent_executor(session_id)
        
        agent = agent_sessions[session_id]
        
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
            
            # Extract plots from current execution only (last 3 messages)
            plots = []
            messages = response.get('messages', [])
            recent_messages = messages[-3:] if len(messages) >= 3 else messages

            for msg in recent_messages:
                if hasattr(msg, 'content') and 'PLOT_SAVED:' in str(msg.content):
                    import re
                    plot_files = re.findall(r'PLOT_SAVED:([^\s]+\.png)', str(msg.content))
                    for plot_file in plot_files:
                        plots.append(f"/static/plots/{plot_file}")
            
            messages = response.get('messages', [])
            final_message = messages[-1] if messages else None
            
            # Use the final message content directly
            if final_message:
                content = final_message.content
            else:
                content = "Task completed successfully."
            
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
async def agent_chat_stream(message: str, session_id: str):
    """Stream agent chat responses with dynamic status updates generated by agents."""
    if not AGENT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent service not available")

    from fastapi.responses import StreamingResponse
    import asyncio

    async def generate_events():
        try:
            global status_agent_runnable

            # Create session if it doesn't exist
            if session_id not in session_store:
                session_store[session_id] = {"session_id": session_id, "created_at": datetime.now()}

            # Initialize agent for this session if needed
            if session_id not in agent_sessions:
                if create_enhanced_agent_executor is None:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Agent service not available'})}\n\n"
                    return

                # Ensure clean checkpointer state for new agent session
                try:
                    from data_scientist_chatbot.app.core.graph_builder import memory
                    if memory:
                        memory.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
                        memory.conn.commit()
                        print(f"DEBUG: Cleaned checkpointer state for streaming agent session {session_id}")
                except Exception as e:
                    print(f"WARNING: Failed to clean checkpointer state for streaming agent session {session_id}: {e}")

                agent_sessions[session_id] = create_enhanced_agent_executor(session_id)

            agent = agent_sessions[session_id]

            # Initialize status agent if not already created
            if status_agent_runnable is None:
                try:
                    from data_scientist_chatbot.app.core.agent_factory import create_status_agent
                    status_agent_runnable = create_status_agent()
                    print("DEBUG: Status agent initialized successfully")
                except Exception as e:
                    print(f"WARNING: Failed to initialize status agent: {e}")

            from langchain_core.messages import HumanMessage

            # Check if agent has stream_events method (LangGraph v0.2+)
            if hasattr(agent, 'stream_events'):
                print("DEBUG: Using stream_events for dynamic status updates")

                config = {"configurable": {"thread_id": session_id}}
                input_data = create_agent_input(message, session_id)

                plots = []
                final_response = None
                current_node = None

                workflow_context = {
                    "current_agent": None,
                    "current_action": None,
                    "tool_calls": [],
                    "agent_decisions": [],
                    "user_goal": message,
                    "execution_progress": {},
                    "session_id": session_id
                }

                for event in agent.stream_events(input_data, config=config, version="v2"):
                    event_name = event.get("event")
                    event_data = event.get("data", {})

                    # Track which node we're in and update workflow context
                    if event_name == "on_chain_start":
                        current_node = event.get("name", "")
                        workflow_context["current_agent"] = current_node
                        workflow_context["current_action"] = "starting"

                        # Generate real-time status from actual agent context
                        if status_agent_runnable:
                            try:
                                status_context = create_workflow_status_context(workflow_context, event)
                                from data_scientist_chatbot.app.agent import get_status_agent_prompt
                                status_prompt_template = get_status_agent_prompt()
                                status_formatted = status_prompt_template.format(
                                    current_agent=status_context.get('current_agent', 'unknown'),
                                    user_goal=status_context.get('user_goal', 'processing')
                                )
                                status_response = await asyncio.wait_for(
                                    status_agent_runnable.ainvoke(status_formatted),
                                    timeout=10.0
                                )
                                status_message = status_response.content.strip()
                                print(f"DEBUG: Real status agent output for {current_node}: {status_message}")
                                yield f"data: {json.dumps({'type': 'status', 'message': status_message})}\n\n"
                            except Exception as e:
                                print(f"DEBUG: Status agent failed for {current_node}: {type(e).__name__}: {str(e)}")
                                import traceback
                                traceback.print_exc()

                    # Capture agent responses for status extraction
                    elif event_name == "on_chat_model_stream":
                        if event_data.get("chunk"):
                            chunk_content = str(event_data["chunk"].content)
                            workflow_context["execution_progress"][current_node] = chunk_content[:100]

                    # Tool execution
                    elif event_name == "on_tool_start":
                        tool_name = event.get("name", "")
                        tool_args = event_data.get("input", {})

                        tool_info = {
                            "tool": tool_name,
                            "args": tool_args,
                            "status": "starting"
                        }
                        workflow_context["tool_calls"].append(tool_info)
                        workflow_context["current_action"] = f"executing {tool_name}"

                        # Generate status for tool execution
                        if status_agent_runnable:
                            try:
                                status_context = create_workflow_status_context(workflow_context, event)
                                from data_scientist_chatbot.app.agent import get_status_agent_prompt
                                status_prompt_template = get_status_agent_prompt()
                                status_formatted = status_prompt_template.format(
                                    current_agent=status_context.get('current_agent', 'unknown'),
                                    user_goal=status_context.get('user_goal', 'processing')
                                )
                                status_response = await asyncio.wait_for(
                                    status_agent_runnable.ainvoke(status_formatted),
                                    timeout=10.0
                                )
                                status_message = status_response.content.strip()
                                yield f"data: {json.dumps({'type': 'status', 'message': status_message})}\n\n"
                                await asyncio.sleep(0.1)
                            except Exception as e:
                                print(f"DEBUG: Status generation failed for tool_start: {type(e).__name__}: {str(e)}")
                                import traceback
                                traceback.print_exc()

                    # Tool completion with plot detection
                    elif event_name == "on_tool_end":
                        tool_name = event.get("name", "")
                        tool_output = event_data.get("output", "")

                        # Update workflow context for tool completion
                        for tool_info in workflow_context["tool_calls"]:
                            if tool_info["tool"] == tool_name and tool_info["status"] == "starting":
                                tool_info["status"] = "completed"
                                tool_info["output"] = str(tool_output)[:100]
                                break

                        workflow_context["current_action"] = f"completed {tool_name}"

                        # Generate status for tool completion
                        if status_agent_runnable:
                            try:
                                status_context = create_workflow_status_context(workflow_context, event)
                                from data_scientist_chatbot.app.agent import get_status_agent_prompt
                                status_prompt_template = get_status_agent_prompt()
                                status_formatted = status_prompt_template.format(
                                    current_agent=status_context.get('current_agent', 'unknown'),
                                    user_goal=status_context.get('user_goal', 'processing')
                                )
                                status_response = await asyncio.wait_for(
                                    status_agent_runnable.ainvoke(status_formatted),
                                    timeout=10.0
                                )
                                status_message = status_response.content.strip()
                                yield f"data: {json.dumps({'type': 'status', 'message': status_message})}\n\n"
                                await asyncio.sleep(0.1)
                            except Exception as e:
                                print(f"DEBUG: Status generation failed for tool_end: {type(e).__name__}: {str(e)}")
                                import traceback
                                traceback.print_exc()

                        # Special handling for python_code_interpreter
                        if tool_name == "python_code_interpreter":
                            if 'PLOT_SAVED:' in str(tool_output):
                                import re
                                plot_files = re.findall(r'PLOT_SAVED:([^\s]+\.png)', str(tool_output))
                                for plot_file in plot_files:
                                    plots.append(f"/static/plots/{plot_file}")
                                yield f"data: {json.dumps({'type': 'status', 'message': '📊 Visualization created successfully!'})}\n\n"
                                await asyncio.sleep(0.2)

                    # Final result
                    elif event_name == "on_chain_end" and event.get("name") == "__start__":
                        final_response = event_data.get("output")
                        break

                # Extract final message content
                if final_response and final_response.get('messages'):
                    final_message = final_response['messages'][-1]
                    content = final_message.content if hasattr(final_message, 'content') else str(final_message)
                else:
                    content = "Task completed successfully."

                yield f"data: {json.dumps({'type': 'final_response', 'response': content, 'plots': plots})}\n\n"

            else:
                # Real-time agent execution with live status monitoring
                print("DEBUG: Using streaming execution with real agent status")


                config = {"configurable": {"thread_id": session_id}}

                # Use persistent status agent for real-time updates
                from data_scientist_chatbot.app.core.state_manager import create_workflow_status_context


                # Stream graph execution to capture real agent transitions
                final_response = None
                all_messages = []
                final_brain_response = None
                action_outputs = []

                # Initialize state with original user message
                original_user_message = HumanMessage(content=message)
                current_state = {"messages": [original_user_message], "session_id": session_id}

                try:
                    for event in agent.stream(create_agent_input(message, session_id), config=config):

                        # Extract agent name and status from each execution step
                        for node_name, node_data in event.items():
                            if node_name in ['brain', 'hands', 'parser', 'action']:

                                # Update current state for status context
                                messages = node_data.get('messages', []) if node_data else []
                                if messages and isinstance(messages, list):
                                    all_messages.extend(messages)
                                    current_state["messages"] = all_messages

                                    # Capture brain responses specifically for final display
                                    if node_name == 'brain':
                                        last_brain_msg = messages[-1]
                                        brain_content = str(last_brain_msg.content) if hasattr(last_brain_msg, 'content') else str(last_brain_msg)
                                        # Only update if this looks like a final response (not tool calls)
                                        if not (hasattr(last_brain_msg, 'tool_calls') and last_brain_msg.tool_calls):
                                            final_brain_response = brain_content
                                            print(f"DEBUG: Captured brain final response: {brain_content[:200]}...")

                                    # Capture action outputs for display
                                    elif node_name == 'action':
                                        last_action_msg = messages[-1]
                                        if hasattr(last_action_msg, 'type') and last_action_msg.type == 'tool':
                                            action_content = str(last_action_msg.content)
                                            if action_content and action_content.strip():
                                                action_outputs.append(action_content)
                                                print(f"DEBUG: Captured action output: {action_content[:200]}...")

                                # Generate real status using status agent in fallback path
                                if status_agent_runnable:
                                    try:
                                        workflow_context = {
                                            "current_agent": node_name,
                                            "current_action": "processing",
                                            "user_goal": message,
                                            "session_id": session_id,
                                            "execution_progress": {},
                                            "tool_calls": []
                                        }

                                        fake_event = {"event": "on_chain_start", "name": node_name}
                                        status_context = create_workflow_status_context(workflow_context, fake_event)
                                        print(f"DEBUG: Status context for {node_name}: {status_context}")
                                        print(f"DEBUG: Status agent runnable exists: {status_agent_runnable is not None}")
                                        print(f"DEBUG: Status agent type: {type(status_agent_runnable)}")
                                        from data_scientist_chatbot.app.agent import get_status_agent_prompt
                                        status_prompt_template = get_status_agent_prompt()
                                        status_formatted = status_prompt_template.format(
                                            current_agent=status_context.get('current_agent', 'unknown'),
                                            user_goal=status_context.get('user_goal', 'processing')
                                        )
                                        status_response = await asyncio.wait_for(
                                            status_agent_runnable.ainvoke(status_formatted),
                                            timeout=10.0
                                        )
                                        status_message = status_response.content.strip() if hasattr(status_response, 'content') else str(status_response).strip()
                                        print(f"DEBUG: Raw status response for {node_name}: {repr(status_response)}")
                                        print(f"DEBUG: Fallback path status agent output for {node_name}: {repr(status_message)}")
                                        yield f"data: {json.dumps({'type': 'status', 'message': status_message})}\n\n"
                                    except Exception as e:
                                        print(f"DEBUG: Fallback status agent failed for {node_name}: {type(e).__name__}: {str(e)}")
                                        import traceback
                                        traceback.print_exc()

                                await asyncio.sleep(0.1)

                        # Store the complete state from the final event
                        final_response = {'messages': all_messages}

                except Exception as stream_error:
                    print(f"Streaming execution failed: {stream_error}")
                    # Fallback to regular invoke
                    final_response = agent.invoke(create_agent_input(message, session_id), config=config)

                response = final_response if final_response else {}

                # Extract plots from current execution only
                plots = []
                messages = response.get('messages', []) if response else []
                recent_messages = messages[-3:] if messages and len(messages) >= 3 else messages

                for msg in recent_messages:
                    if hasattr(msg, 'content'):
                        content = str(msg.content)
                        if '📊 Generated' in content and 'visualization' in content:
                            import re
                            urls = re.findall(r"'/static/plots/([^']+\.png)'", content)
                            plots.extend([f"/static/plots/{url}" for url in urls])
                        elif 'PLOT_SAVED:' in content:
                            plot_files = re.findall(r'PLOT_SAVED:([^\s]+\.png)', content)
                            plots.extend([f"/static/plots/{pf}" for pf in plot_files])

                if plots:
                    yield f"data: {json.dumps({'type': 'status', 'message': '📊 Visualization generated!'})}\n\n"
                    await asyncio.sleep(0.3)

                # Combine action outputs with brain interpretation
                content_parts = []

                # Add action outputs first (technical results)
                if action_outputs and isinstance(action_outputs, list):
                    for action_output in action_outputs:
                        if action_output:
                            content_parts.append(action_output)

                # Add brain interpretation (business insights)
                if final_brain_response:
                    content_parts.append(final_brain_response)
                    print(f"DEBUG: Using captured brain response: {final_brain_response[:100]}...")

                # Fallback if no specific outputs captured
                if not content_parts:
                    messages = response.get('messages', []) if response else []
                    final_message = messages[-1] if messages and len(messages) > 0 else None
                    fallback_content = final_message.content if final_message and hasattr(final_message, 'content') else "Task completed successfully."
                    content_parts.append(fallback_content)
                    print(f"DEBUG: Fallback to last message: {fallback_content[:100] if fallback_content else 'None'}...")

                # Combine all parts
                content = "\n\n".join(content_parts) if content_parts else "Task completed successfully."
                print(f"DEBUG: Final combined response parts: {len(content_parts) if content_parts else 0}")
                if content_parts:
                    for i, part in enumerate(content_parts):
                        print(f"DEBUG: Part {i+1}: {part[:100] if part else 'None'}...")
                print(f"DEBUG: Combined content length: {len(content) if content else 0}")
                print(f"DEBUG: Combined content preview: {content[:200] if content else 'None'}...")

                final_json = {'type': 'final_response', 'response': content, 'plots': plots}
                print(f"DEBUG: Final JSON keys: {list(final_json.keys())}")
                print(f"DEBUG: Final JSON response length: {len(final_json['response'])}")

                yield f"data: {json.dumps(final_json)}\n\n"

        except Exception as e:
            print(f"DEBUG: Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'Streaming error: {str(e)}'})}\n\n"

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

# Session Management APIs
@app.get("/api/sessions")
async def get_sessions():
    """Get list of all chat sessions"""
    sessions = []
    for session_id, session_data in session_store.items():
        if session_id == GLOBAL_SESSION_ID:
            continue

        session_title = "New Chat"

        # Check if there's a custom title stored
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

@app.post("/api/sessions/new")
async def create_new_session():
    """Create a new chat session"""
    new_session_id = str(uuid.uuid4())
    session_store[new_session_id] = {
        "session_id": new_session_id,
        "created_at": datetime.now()
    }

    # Ensure clean checkpointer state for new session
    try:
        from data_scientist_chatbot.app.core.graph_builder import memory
        if memory:
            # Clear any existing checkpoints for this new thread_id (just in case)
            memory.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (new_session_id,))
            memory.conn.commit()
            print(f"DEBUG: Ensured clean checkpointer state for new session {new_session_id}")
    except Exception as e:
        print(f"WARNING: Failed to clean checkpointer state for new session {new_session_id}: {e}")

    return {"session_id": new_session_id}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id in session_store:
        del session_store[session_id]
    if session_id in agent_sessions:
        del agent_sessions[session_id]

    # Clear checkpointer state for this session
    try:
        from data_scientist_chatbot.app.core.graph_builder import memory
        if memory:
            # Clear all checkpoints for this thread_id
            memory.conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
            memory.conn.commit()
            print(f"DEBUG: Cleared checkpointer state for session {session_id}")
    except Exception as e:
        print(f"WARNING: Failed to clear checkpointer state for session {session_id}: {e}")

    return {"success": True}

@app.put("/api/sessions/{session_id}/rename")
async def rename_session(session_id: str, request: dict):
    """Rename a chat session"""
    try:
        new_title = request.get("title", "").strip()
        if not new_title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")

        # For now, we'll store session titles in memory
        # In a real application, this would be stored in a database
        if not hasattr(rename_session, 'session_titles'):
            rename_session.session_titles = {}

        rename_session.session_titles[session_id] = new_title

        return {"success": True, "title": new_title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename session: {str(e)}")

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get conversation history for a session from checkpointer"""
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
                import pickle
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)