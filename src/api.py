import io
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np

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
    import os
    from pathlib import Path
    import importlib.util
    
    project_root = Path(__file__).parent.parent
    chatbot_path = project_root / "data_scientist_chatbot"
    app_path = chatbot_path / "app"
    
    sys.path.insert(0, str(chatbot_path))
    sys.path.insert(0, str(app_path))
    
    from data_scientist_chatbot.app.agent import create_agent_executor, create_enhanced_agent_executor
    
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
from .core.pipeline_orchestrator import RobustPipelineOrchestrator, PipelineConfig
from .core.project_definition import ProjectDefinition, Objective, Domain, Priority, RiskLevel
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

def get_session_agent(session_id: str):
    if CHAT_AVAILABLE and session_id not in session_agents:
        session_agents[session_id] = create_enhanced_agent_executor(session_id)
    return session_agents.get(session_id)

app = FastAPI(
    title="DataInsight AI",
    description="Intelligent automated data preprocessing and feature engineering platform",
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

class TaskConfig(BaseModel):
    task: str
    target_column: Optional[str] = None
    feature_generation_enabled: bool = False
    feature_selection_enabled: bool = False
    enable_intelligence: bool = True

class StrategicTaskConfig(BaseModel):
    project_id: str
    objective: str
    domain: str
    priority: str = "medium"
    risk_level: str = "medium"
    target_column: Optional[str] = None
    business_goal: str
    success_criteria: List[str] = []
    stakeholders: List[str] = []
    timeline_months: int = 3
    interpretability_required: bool = False
    compliance_rules: List[str] = []
    max_latency_ms: Optional[int] = None
    max_training_hours: Optional[float] = None
    min_accuracy: Optional[float] = None

class StrategyConfig(BaseModel):
    objective: str
    domain: str = "general"
    risk_level: str = "medium"
    max_latency_ms: Optional[int] = None
    max_training_hours: Optional[int] = None
    min_accuracy: Optional[float] = None
    protected_attributes: List[str] = []
    interpretability_required: bool = False
    compliance_rules: List[str] = []
    target_column: Optional[str] = None

class DataIngestionRequest(BaseModel):
    url: str
    data_type: str = "csv"
    enable_profiling: bool = True

class ProcessingResult(BaseModel):
    status: str
    message: str
    data_shape: Optional[tuple] = None
    column_roles: Optional[Dict[str, List[str]]] = None
    processing_time: Optional[float] = None
    artifacts: Optional[Dict[str, str]] = None
    intelligence_summary: Optional[Dict[str, Any]] = None

class IntelligenceProfile(BaseModel):
    semantic_types: Dict[str, str]
    domain_analysis: Dict[str, Any]
    relationship_summary: Dict[str, Any]
    overall_recommendations: List[str]

session_store = {}

# Make session_store globally accessible for tools
import builtins
builtins._session_store = session_store

data_profiler = IntelligentDataProfiler()

class ProfilingRequest(BaseModel):
    deep_analysis: bool = True
    include_relationships: bool = True
    include_domain_detection: bool = True

class StrategicAnalysisRequest(BaseModel):
    """Request model for comprehensive strategic analysis"""
    project_definition: StrategicTaskConfig
    include_historical_insights: bool = True
    translation_strategy: str = "balanced"  # conservative, balanced, aggressive, compliance_first

class FeatureRecommendationRequest(BaseModel):
    target_column: Optional[str] = None
    max_recommendations: int = 10
    priority_filter: Optional[str] = None  # 'high', 'medium', 'low'

class LearningFeedback(BaseModel):
    session_id: str
    success_rating: float  # 0.0 to 1.0
    execution_time: float
    user_satisfaction: float  # 0.0 to 1.0
    issues_encountered: List[str] = []
    suggestions: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    status: str
    response: str
    plots: List[str] = []

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat-first landing interface."""
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the classic dashboard interface."""
    html_path = Path(__file__).parent.parent / "static" / "dashboard.html"
    return HTMLResponse(html_path.read_text())

@app.post("/api/agent/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """Handle chat messages with the AI agent"""
    try:
        session_id = GLOBAL_SESSION_ID
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
async def upload_data(file: UploadFile = File(...), enable_profiling: bool = Form(True)):
    """Upload and validate dataset with intelligent profiling."""
    try:
        contents = await file.read()
        df = ingest_data(io.BytesIO(contents), filename=file.filename)
        
        if df is None:
            raise HTTPException(status_code=400, detail="Failed to parse uploaded file")
        
        # Validate data quality
        validator = DataQualityValidator(df)
        validation_report = validator.validate()
        
        # Generate session ID and store basic data
        session_id = GLOBAL_SESSION_ID
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
                column_profiles = intelligence_profile.get('column_profiles', {})
                domain_analysis = intelligence_profile.get('domain_analysis', {})
                
                semantic_types = {col: profile.semantic_type.value 
                                for col, profile in column_profiles.items()}
                
                detected_domains = domain_analysis.get('detected_domains', [])
                primary_domain = detected_domains[0] if detected_domains else None
                
                intelligence_summary = {
                    "semantic_types": semantic_types,
                    "primary_domain": primary_domain.get('domain') if primary_domain else 'unknown',
                    "domain_confidence": primary_domain.get('confidence', 0) if primary_domain else 0,
                    "key_insights": intelligence_profile.get('overall_recommendations', [])[:3],
                    "relationships_found": len(intelligence_profile.get('relationship_analysis', {}).get('relationships', [])),
                    "profiling_completed": True
                }
                
                response_data["intelligence_summary"] = intelligence_summary
                session_data["intelligence_profile"] = intelligence_profile
                
            except Exception as prof_e:
                response_data["intelligence_summary"] = {
                    "profiling_completed": False,
                    "profiling_error": str(prof_e)
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
                
                # Execute load code in sandbox
                load_result = execute_python_in_sandbox(load_code, session_id)
                
                if load_result["success"]:
                    clean_output = load_result['stdout']
                    html_output = convert_pandas_output_to_html(clean_output)

                    # Create enhanced message with profiling data
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
                else:
                    response_data["agent_analysis"] = f"⚠️ Data loaded but with issues: {load_result['stderr']}"
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
        
        # Generate session ID and store basic data
        session_id = GLOBAL_SESSION_ID
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
                column_profiles = intelligence_profile.get('column_profiles', {})
                domain_analysis = intelligence_profile.get('domain_analysis', {})
                
                semantic_types = {col: profile.semantic_type.value 
                                for col, profile in column_profiles.items()}
                
                detected_domains = domain_analysis.get('detected_domains', [])
                primary_domain = detected_domains[0] if detected_domains else None
                
                intelligence_summary = {
                    "semantic_types": semantic_types,
                    "primary_domain": primary_domain.get('domain') if primary_domain else 'unknown',
                    "domain_confidence": primary_domain.get('confidence', 0) if primary_domain else 0,
                    "key_insights": intelligence_profile.get('overall_recommendations', [])[:3],
                    "relationships_found": len(intelligence_profile.get('relationship_analysis', {}).get('relationships', [])),
                    "profiling_completed": True
                }
                
                response_data["intelligence_summary"] = intelligence_summary
                session_data["intelligence_profile"] = intelligence_profile
                
            except Exception as prof_e:
                response_data["intelligence_summary"] = {
                    "profiling_completed": False,
                    "profiling_error": str(prof_e)
                }
        
        session_store[session_id] = session_data

        # Store data in builtins for agent access
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

@app.post("/api/data/{session_id}/process/strategic")
async def process_data_strategic(session_id: str, config: StrategicTaskConfig):
    """Process data with strategic business-driven pipeline"""
    if session_id not in session_store:
        return {"status": "error", "detail": "Session not found"}
    
    try:
        start_time = datetime.now()
        session_data = session_store[session_id]
        df = session_data["dataframe"]
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        
        from .core.project_definition import BusinessContext, TechnicalConstraints, RegulatoryConstraints
        
        project_definition = ProjectDefinition(
            project_id=config.project_id,
            name=getattr(config, 'project_name', f"Project {config.project_id}"),
            description=getattr(config, 'description', "Automated ML project"),
            objective=Objective(config.objective.lower()),
            domain=Domain(config.domain.lower()),
            priority=Priority(config.priority.lower()),
            risk_level=RiskLevel(config.risk_level.lower()),
            business_context=BusinessContext(
                business_unit=getattr(config, 'business_unit', "Data Science"),
                success_criteria=getattr(config, 'success_criteria', ["Model accuracy > 80%"]),
                stakeholders=getattr(config, 'stakeholders', ["Data Analyst"]),
                timeline_months=getattr(config, 'timeline_months', 3)
            ),
            technical_constraints=TechnicalConstraints(
                max_training_hours=getattr(config, 'max_training_hours', 2.0),
                max_memory_gb=8.0,
                min_accuracy=getattr(config, 'min_accuracy', 0.8)
            ),
            regulatory_constraints=RegulatoryConstraints(
                privacy_level="standard",
                audit_trail_required=False,
                compliance_rules=getattr(config, 'compliance_rules', [])
            )
        )
        
        pipeline_config = PipelineConfig(
            enable_adaptive_learning=True,
            enable_security=True,
            enable_explainability=True,
            enable_compliance=True
        )
        
        orchestrator = RobustPipelineOrchestrator(pipeline_config)
        
        strategic_result = orchestrator.execute_pipeline(
            data_path=temp_file.name,
            project_definition=project_definition,
            target_column=config.target_column
        )
        
        Path(temp_file.name).unlink()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        session_store[session_id].update({
            "strategic_result": strategic_result,
            "project_definition": project_definition.__dict__,
            "processing_config": config.dict(),
            "processing_time": processing_time
        })
        
        return {
            "status": "success",
            "message": "Strategic pipeline execution completed",
            "project_id": config.project_id,
            "objective_achieved": strategic_result.get("strategic_summary", {}).get("objective_achieved", False),
            "business_impact": strategic_result.get("strategic_summary", {}).get("business_impact", {}),
            "executive_summary": strategic_result.get("strategic_summary", {}).get("executive_summary", {}),
            "processing_time": processing_time,
            "artifacts": {
                "strategic_report": f"/api/data/{session_id}/download/strategic-report",
                "executive_dashboard": f"/api/data/{session_id}/download/executive-dashboard",
                "technical_artifacts": f"/api/data/{session_id}/download/technical-artifacts"
            }
        }
    
    except Exception as e:
        return {"status": "error", "detail": f"Strategic processing error: {str(e)}"}

@app.post("/api/data/{session_id}/process")
async def process_data(session_id: str, config: TaskConfig):
    """Process data with unified RobustPipelineOrchestrator workflow."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        start_time = datetime.now()
        session_data = session_store[session_id]
        df = session_data["dataframe"]
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_file.close()  
        df.to_csv(temp_file.name, index=False)
        
        # Phase 1: RECALL - Get experience from database
        recommended_strategy = None
        if DATABASE_AVAILABLE:
            try:
                meta_db = PersistentMetaDatabase()
                from .learning.persistent_storage import create_dataset_characteristics
                
                # Create dataset characteristics for experience lookup
                dataset_characteristics = create_dataset_characteristics(
                    df, config.target_column, domain='general'
                )
                
                # Recall similar successful strategies
                recommendations = meta_db.get_recommended_strategies(
                    dataset_characteristics,
                    objective='accuracy',
                    domain='general'
                )
                
                if recommendations.get('recommended_strategies'):
                    recommended_strategy = recommendations['recommended_strategies'][0]
                
            except Exception as e:
                logging.warning(f"Experience recall failed: {e}")
        
        if recommended_strategy and recommended_strategy.get('confidence', 0) > 0.6:
            strategy_config = recommended_strategy.get('config', {})
            pipeline_config = PipelineConfig(
                enable_feature_engineering=strategy_config.get('feature_engineering_enabled', config.feature_generation_enabled),
                enable_feature_selection=strategy_config.get('feature_selection_enabled', config.feature_selection_enabled),
                enable_intelligence=config.enable_intelligence,
                security_level=strategy_config.get('security_level', 'standard'),
                privacy_level=strategy_config.get('privacy_level', 'medium')
            )
        else:
            pipeline_config = PipelineConfig(
                enable_feature_engineering=config.feature_generation_enabled,
                enable_feature_selection=config.feature_selection_enabled,
                enable_intelligence=config.enable_intelligence
            )
        
        # Initialize RobustPipelineOrchestrator
        orchestrator = RobustPipelineOrchestrator(pipeline_config)
        
        # Create default project definition for legacy compatibility
        from .core.project_definition import BusinessContext, TechnicalConstraints, RegulatoryConstraints
        
        project_definition = ProjectDefinition(
            project_id=f"api_{session_id}",
            name=f"Data Analysis Session {session_id}",
            description="Automated data analysis and model building session",
            objective=Objective.ACCURACY,
            domain=Domain.GENERAL,
            priority=Priority.MEDIUM,
            risk_level=RiskLevel.MEDIUM,
            business_context=BusinessContext(
                business_unit="Data Science",
                success_criteria=["Model accuracy > 80%", "Processing time < 30 minutes"],
                stakeholders=["Data Analyst", "Business User"],
                timeline_months=1
            ),
            technical_constraints=TechnicalConstraints(
                max_training_hours=2.0,
                max_memory_gb=8.0,
                min_accuracy=0.8
            ),
            regulatory_constraints=RegulatoryConstraints(
                privacy_level="standard",
                audit_trail_required=False,
                compliance_rules=[]
            )

        )

        # Execute pipeline with the project definition
        result_dict = orchestrator.execute_pipeline(
            data_path=temp_file.name,
            project_definition=project_definition,
            target_column=config.target_column,
            custom_config={'feature_generation_enabled': config.feature_generation_enabled, 'feature_selection_enabled': config.feature_selection_enabled}
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        session_store[session_id].update({
            "pipeline_result": result_dict,
            "processing_config": config.dict(),
            "processing_time": processing_time
        })

        return {
            "status": "success",
            "message": "Data processed successfully",
            "processing_time": processing_time,
            "artifacts": {
                "pipeline_metadata": f"/api/data/{session_id}/download/pipeline-metadata",
                "processed_data": f"/api/data/{session_id}/download/data"
            }
        }

    except Exception as e:
        return {"status": "error", "detail": f"Processing error: {str(e)}"}


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

@app.post("/api/learning/feedback")
async def record_learning_feedback(feedback: LearningFeedback):
    """Record user feedback for adaptive learning system."""
    try:
        if DATABASE_AVAILABLE:
            db_service = get_database_service()
            meta_db = PersistentMetaDatabase()
            
            # Get session data for context
            if feedback.session_id in session_store:
                session_data = session_store[feedback.session_id]
                pipeline_result = session_data.get("pipeline_result", {})
                execution_id = pipeline_result.get("execution_id") if isinstance(pipeline_result, dict) else None
                
                if not execution_id:
                    execution_id = f"exec_{feedback.session_id}"
                
                # Store feedback directly to database
                feedback_data = {
                    "user_rating": feedback.success_rating,
                    "issues_encountered": feedback.issues_encountered,
                    "suggestions": feedback.suggestions,
                    "performance_expectation_met": feedback.success_rating >= 0.7,
                    "would_recommend": feedback.user_satisfaction >= 0.7
                }
                
                success = meta_db.record_user_feedback(execution_id, feedback_data)
                
                if success:
                    # Update the original pipeline execution with user feedback
                    update_success = meta_db.update_execution_with_feedback(
                        execution_id, 
                        user_satisfaction=feedback.user_satisfaction,
                        success_rating=feedback.success_rating
                    )
                    
                    if update_success:
                        return {
                            "status": "success",
                            "message": "Feedback integrated into execution record",
                            "feedback_id": f"feedback_{execution_id}_{int(datetime.now().timestamp())}",
                            "learning_impact": "System will prioritize similar strategies based on your feedback"
                        }
                    else:
                        return {
                            "status": "partial_success", 
                            "message": "Feedback stored but execution update failed",
                            "feedback_id": f"feedback_{execution_id}_{int(datetime.now().timestamp())}"
                        }
                else:
                    raise Exception("Database feedback storage failed")
            else:
                return {
                    "status": "error",
                    "message": "Session not found - cannot link feedback to execution"
                }
        
        # Fallback to legacy adaptive learning system
        from .learning.adaptive_system import AdaptiveLearningSystem
        learning_system = AdaptiveLearningSystem()
        
        if feedback.session_id in session_store:
            session_data = session_store[feedback.session_id]
            intelligence_profile = session_data.get("intelligence_profile", {})
            processing_config = session_data.get("processing_config", {})
            
            pipeline_results = {
                "performance_metrics": {"user_satisfaction": feedback.user_satisfaction},
                "execution_time": feedback.execution_time,
                "successful_stages": 7 if feedback.success_rating > 0.7 else int(feedback.success_rating * 7),
                "total_stages": 7
            }
            
            execution_metadata = {
                "total_time": feedback.execution_time,
                "config": processing_config,
                "user_feedback": {
                    "success_rating": feedback.success_rating,
                    "satisfaction": feedback.user_satisfaction,
                    "issues": feedback.issues_encountered,
                    "suggestions": feedback.suggestions
                }
            }
            
            learning_system.record_pipeline_execution(
                pipeline_results, intelligence_profile, execution_metadata
            )
            
            return {
                "status": "success",
                "message": "Feedback recorded successfully",
                "feedback_id": f"feedback_{feedback.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "learning_impact": "Feedback will improve future recommendations"
            }
        else:
            return {
                "status": "error",
                "message": "Session not found"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback recording error: {str(e)}")

@app.get("/api/learning/insights")
async def get_learning_insights():
    """Get insights about the adaptive learning system state."""
    try:
        from .learning.adaptive_system import AdaptiveLearningSystem
        learning_system = AdaptiveLearningSystem()
        
        insights = learning_system.get_learning_summary()
        
        return {
            "status": "success",
            "learning_insights": insights,
            "system_health": "operational",
            "insights_generated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Learning insights error: {str(e)}")

@app.get("/api/data/{session_id}/adaptive-recommendations")
async def get_adaptive_recommendations(session_id: str):
    """Get context-aware recommendations based on learned patterns."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = session_store[session_id]
        intelligence_profile = session_data.get("intelligence_profile")
        
        if not intelligence_profile:
            raise HTTPException(status_code=400, detail="Intelligence profile required. Run profiling first.")
        
        from .learning.adaptive_system import AdaptiveLearningSystem
        learning_system = AdaptiveLearningSystem()
        
        current_config = session_data.get("processing_config", {})
        adaptive_recs = learning_system.get_adaptive_recommendations(
            intelligence_profile, current_config
        )
        
        return {
            "status": "success",
            "adaptive_recommendations": adaptive_recs.get('adaptive_recommendations', {}),
            "confidence_scores": adaptive_recs.get('confidence_scores', {}),
            "learning_insights": adaptive_recs.get('learning_insights', {}),
            "recommendations_generated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adaptive recommendations error: {str(e)}")

@app.get("/api/data/{session_id}/pipeline-status")
async def get_pipeline_status(session_id: str):
    """Get real-time pipeline execution status."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = session_store[session_id]
        
        # Check for unified pipeline results
        pipeline_result = session_data.get("pipeline_result")
        
        if pipeline_result:
            # Extract status from unified pipeline
            execution_summary = pipeline_result.get('execution_summary', {})
            
            status_info = {
                "pipeline_type": "unified",
                "overall_status": pipeline_result.get('status', 'unknown'),
                "session_id": pipeline_result.get('session_id'),
                "total_stages": execution_summary.get('total_stages', 0),
                "successful_stages": execution_summary.get('successful_stages', 0),
                "total_execution_time": execution_summary.get('total_time', 0),
                "stage_details": pipeline_result.get('results', {}),
                "last_updated": datetime.now().isoformat()
            }
        else:
            # No pipeline execution found
            status_info = {
                "pipeline_type": "none",
                "overall_status": "not_started",
                "message": "No pipeline execution found for this session",
                "last_updated": datetime.now().isoformat()
            }
        
        # Add session metadata
        status_info.update({
            "session_metadata": {
                "data_uploaded": "dataframe" in session_data,
                "intelligence_profiled": "intelligence_profile" in session_data,
                "features_analyzed": "feature_analysis" in session_data,
                "data_enhanced": "enhanced_data" in session_data
            }
        })
        
        return {
            "status": "success",
            "pipeline_status": status_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline status error: {str(e)}")

@app.post("/api/data/{session_id}/pipeline-recovery")
async def trigger_pipeline_recovery(session_id: str):
    """Manually trigger pipeline recovery for failed executions."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = session_store[session_id]
        
        # Check if there's a failed pipeline to recover
        pipeline_result = session_data.get("pipeline_result")
        
        if not pipeline_result:
            raise HTTPException(status_code=400, detail="No pipeline execution found to recover")
        
        if pipeline_result.get('status') != 'pipeline_failure':
            return {
                "status": "success",
                "message": "Pipeline is not in failed state, no recovery needed",
                "current_status": pipeline_result.get('status')
            }
        
        # Attempt recovery by re-running with more conservative settings
        df = session_data["dataframe"]
        
        # Create temporary file for recovery attempt
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        
        # Initialize robust orchestrator with recovery configuration
        recovery_config = PipelineConfig(
            max_memory_usage=0.6,  # More conservative
            enable_caching=False,   # Disable caching for recovery
            auto_recovery=False,    # Manual recovery mode
            parallel_processing=False  # Single-threaded for stability
        )
        
        robust_orchestrator = RobustPipelineOrchestrator(recovery_config)
        
        # Get original processing config
        processing_config = session_data.get("processing_config", {})
        task_type = processing_config.get("task", "supervised")
        target_column = processing_config.get("target_column")
        
        # Execute recovery pipeline
        recovery_result = robust_orchestrator.execute_pipeline(
            data_path=temp_file.name,
            task_type=task_type,
            target_column=target_column
        )
        
        # Clean up temp file
        Path(temp_file.name).unlink()
        
        # Update session with recovery results
        session_data["pipeline_result"] = recovery_result
        session_data["recovery_attempted"] = datetime.now().isoformat()
        
        return {
            "status": "success",
            "message": "Pipeline recovery completed",
            "recovery_result": {
                "new_status": recovery_result.get('status'),
                "recovery_time": datetime.now().isoformat(),
                "recovery_successful": recovery_result.get('status') == 'success'
            }
        }
    
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Pipeline recovery failed: {str(e)}",
            "recovery_time": datetime.now().isoformat(),
            "suggestions": [
                "Try with a smaller dataset",
                "Disable feature generation",
                "Use legacy pipeline mode"
            ]
        }

@app.post("/api/strategic/analyze")
async def strategic_analysis(request: StrategicAnalysisRequest):
    """Pillar 0: Strategic Control Layer - Comprehensive strategic analysis and recommendation"""
    try:
        from .core.project_definition import ProjectDefinition, Objective, Domain, Priority, RiskLevel
        from .core.project_definition import BusinessContext, TechnicalConstraints, RegulatoryConstraints
        
        config = request.project_definition
        project_definition = ProjectDefinition(
            project_id=config.project_id,
            name=getattr(config, 'project_name', f"Project {config.project_id}"),
            description=getattr(config, 'description', "Automated ML project"),
            objective=Objective(config.objective.lower()),
            domain=Domain(config.domain.lower()),
            priority=Priority(config.priority.lower()),
            risk_level=RiskLevel(config.risk_level.lower()),
            business_context=BusinessContext(
                business_unit=getattr(config, 'business_unit', "Data Science"),
                success_criteria=getattr(config, 'success_criteria', ["Model accuracy > 80%"]),
                stakeholders=getattr(config, 'stakeholders', ["Data Analyst"]),
                timeline_months=getattr(config, 'timeline_months', 3)
            ),
            technical_constraints=TechnicalConstraints(
                max_training_hours=getattr(config, 'max_training_hours', 2.0),
                max_memory_gb=8.0,
                min_accuracy=getattr(config, 'min_accuracy', 0.8)
            ),
            regulatory_constraints=RegulatoryConstraints(
                privacy_level="standard",
                audit_trail_required=False,
                compliance_rules=getattr(config, 'compliance_rules', [])
            )
        )    
        
        validation_results = project_definition.validate_project_definition()
        
        return {
            "status": "success",
            "strategic_analysis": {
                "project_overview": {
                    "project_id": project_definition.project_id,
                    "objective": project_definition.objective.value,
                    "domain": project_definition.domain.value,
                    "priority": project_definition.priority.value,
                    "risk_level": project_definition.risk_level.value
                },
                
            },
            "analysis_metadata": {
                "translation_strategy": request.translation_strategy,
                "analysis_timestamp": datetime.now().isoformat(),
                "strategic_control_version": "2.0"
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Strategic analysis failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def _calculate_feasibility_score(project_def: ProjectDefinition, tech_config) -> float:
    """Calculate project feasibility score (0.0 to 1.0)"""
    score = 0.8  # Base feasibility
    
    # Adjust for complexity vs timeline
    if tech_config.model_complexity.value == "experimental" and project_def.business_context.get('timeline_months', 6) < 6:
        score -= 0.2
    
    # Adjust for resource constraints
    if tech_config.max_training_time_hours and tech_config.max_training_time_hours > 48:
        score -= 0.1
    
    # Adjust for compliance requirements
    if len(tech_config.audit_requirements) > 3:
        score -= 0.1
    
    return max(0.0, min(1.0, score))

def _calculate_risk_mitigation_score(project_def: ProjectDefinition, tech_config) -> float:
    """Calculate risk mitigation score (0.0 to 1.0)"""
    score = 0.6  # Base score
    
    # Higher validation rigor increases score
    if tech_config.validation_rigor == "regulatory":
        score += 0.2
    elif tech_config.validation_rigor == "extensive":
        score += 0.1
    
    # Interpretability increases score for critical projects
    if project_def.risk_level.value == "critical" and tech_config.interpretability_level != "none":
        score += 0.2
    
    # Ensemble methods increase robustness
    if tech_config.ensemble_strategy:
        score += 0.1
    
    return max(0.0, min(1.0, score))

def _calculate_resource_optimization_score(tech_config) -> float:
    """Calculate resource optimization score (0.0 to 1.0)"""
    score = 0.7  # Base optimization
    
    # Budget priority alignment
    if tech_config.compute_budget_priority == "cost_optimized":
        score += 0.2
    elif tech_config.compute_budget_priority == "balanced":
        score += 0.1
    
    # Reasonable resource limits
    if tech_config.max_training_time_hours and tech_config.max_training_time_hours <= 24:
        score += 0.1
    
    return max(0.0, min(1.0, score))

def _calculate_compliance_readiness_score(project_def: ProjectDefinition, tech_config) -> float:
    """Calculate compliance readiness score (0.0 to 1.0)"""
    score = 0.5  # Base compliance
    
    # Interpretability for regulated domains
    if project_def.domain.value in ["finance", "healthcare"] and tech_config.interpretability_level != "none":
        score += 0.3
    
    # Audit trail capabilities
    if len(tech_config.audit_requirements) > 0:
        score += 0.2
    
    # Data governance measures
    if tech_config.data_governance.get("encryption_required", False):
        score += 0.1
    
    return max(0.0, min(1.0, score))

def _generate_executive_recommendations(project_def: ProjectDefinition, tech_config, assessment: Dict[str, Any]) -> List[str]:
    """Generate executive-level strategic recommendations"""
    recommendations = []
    
    # Feasibility recommendations
    if assessment["feasibility_score"] < 0.6:
        recommendations.append("Consider extending project timeline or reducing scope complexity to improve feasibility")
    
    # Risk mitigation recommendations
    if assessment["risk_mitigation_score"] < 0.7:
        recommendations.append("Implement additional validation measures and consider ensemble methods for risk mitigation")
    
    # Resource optimization recommendations
    if assessment["resource_optimization_score"] < 0.7:
        recommendations.append("Review resource allocation to optimize cost-performance balance")
    
    # Compliance recommendations
    if project_def.domain.value in ["finance", "healthcare"] and assessment["compliance_readiness_score"] < 0.8:
        recommendations.append("Strengthen compliance measures including interpretability and audit trail capabilities")
    
    # Performance optimization recommendations
    if tech_config.model_complexity.value == "experimental":
        recommendations.append("Consider proven algorithms first, then explore experimental approaches if needed")
    
    # Timeline recommendations
    if project_def.business_context.get('timeline_months', 6) < 3:
        recommendations.append("Aggressive timeline detected - prioritize simple, proven approaches for rapid deployment")
    
    return recommendations

def _estimate_strategic_timeline(project_def: ProjectDefinition, tech_config) -> Dict[str, Any]:
    """Estimate strategic project timeline"""
    
    base_weeks = 8  # Base project duration
    
    # Adjust for complexity
    complexity_multiplier = {
        "simple": 0.7,
        "moderate": 1.0,
        "complex": 1.3,
        "experimental": 1.6
    }
    
    total_weeks = base_weeks * complexity_multiplier.get(tech_config.model_complexity.value, 1.0)
    
    if tech_config.validation_rigor == "regulatory":
        total_weeks *= 1.3
    elif tech_config.validation_rigor == "extensive":
        total_weeks *= 1.2
    
    if tech_config.interpretability_level == "full":
        total_weeks *= 1.2
    
    phases = {
        "data_preparation": max(1, int(total_weeks * 0.2)),
        "model_development": max(2, int(total_weeks * 0.4)),
        "validation_testing": max(1, int(total_weeks * 0.2)),
        "deployment_preparation": max(1, int(total_weeks * 0.15)),
        "monitoring_setup": max(1, int(total_weeks * 0.05))
    }
    
    return {
        "total_estimated_weeks": int(total_weeks),
        "confidence_level": 0.8 if tech_config.model_complexity.value in ["simple", "moderate"] else 0.6,
        "phase_breakdown": phases,
        "critical_path": ["data_preparation", "model_development", "validation_testing"]
    }

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

@app.get("/api/database/health")
async def database_health():
    """Database health check endpoint"""
    if not DATABASE_AVAILABLE:
        return {"status": "unavailable", "message": "Database components not installed"}
    
    try:
        db_manager = get_database_manager()
        health_status = db_manager.test_connection()
        pool_status = db_manager.get_connection_pool_status()
        
        return {
            "status": "healthy" if health_status["status"] == "healthy" else "unhealthy",
            "database": health_status,
            "connection_pool": pool_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/database/analytics")
async def database_analytics():
    """Database analytics and performance metrics"""
    if not DATABASE_AVAILABLE:
        return {"status": "unavailable"}
    
    try:
        db_service = get_database_service()
        
        system_summary = db_service.get_system_summary()
        domain_analytics = db_service.get_domain_analytics()
        strategy_effectiveness = db_service.get_strategy_effectiveness()
        
        return {
            "status": "success",
            "system_summary": system_summary,
            "domain_analytics": domain_analytics,
            "strategy_effectiveness": strategy_effectiveness,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/database/cleanup")
async def database_cleanup(retention_days: int = 365):
    """Clean up old database records"""
    if not DATABASE_AVAILABLE:
        return {"status": "unavailable"}
    
    try:
        db_service = get_database_service()
        deleted_count = db_service.cleanup_old_data(retention_days)
        
        return {
            "status": "success",
            "deleted_records": deleted_count,
            "retention_days": retention_days,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Chat/Conversational AI Endpoints

class ChatIntent(BaseModel):
    session_id: str
    user_message: str

class KnowledgeQueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    include_explanation: bool = True

class DataExplorationRequest(BaseModel):
    session_id: str
    request_type: str
    parameters: Optional[Dict[str, Any]] = None

@app.post("/api/explore")
async def explore_data(request: DataExplorationRequest):
    """Handle data exploration and visualization requests"""
    try:
        session_id = request.session_id
        
        if session_id not in session_store or "dataframe" not in session_store[session_id]:
            raise HTTPException(status_code=400, detail="No data found for this session. Please upload data first.")
        
        df = session_store[session_id]["dataframe"]
        request_type = request.request_type.lower()
        params = request.parameters or {}
        
        if request_type in ["head", "first", "top", "beginning"]:
            n = params.get("n", 10)
            result = data_explorer.get_data_head(df, n)
            return {
                "status": "success",
                "type": "data_table",
                "data": result,
                "description": f"First {n} rows of the dataset"
            }
        
        elif request_type in ["tail", "last", "bottom", "end"]:
            n = params.get("n", 10)
            result = data_explorer.get_data_tail(df, n)
            return {
                "status": "success",
                "type": "data_table", 
                "data": result,
                "description": f"Last {n} rows of the dataset"
            }
        
        elif request_type in ["sample", "random"]:
            n = params.get("n", 10)
            result = data_explorer.get_data_sample(df, n)
            return {
                "status": "success",
                "type": "data_table",
                "data": result,
                "description": f"Random sample of {n} rows from the dataset"
            }
        
        elif request_type in ["info", "summary", "describe"]:
            result = data_explorer.get_data_info(df)
            return {
                "status": "success",
                "type": "data_info",
                "data": result,
                "description": "Dataset information and statistics"
            }
        
        elif request_type in ["correlation", "heatmap", "corr"]:
            viz_result = data_explorer.create_correlation_heatmap(df)
            return {
                "status": "success",
                "type": "visualization",
                "chart_base64": viz_result.chart_base64,
                "description": viz_result.description,
                "insights": viz_result.insights
            }
        
        elif request_type in ["distribution", "histogram", "dist"]:
            columns = params.get("columns")
            viz_result = data_explorer.create_distribution_plots(df, columns)
            return {
                "status": "success",
                "type": "visualization",
                "chart_base64": viz_result.chart_base64,
                "description": viz_result.description,
                "insights": viz_result.insights
            }
        
        elif request_type in ["scatter", "scatterplot"]:
            x_col = params.get("x_column")
            y_col = params.get("y_column")
            color_col = params.get("color_column")
            
            if not x_col or not y_col:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) >= 2:
                    x_col = x_col or numeric_cols[0]
                    y_col = y_col or numeric_cols[1]
                else:
                    raise HTTPException(status_code=400, detail="Need at least 2 numeric columns for scatter plot")
            
            viz_result = data_explorer.create_scatter_plot(df, x_col, y_col, color_col)
            return {
                "status": "success",
                "type": "visualization",
                "chart_base64": viz_result.chart_base64,
                "description": viz_result.description,
                "insights": viz_result.insights
            }
        
        elif request_type in ["boxplot", "box", "outliers"]:
            columns = params.get("columns")
            viz_result = data_explorer.create_box_plots(df, columns)
            return {
                "status": "success",
                "type": "visualization",
                "chart_base64": viz_result.chart_base64,
                "description": viz_result.description,
                "insights": viz_result.insights
            }
        
        elif request_type in ["missing", "null", "na"]:
            viz_result = data_explorer.create_missing_data_heatmap(df)
            return {
                "status": "success",
                "type": "visualization",
                "chart_base64": viz_result.chart_base64,
                "description": viz_result.description,
                "insights": viz_result.insights
            }
        
        elif request_type == "suggestions":
            characteristics = data_explorer.analyze_data_characteristics(df)
            return {
                "status": "success",
                "type": "suggestions",
                "data": characteristics,
                "description": "Data characteristics and visualization suggestions"
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown request type: {request_type}")
    
    except Exception as e:
        logger.error(f"Data exploration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Exploration failed: {str(e)}")

class AgentChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    async_execution: bool = False

class AgentTaskStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None

import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

agent_sessions = {}
async_tasks = {}
task_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="DataInsight-Agent-")
performance_monitor = None
context_manager = None
create_enhanced_agent_executor = None

try:
    from data_scientist_chatbot.app.agent import create_enhanced_agent_executor
    from data_scientist_chatbot.app.performance_monitor import PerformanceMonitor
    from data_scientist_chatbot.app.context_manager import ContextManager
    performance_monitor = PerformanceMonitor()
    context_manager = ContextManager()
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Agent import failed: {e}")
    print("Agent service will not be available - install missing dependencies")
    AGENT_AVAILABLE = False
    create_enhanced_agent_executor = None

def run_agent_task(agent, user_message, session_id):
    start_time = time.time()
    
    try:
        if performance_monitor:
            performance_monitor.record_metric(
                session_id=session_id,
                metric_name="agent_task_started",
                value=1.0,
                context={"message_length": len(user_message)}
            )
        
        from langchain_core.messages import HumanMessage
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_message)], "session_id": session_id},
            config={"configurable": {"thread_id": session_id}}
        )
        
        execution_time = time.time() - start_time
        if performance_monitor:
            performance_monitor.record_metric(
                session_id=session_id,
                metric_name="agent_task_completed",
                value=execution_time,
                context={
                    "success": True,
                    "response_length": len(response['messages'][-1].content) if response.get('messages') else 0
                }
            )
        
        return {"success": True, "response": response['messages'][-1].content, "execution_time": execution_time}
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        if performance_monitor:
            performance_monitor.record_metric(
                session_id=session_id,
                metric_name="agent_task_failed",
                value=execution_time,
                context={"error": str(e), "success": False}
            )
        
        return {"success": False, "error": str(e), "execution_time": execution_time}

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
            from langchain_core.messages import HumanMessage
            print(f"DEBUG: Using agent type: {type(agent)}")
            response = agent.invoke({
                "messages": [HumanMessage(content=user_message)],
                "session_id": session_id
            })
            
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
            # Import status extraction functions
            from data_scientist_chatbot.app.agent import extract_brain_status, extract_hands_status, extract_parser_status

            # Create session if it doesn't exist
            if session_id not in session_store:
                session_store[session_id] = {"session_id": session_id, "created_at": datetime.now()}

            # Initialize agent for this session if needed
            if session_id not in agent_sessions:
                if create_enhanced_agent_executor is None:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Agent service not available'})}\n\n"
                    return
                agent_sessions[session_id] = create_enhanced_agent_executor(session_id)

            agent = agent_sessions[session_id]

            from langchain_core.messages import HumanMessage

            # Check if agent has stream_events method (LangGraph v0.2+)
            if hasattr(agent, 'stream_events'):
                print("DEBUG: Using stream_events for dynamic status updates")

                config = {"configurable": {"thread_id": session_id}}
                input_data = {
                    "messages": [HumanMessage(content=message)],
                    "session_id": session_id
                }

                plots = []
                final_response = None
                current_node = None

                for event in agent.stream_events(input_data, config=config, version="v2"):
                    event_name = event.get("event")
                    event_data = event.get("data", {})

                    # Track which node we're in
                    if event_name == "on_chain_start":
                        current_node = event.get("name", "")

                    # Capture agent responses for status extraction
                    elif event_name == "on_chat_model_stream":
                        if event_data.get("chunk"):
                            chunk_content = str(event_data["chunk"].content)

                            # Extract status from brain agent
                            if current_node == "brain":
                                status = extract_brain_status(chunk_content)
                                if status:
                                    yield f"data: {json.dumps({'type': 'status', 'message': status, 'agent': 'brain'})}\n\n"
                                    await asyncio.sleep(0.1)

                            # Extract status from hands agent
                            elif current_node == "hands":
                                status = extract_hands_status(chunk_content)
                                if status:
                                    yield f"data: {json.dumps({'type': 'status', 'message': status, 'agent': 'hands'})}\n\n"
                                    await asyncio.sleep(0.1)

                    # Tool execution
                    elif event_name == "on_tool_start":
                        tool_name = event.get("name", "")
                        tool_args = event_data.get("input", {})
                        status = extract_parser_status(tool_name, tool_args)
                        yield f"data: {json.dumps({'type': 'status', 'message': status, 'tool': tool_name})}\n\n"
                        await asyncio.sleep(0.1)

                    # Tool completion with plot detection
                    elif event_name == "on_tool_end":
                        tool_name = event.get("name", "")
                        if tool_name == "python_code_interpreter":
                            tool_output = event_data.get("output", "")
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
                # Fallback: Use regular invoke with status extraction
                print("DEBUG: Using invoke with dynamic status extraction")

                yield f"data: {json.dumps({'type': 'status', 'message': '🚀 Starting analysis...'})}\n\n"
                await asyncio.sleep(0.5)

                # Simulate progressive status updates
                yield f"data: {json.dumps({'type': 'status', 'message': '🧠 Analyzing request and planning approach...'})}\n\n"
                await asyncio.sleep(0.5)

                # Execute agent and capture intermediate results
                config = {"configurable": {"thread_id": session_id}}
                response = agent.invoke({
                    "messages": [HumanMessage(content=message)],
                    "session_id": session_id
                }, config=config)

                yield f"data: {json.dumps({'type': 'status', 'message': '👨‍💻 Generating and executing code...'})}\n\n"
                await asyncio.sleep(0.5)

                # Extract plots from current execution only
                plots = []
                messages = response.get('messages', [])
                recent_messages = messages[-3:] if len(messages) >= 3 else messages

                for msg in recent_messages:
                    if hasattr(msg, 'content') and 'PLOT_SAVED:' in str(msg.content):
                        import re
                        plot_files = re.findall(r'PLOT_SAVED:([^\s]+\.png)', str(msg.content))
                        for plot_file in plot_files:
                            plots.append(f"/static/plots/{plot_file}")

                if plots:
                    yield f"data: {json.dumps({'type': 'status', 'message': '📊 Visualization generated!'})}\n\n"
                    await asyncio.sleep(0.3)

                messages = response.get('messages', [])
                final_message = messages[-1] if messages else None
                content = final_message.content if final_message else "Task completed successfully."

                yield f"data: {json.dumps({'type': 'final_response', 'response': content, 'plots': plots})}\n\n"

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

@app.get("/api/agent/performance/{session_id}")
async def get_agent_performance(session_id: str):
    """Get performance metrics for a specific agent session."""
    if not AGENT_AVAILABLE or not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        performance_summary = performance_monitor.get_performance_summary(session_id=session_id, hours=24)
        slow_operations = performance_monitor.get_slow_operations(threshold_seconds=0.5, limit=5)
        cache_optimization = performance_monitor.optimize_cache()
        
        return {
            "session_id": session_id,
            "performance_summary": performance_summary,
            "slow_operations": slow_operations,
            "cache_optimization": cache_optimization,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving performance metrics: {str(e)}")

@app.get("/api/agent/system-performance")
async def get_system_performance():
    """Get overall system performance metrics."""
    if not AGENT_AVAILABLE or not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        # Record current system metrics
        performance_monitor.record_system_metrics()
        
        # Get comprehensive performance summary
        overall_summary = performance_monitor.get_performance_summary(hours=24)
        slow_operations = performance_monitor.get_slow_operations(threshold_seconds=1.0, limit=10)
        cache_stats = performance_monitor.optimize_cache()
        
        return {
            "system_performance": overall_summary,
            "slow_operations": slow_operations,
            "cache_performance": cache_stats,
            "active_sessions": len(agent_sessions),
            "active_tasks": len(async_tasks),
            "thread_pool_stats": {
                "max_workers": task_executor._max_workers if task_executor else 0,
                "current_threads": len(task_executor._threads) if task_executor and hasattr(task_executor, '_threads') else 0
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving system performance: {str(e)}")

@app.post("/api/agent/performance/cleanup")
async def cleanup_performance_data(days_old: int = 7):
    """Clean up old performance data to manage storage."""
    if not AGENT_AVAILABLE or not performance_monitor:
        raise HTTPException(status_code=503, detail="Performance monitoring not available")
    
    try:
        deleted_metrics = performance_monitor.cleanup_old_metrics(days_old)
        deleted_contexts = context_manager.cleanup_old_contexts(days_old) if context_manager else 0
        
        return {
            "deleted_metrics": deleted_metrics,
            "deleted_contexts": deleted_contexts,
            "days_old": days_old,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up performance data: {str(e)}")

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