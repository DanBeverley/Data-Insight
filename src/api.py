"""DataInsight AI - FastAPI Backend

Professional REST API for the DataInsight AI platform with comprehensive
data processing, feature engineering, and EDA capabilities.
"""

import io
import json
import tempfile
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

from .common.data_ingestion import ingest_data, ingest_from_url
from .data_quality.validator import DataQualityValidator
from .utils import generate_eda_report
from .intelligence.data_profiler import IntelligentDataProfiler
from .core.pipeline_orchestrator import RobustPipelineOrchestrator, PipelineConfig
from .core.project_definition import ProjectDefinition, Objective, Domain, Priority, RiskLevel
from .core.strategy_translator import StrategyTranslator
from .llm.interface import LLMInterface
from .llm.rag_system import LocalRAGSystem

try:
    from .database.service import get_database_service
    from .database.connection import get_database_manager
    from .learning.persistent_storage import PersistentMetaDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

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

# Initialize LLM interface and RAG system
llm_interface = LLMInterface()
rag_system = LocalRAGSystem()

# Serve static files with explicit routes
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

# Mount remaining static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Data models
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

# In-memory storage (in production, use Redis/database)
session_store = {}

# Initialize intelligence components
data_profiler = IntelligentDataProfiler()

# Additional models for new endpoints
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

@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...), enable_profiling: bool = Form(True)):
    """Upload and validate dataset with intelligent profiling."""
    try:
        # Read uploaded file
        contents = await file.read()
        df = ingest_data(io.BytesIO(contents), filename=file.filename)
        
        if df is None:
            raise HTTPException(status_code=400, detail="Failed to parse uploaded file")
        
        # Validate data quality
        validator = DataQualityValidator(df)
        validation_report = validator.validate()
        
        # Generate session ID and store basic data
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
        
        # Add intelligent profiling if enabled
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
                # Don't fail the upload if profiling fails
                response_data["intelligence_summary"] = {
                    "profiling_completed": False,
                    "profiling_error": str(prof_e)
                }
        
        session_store[session_id] = session_data

        # Push compact dataset summary to RAG for this session
        try:
            dataset_content = f"""
Dataset Summary:\n- Shape: {df.shape[0]} rows x {df.shape[1]} columns\n- Columns: {', '.join(df.columns[:30])}{'...' if len(df.columns) > 30 else ''}\n- Dtypes counts: {pd.Series(df.dtypes.astype(str)).value_counts().to_dict()}
"""
            rag_system.add_document(
                content=dataset_content,
                metadata={"type": "dataset_summary", "filename": file.filename, "columns": df.columns.tolist()},
                document_type="dataset_summary",
                session_id=session_id
            )
        except Exception as e:
            logging.warning(f"Failed to add dataset summary to RAG: {e}")

        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

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
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
        
        # Add intelligent profiling if enabled
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
                # Don't fail the ingestion if profiling fails
                response_data["intelligence_summary"] = {
                    "profiling_completed": False,
                    "profiling_error": str(prof_e)
                }
        
        session_store[session_id] = session_data

        # Push compact dataset summary to RAG for this session (URL ingest)
        try:
            dataset_content = f"""
Dataset Summary:\n- Shape: {df.shape[0]} rows x {df.shape[1]} columns\n- Columns: {', '.join(df.columns[:30])}{'...' if len(df.columns) > 30 else ''}\n- Dtypes counts: {pd.Series(df.dtypes.astype(str)).value_counts().to_dict()}
"""
            rag_system.add_document(
                content=dataset_content,
                metadata={"type": "dataset_summary", "source_url": request.url, "columns": df.columns.tolist()},
                document_type="dataset_summary",
                session_id=session_id
            )
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
        
        # Use basic analysis to avoid dependency issues
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
        
        project_definition = ProjectDefinition(
            project_id=config.project_id,
            objective=Objective(config.objective.lower()),
            domain=Domain(config.domain.lower()),
            priority=Priority(config.priority.lower()),
            risk_level=RiskLevel(config.risk_level.lower()),
            business_context={
                'goal': config.business_goal,
                'success_criteria': config.success_criteria,
                'stakeholders': config.stakeholders,
                'timeline_months': config.timeline_months
            },
            technical_constraints={
                'max_latency_ms': config.max_latency_ms,
                'max_training_hours': config.max_training_hours,
                'min_accuracy': config.min_accuracy
            },
            regulatory_constraints={
                'interpretability_required': config.interpretability_required,
                'compliance_rules': config.compliance_rules
            }
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
        return {"status": "error", "detail": "Session not found"}
    
    try:
        start_time = datetime.now()
        session_data = session_store[session_id]
        df = session_data["dataframe"]
        
        # Create temporary file for pipeline execution
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
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
        project_definition = ProjectDefinition(
            project_id=f"api_{session_id}",
            objective=Objective.ACCURACY,
            domain=Domain.GENERAL,
            priority=Priority.MEDIUM,
            risk_level=RiskLevel.MEDIUM
        )
        
        result_dict = orchestrator.execute_pipeline(
            data_path=temp_file.name,
            project_definition=project_definition,
            target_column=config.target_column
        )
        
        # Clean up temporary file
        Path(temp_file.name).unlink()
        
        # Validate pipeline execution
        if result_dict.get('status') != 'success':
            raise Exception(f"Pipeline execution failed: {result_dict.get('error', 'Unknown error')}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        session_store[session_id].update({
            "pipeline_result": result_dict,
            "processing_config": config.dict(),
            "processing_time": processing_time,
            "recommended_strategy": recommended_strategy
        })
        
        # Extract intelligence summary
        intelligence_profile = session_data.get("intelligence_profile")
        intelligence_summary = None
        if intelligence_profile:
            domain_analysis = intelligence_profile.get('domain_analysis', {})
            detected_domains = domain_analysis.get('detected_domains', [])
            intelligence_summary = {
                "primary_domain": detected_domains[0].get('domain') if detected_domains else 'unknown',
                "total_recommendations": len(intelligence_profile.get('overall_recommendations', [])),
                "relationships_analyzed": len(intelligence_profile.get('relationship_analysis', {}).get('relationships', [])),
                "feature_generation_applied": config.feature_generation_enabled,
                "feature_selection_applied": config.feature_selection_enabled,
                "original_shape": df.shape,
                "final_shape": result_dict.get('final_data_shape', df.shape)
            }
        
        return ProcessingResult(
            status="success",
            message="Data processed with unified pipeline",
            data_shape=result_dict.get('final_data_shape', df.shape),
            processing_time=processing_time,
            intelligence_summary=intelligence_summary,
            artifacts={
                "pipeline_metadata": f"/api/data/{session_id}/download/pipeline-metadata",
                "processed_data": f"/api/data/{session_id}/download/data",
                "intelligence_report": f"/api/data/{session_id}/download/intelligence"
            }
        )
    
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
                
                # Test if pipeline is serializable
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
            
            # Copy other sections as-is
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
                                # Don't include pipeline objects in metadata
                                safe_metadata[key] = f"<Pipeline object of type {type(value)}>"
                            elif key in ['dataframe', 'processed_data', 'final_data', 'enhanced_data']:
                                # Handle DataFrame objects
                                if hasattr(value, 'shape'):
                                    safe_metadata[key] = f"<DataFrame with shape {value.shape}>"
                                else:
                                    safe_metadata[key] = f"<Data object of type {type(value)}>"
                            elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict)):
                                # Convert objects to string representation
                                safe_metadata[key] = str(value)[:500] + "..." if len(str(value)) > 500 else str(value)
                            else:
                                # Try to include the value directly
                                try:
                                    json.dumps(value)  # Test if serializable
                                    safe_metadata[key] = value
                                except:
                                    # If not serializable, convert to string
                                    safe_metadata[key] = str(value)[:500] + "..." if len(str(value)) > 500 else str(value)
                        except Exception as key_error:
                            print(f"❌ Error processing key {key}: {str(key_error)}")
                            safe_metadata[key] = f"<Error processing key: {str(key_error)}>"
                else:
                    safe_metadata = {
                        "pipeline_result_type": str(type(pipeline_result)),
                        "pipeline_result_str": str(pipeline_result)[:1000] + "..." if len(str(pipeline_result)) > 1000 else str(pipeline_result)
                    }
                
                # Add session info
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
            # Look for enhanced data in multiple locations
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
        
        # Ensure we have intelligence profile
        if "intelligence_profile" not in session_store[session_id]:
            intelligence_profile = data_profiler.profile_dataset(df)
            session_store[session_id]["intelligence_profile"] = intelligence_profile
        else:
            intelligence_profile = session_store[session_id]["intelligence_profile"]
        
        # Initialize feature intelligence if not exists
        from .intelligence.feature_intelligence import AdvancedFeatureIntelligence
        feature_intelligence = AdvancedFeatureIntelligence()
        
        # Get feature engineering analysis
        fe_analysis = feature_intelligence.analyze_feature_engineering_opportunities(
            df, intelligence_profile, target_column
        )
        
        # Extract and filter recommendations
        recommendations = fe_analysis.get('feature_engineering_recommendations', [])
        
        # Filter by priority if specified
        if priority_filter:
            recommendations = [rec for rec in recommendations if rec.priority == priority_filter]
        
        # Limit number of recommendations
        recommendations = recommendations[:max_recommendations]
        
        # Convert to serializable format
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
        
        # Store analysis for later use
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
        # Initialize strategic control components
        from .core.project_definition import ProjectDefinition, Objective, Domain, Priority, RiskLevel
        from .core.strategy_translator import StrategyTranslator, TranslationStrategy
        
        # Convert API request to ProjectDefinition
        config = request.project_definition
        project_definition = ProjectDefinition(
            project_id=config.project_id,
            objective=Objective(config.objective.lower()),
            domain=Domain(config.domain.lower()),
            priority=Priority(config.priority.lower()),
            risk_level=RiskLevel(config.risk_level.lower()),
            business_context={
                'goal': config.business_goal,
                'success_criteria': config.success_criteria,
                'stakeholders': config.stakeholders,
                'timeline_months': config.timeline_months
            },
            technical_constraints={
                'max_latency_ms': config.max_latency_ms,
                'max_training_hours': config.max_training_hours,
                'min_accuracy': config.min_accuracy
            },
            regulatory_constraints={
                'interpretability_required': config.interpretability_required,
                'compliance_rules': config.compliance_rules
            }
        )
        
        # Initialize strategy translator with specified approach
        translation_strategy = TranslationStrategy(request.translation_strategy)
        strategy_translator = StrategyTranslator(translation_strategy)
        
        # Perform comprehensive strategic analysis
        technical_configuration = strategy_translator.translate(project_definition)
        pipeline_config = strategy_translator.translate_to_pipeline_config(project_definition)
        
        # Validate project definition
        validation_results = project_definition.validate_project_definition()
        
        # Calculate strategic scores and recommendations
        strategic_assessment = {
            "feasibility_score": _calculate_feasibility_score(project_definition, technical_configuration),
            "complexity_assessment": technical_configuration.model_complexity.value,
            "risk_mitigation_score": _calculate_risk_mitigation_score(project_definition, technical_configuration),
            "resource_optimization_score": _calculate_resource_optimization_score(technical_configuration),
            "compliance_readiness_score": _calculate_compliance_readiness_score(project_definition, technical_configuration)
        }
        
        # Generate executive recommendations
        executive_recommendations = _generate_executive_recommendations(
            project_definition, technical_configuration, strategic_assessment
        )
        
        # Strategic timeline estimation
        timeline_estimate = _estimate_strategic_timeline(project_definition, technical_configuration)
        
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
                "strategic_assessment": strategic_assessment,
                "technical_configuration": {
                    "recommended_algorithms": technical_configuration.recommended_algorithms,
                    "model_complexity": technical_configuration.model_complexity.value,
                    "ensemble_strategy": technical_configuration.ensemble_strategy,
                    "performance_targets": {
                        "accuracy_target": technical_configuration.accuracy_target,
                        "latency_target_ms": technical_configuration.latency_target_ms,
                        "throughput_target": technical_configuration.throughput_target
                    },
                    "resource_allocation": {
                        "max_training_hours": technical_configuration.max_training_time_hours,
                        "max_memory_gb": technical_configuration.max_memory_gb,
                        "compute_budget_priority": technical_configuration.compute_budget_priority
                    },
                    "feature_engineering": {
                        "complexity": technical_configuration.feature_engineering_complexity,
                        "selection_strategy": technical_configuration.feature_selection_strategy
                    },
                    "validation_strategy": {
                        "rigor": technical_configuration.validation_rigor,
                        "cross_validation_folds": technical_configuration.cross_validation_folds,
                        "test_set_percentage": technical_configuration.test_set_percentage
                    },
                    "interpretability": {
                        "level": technical_configuration.interpretability_level,
                        "methods": technical_configuration.explanation_methods
                    },
                    "security_compliance": {
                        "security_level": technical_configuration.security_level,
                        "audit_requirements": technical_configuration.audit_requirements,
                        "data_governance": technical_configuration.data_governance
                    }
                },
                "pipeline_configuration": pipeline_config,
                "executive_recommendations": executive_recommendations,
                "timeline_estimate": timeline_estimate,
                "validation_results": validation_results
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
    
    # Adjust for validation rigor
    if tech_config.validation_rigor == "regulatory":
        total_weeks *= 1.3
    elif tech_config.validation_rigor == "extensive":
        total_weeks *= 1.2
    
    # Adjust for interpretability requirements
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
        
        # Generate relationship graph
        from .intelligence.relationship_discovery import RelationshipDiscovery
        relationship_discovery = RelationshipDiscovery()
        
        # Convert dict relationships back to objects for graph generation
        relationship_objects = []
        for rel_dict in relationships:
            # Create a mock relationship object with required attributes
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

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatIntent(BaseModel):
    session_id: str
    user_message: str

class ChatResponse(BaseModel):
    intent: Dict[str, Any]
    strategy: Dict[str, Any]
    follow_up_questions: List[str]

@app.post("/api/chat/start_project")
async def start_project_conversation(request: ChatMessage):
    """Start project conversation with intent-to-strategy conversion"""
    try:
        # Use provided session_id, otherwise select the most recent session with data
        session_id = request.session_id
        if not session_id:
            try:
                candidates = [
                    (sid, data.get("created_at"))
                    for sid, data in session_store.items()
                    if isinstance(data, dict) and "dataframe" in data
                ]
                if candidates:
                    candidates.sort(key=lambda x: x[1] or "", reverse=True)
                    session_id = candidates[0][0]
                else:
                    session_id = f"chat_{int(datetime.now().timestamp())}"
            except Exception:
                session_id = f"chat_{int(datetime.now().timestamp())}"
        
        # Get data context if session exists
        data_context = None
        if session_id in session_store and "dataframe" in session_store[session_id]:
            df = session_store[session_id]["dataframe"]
            data_context = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_data": df.head(3).to_dict()
            }
        
        # Get RAG context for the query
        rag_context = rag_system.get_context_for_query(request.message, session_id)
        
        # Enhance data context with RAG information
        enhanced_context = data_context.copy() if data_context else {}
        enhanced_context['rag_context'] = rag_context
        
        # Extract intent from user message with RAG context
        intent_result = llm_interface.extract_intent(request.message, enhanced_context)
        
        # Convert intent to strategy
        strategy_result = llm_interface.convert_intent_to_strategy(intent_result, enhanced_context)
        
        # Generate follow-up questions
        follow_up_questions = llm_interface.generate_follow_up_questions(intent_result, enhanced_context)
        
        # Add to conversation history
        llm_interface.add_to_conversation("user", request.message, {"session_id": session_id})
        llm_interface.add_to_conversation("assistant", "Intent analyzed and strategy generated", {
            "intent": intent_result,
            "strategy": strategy_result
        })
        
        # Store chat context in session
        if session_id not in session_store:
            session_store[session_id] = {}
        
        session_store[session_id]["chat_context"] = {
            "conversation_history": llm_interface.get_conversation_context(),
            "current_intent": intent_result,
            "current_strategy": strategy_result,
            "last_updated": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "session_id": session_id,
            "response": {
                "intent": intent_result,
                "strategy": strategy_result,
                "follow_up_questions": follow_up_questions,
                "suggested_actions": _generate_action_suggestions(intent_result, strategy_result)
            },
            "chat_metadata": {
                "has_data": data_context is not None,
                "conversation_length": len(llm_interface.conversation_history),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Chat processing failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/chat/{session_id}/continue")
async def continue_conversation(session_id: str, request: ChatMessage):
    """Continue existing conversation with context awareness"""
    try:
        if session_id not in session_store:
            return {
                "status": "error",
                "error": "Session not found. Please start a new conversation."
            }
        
        # Get current chat context
        chat_context = session_store[session_id].get("chat_context", {})
        current_intent = chat_context.get("current_intent", {})
        current_strategy = chat_context.get("current_strategy", {})
        
        # Get data context
        data_context = None
        if "dataframe" in session_store[session_id]:
            df = session_store[session_id]["dataframe"]
            data_context = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        
        # Process follow-up message with context
        merged_context = {}
        if data_context:
            merged_context.update(data_context)
        merged_context.update({
            "previous_intent": current_intent,
            "previous_strategy": current_strategy
        })

        updated_intent = llm_interface.extract_intent(request.message, merged_context)
        
        # Update strategy based on refined intent
        updated_strategy = llm_interface.convert_intent_to_strategy(updated_intent, data_context)
        
        # Generate new follow-up questions
        follow_up_questions = llm_interface.generate_follow_up_questions(updated_intent, data_context)
        
        # Add to conversation
        llm_interface.add_to_conversation("user", request.message, {"session_id": session_id})
        llm_interface.add_to_conversation("assistant", "Intent refined and strategy updated", {
            "updated_intent": updated_intent,
            "updated_strategy": updated_strategy
        })
        
        # Update session
        session_store[session_id]["chat_context"] = {
            "conversation_history": llm_interface.get_conversation_context(),
            "current_intent": updated_intent,
            "current_strategy": updated_strategy,
            "last_updated": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "response": {
                "intent": updated_intent,
                "strategy": updated_strategy,
                "follow_up_questions": follow_up_questions,
                "changes": _compare_intent_and_strategy(current_intent, current_strategy, updated_intent, updated_strategy)
            },
            "chat_metadata": {
                "conversation_length": len(llm_interface.conversation_history),
                "context_updated": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Conversation continuation failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/chat/{session_id}/execute")
async def execute_chat_strategy(session_id: str):
    """Execute the strategy derived from chat conversation"""
    try:
        if session_id not in session_store:
            return {
                "status": "error",
                "error": "Session not found"
            }
        
        # Get chat context and strategy
        chat_context = session_store[session_id].get("chat_context", {})
        strategy = chat_context.get("current_strategy", {})
        
        if not strategy:
            return {
                "status": "error",
                "error": "No strategy found. Please complete the conversation first."
            }
        
        # Convert chat strategy to processing config
        processing_config = {
            "task": strategy.get("task_type", "classification"),
            "target_column": strategy.get("target_column"),
            "enable_feature_generation": strategy.get("configuration", {}).get("enable_feature_generation", False),
            "enable_feature_selection": strategy.get("configuration", {}).get("enable_feature_selection", False),
            "enable_intelligence": strategy.get("configuration", {}).get("enable_intelligence", True)
        }
        
        # Store the config for processing
        session_store[session_id]["processing_config"] = processing_config
        
        # Execute the pipeline using the derived strategy
        if "dataframe" not in session_store[session_id]:
            return {
                "status": "error", 
                "error": "No data found. Please upload data first."
            }
        
        df = session_store[session_id]["dataframe"]
        
        # Create temporary file for processing
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        
        # Execute pipeline with robust orchestrator
        config = PipelineConfig(
            max_memory_usage=0.8,
            enable_caching=True,
            auto_recovery=True
        )
        
        robust_orchestrator = RobustPipelineOrchestrator(config)
        
        result = robust_orchestrator.execute_pipeline(
            data_path=temp_file.name,
            task_type=processing_config["task"],
            target_column=processing_config.get("target_column"),
            enable_feature_generation=processing_config["enable_feature_generation"],
            enable_feature_selection=processing_config["enable_feature_selection"],
            enable_intelligence=processing_config["enable_intelligence"]
        )
        
        # Clean up temp file
        Path(temp_file.name).unlink()
        
        # Store results
        session_store[session_id]["pipeline_result"] = result
        
        # Add execution report to RAG system
        rag_system.add_execution_report(result, session_id)
        
        # Add execution to conversation
        llm_interface.add_to_conversation("system", "Pipeline executed successfully", {
            "execution_result": {"status": result.get("status"), "execution_id": result.get("execution_id")}
        })
        
        return {
            "status": "success",
            "execution_result": result,
            "message": "Strategy executed successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Strategy execution failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/chat/{session_id}/explain")
async def explain_results(session_id: str, question: Optional[str] = None):
    """Explain pipeline results in natural language"""
    try:
        if session_id not in session_store:
            return {
                "status": "error",
                "error": "Session not found"
            }
        
        # Get pipeline results
        pipeline_result = session_store[session_id].get("pipeline_result")
        if not pipeline_result:
            return {
                "status": "error",
                "error": "No results to explain. Please execute a pipeline first."
            }
        
        # Generate explanation
        explanation = llm_interface.generate_explanation(pipeline_result, question)
        
        # Add to conversation
        llm_interface.add_to_conversation("user", question or "Explain the results", {"session_id": session_id})
        llm_interface.add_to_conversation("assistant", explanation, {"type": "explanation"})
        
        return {
            "status": "success",
            "explanation": explanation,
            "result_summary": {
                "status": pipeline_result.get("status"),
                "execution_time": pipeline_result.get("execution_summary", {}).get("total_time"),
                "stages_completed": pipeline_result.get("execution_summary", {}).get("successful_stages"),
                "data_shape": pipeline_result.get("results", {}).get("processed_data", {}).get("shape") if isinstance(pipeline_result.get("results", {}), dict) else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Explanation generation failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/chat/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    try:
        if session_id not in session_store:
            return {
                "status": "error",
                "error": "Session not found"
            }
        
        chat_context = session_store[session_id].get("chat_context", {})
        conversation_history = chat_context.get("conversation_history", {})
        
        return {
            "status": "success",
            "conversation_history": conversation_history,
            "current_intent": chat_context.get("current_intent"),
            "current_strategy": chat_context.get("current_strategy"),
            "session_metadata": {
                "last_updated": chat_context.get("last_updated"),
                "has_data": "dataframe" in session_store[session_id],
                "has_results": "pipeline_result" in session_store[session_id]
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"History retrieval failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def _generate_action_suggestions(intent: Dict, strategy: Dict) -> List[str]:
    """Generate actionable suggestions based on intent and strategy"""
    suggestions = []
    
    if intent.get("clarification_needed", False):
        suggestions.append("Please provide more details about your target variable or analysis goals")
    
    if not strategy.get("target_column"):
        suggestions.append("Specify which column you want to predict or analyze")
    
    if strategy.get("task_type") == "classification":
        suggestions.append("Consider enabling feature generation for better classification performance")
    elif strategy.get("task_type") == "regression":
        suggestions.append("Feature selection might help improve regression accuracy")
    
    if intent.get("confidence", 0) < 0.7:
        suggestions.append("Consider providing more specific details about your analysis requirements")
    
    return suggestions

def _compare_intent_and_strategy(old_intent: Dict, old_strategy: Dict, new_intent: Dict, new_strategy: Dict) -> Dict:
    """Compare old and new intent/strategy to highlight changes"""
    changes = {
        "intent_changes": [],
        "strategy_changes": []
    }
    
    # Compare intent changes
    if old_intent.get("task_type") != new_intent.get("task_type"):
        changes["intent_changes"].append(f"Task type changed from {old_intent.get('task_type')} to {new_intent.get('task_type')}")
    
    if old_intent.get("target_variable") != new_intent.get("target_variable"):
        changes["intent_changes"].append(f"Target variable updated to {new_intent.get('target_variable')}")
    
    # Compare strategy changes
    if old_strategy.get("target_column") != new_strategy.get("target_column"):
        changes["strategy_changes"].append(f"Target column updated to {new_strategy.get('target_column')}")
    
    old_config = old_strategy.get("configuration", {})
    new_config = new_strategy.get("configuration", {})
    
    for key in ["enable_feature_generation", "enable_feature_selection", "enable_intelligence"]:
        if old_config.get(key) != new_config.get(key):
            changes["strategy_changes"].append(f"{key} changed to {new_config.get(key)}")
    
    return changes

# RAG System Endpoints

@app.post("/api/rag/add_document")
async def add_document_to_rag(content: str = Form(), metadata: str = Form(default="{}"), 
                             document_type: str = Form(default="general"), session_id: Optional[str] = Form(default=None)):
    """Add a document to the RAG system"""
    try:
        metadata_dict = json.loads(metadata) if metadata else {}
        
        doc_id = rag_system.add_document(
            content=content,
            metadata=metadata_dict,
            document_type=document_type,
            session_id=session_id
        )
        
        return {
            "status": "success" if doc_id else "failed",
            "document_id": doc_id,
            "message": "Document added successfully" if doc_id else "Failed to add document"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Document addition failed: {str(e)}"
        }

@app.get("/api/rag/search")
async def search_rag_documents(query: str, top_k: int = 5, session_id: Optional[str] = None):
    """Search for relevant documents in RAG system"""
    try:
        results = rag_system.retrieve_relevant_documents(query, top_k, session_id)
        
        search_results = []
        for result in results:
            search_results.append({
                "document_id": result.document.id,
                "content": result.document.content[:300] + "..." if len(result.document.content) > 300 else result.document.content,
                "metadata": result.document.metadata,
                "score": result.score,
                "relevance": result.relevance,
                "timestamp": result.document.timestamp
            })
        
        return {
            "status": "success",
            "query": query,
            "results": search_results,
            "total_results": len(search_results)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Search failed: {str(e)}"
        }

@app.get("/api/rag/{session_id}/context")
async def get_session_rag_context(session_id: str):
    """Get all RAG context for a specific session"""
    try:
        context = rag_system.get_session_context(session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "context": context
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Context retrieval failed: {str(e)}"
        }

@app.get("/api/rag/analytics")
async def get_rag_analytics():
    """Get RAG system analytics and statistics"""
    try:
        analytics = rag_system.get_analytics()
        
        return {
            "status": "success",
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Analytics generation failed: {str(e)}"
        }

@app.post("/api/rag/cleanup")
async def cleanup_rag_documents(days: int = 30):
    """Clean up old RAG documents"""
    try:
        deleted_count = rag_system.cleanup_old_documents(days)
        
        return {
            "status": "success",
            "deleted_documents": deleted_count,
            "retention_days": days,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Cleanup failed: {str(e)}"
        }

# ===== KNOWLEDGE GRAPH API ENDPOINTS =====

try:
    from .knowledge_graph.service import KnowledgeGraphService
    from .knowledge_graph.query_translator import NaturalLanguageQueryTranslator, QueryResult
    KNOWLEDGE_GRAPH_AVAILABLE = True
    
    # Initialize knowledge graph service and translator
    knowledge_graph_service = None
    query_translator = None
    
    def get_knowledge_graph_service():
        global knowledge_graph_service, query_translator
        if knowledge_graph_service is None and DATABASE_AVAILABLE:
            try:
                from .knowledge_graph.service import Neo4jGraphDatabase, PostgresGraphDatabase
                
                # Use Neo4j by default, fallback to PostgreSQL
                try:
                    db = Neo4jGraphDatabase(
                        uri="bolt://localhost:7687",
                        username="neo4j", 
                        password="password"
                    )
                    knowledge_graph_service = KnowledgeGraphService(db)
                    if knowledge_graph_service.initialize():
                        query_translator = NaturalLanguageQueryTranslator(llm_interface)
                        return knowledge_graph_service
                except Exception:
                    # Fallback to PostgreSQL
                    db = PostgresGraphDatabase("postgresql://user:password@localhost:5432/datainsight")
                    knowledge_graph_service = KnowledgeGraphService(db)
                    if knowledge_graph_service.initialize():
                        query_translator = NaturalLanguageQueryTranslator(llm_interface)
                        return knowledge_graph_service
            except Exception as e:
                print(f"Failed to initialize knowledge graph: {e}")
        
        return knowledge_graph_service

except ImportError:
    KNOWLEDGE_GRAPH_AVAILABLE = False


class KnowledgeQueryRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    include_explanation: bool = True


@app.post("/api/knowledge/query")
async def query_knowledge_graph(request: KnowledgeQueryRequest):
    """Query the knowledge graph using natural language"""
    if not KNOWLEDGE_GRAPH_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="Knowledge graph functionality not available"
        )
    
    kg_service = get_knowledge_graph_service()
    if not kg_service or not query_translator:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph service not initialized"
        )
    
    try:
        start_time = datetime.now()
        
        # Translate natural language to graph query
        graph_query = query_translator.translate_query(
            request.query, 
            database_type="neo4j"  # or detect from service
        )
        
        if not graph_query:
            return {
                "status": "error",
                "error": "Could not translate query to graph format",
                "query": request.query
            }
        
        # Execute the query
        results = kg_service.database.execute_query(graph_query)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Generate natural language response
        natural_response = ""
        if request.include_explanation:
            natural_response = query_translator.synthesize_natural_response(
                graph_query, results, request.query
            )
        
        return {
            "status": "success",
            "original_query": request.query,
            "graph_query": graph_query,
            "results": results[:request.limit],
            "result_count": len(results),
            "execution_time": execution_time,
            "natural_language_response": natural_response,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Knowledge graph query failed: {str(e)}",
            "query": request.query
        }


@app.get("/api/knowledge/insights")
async def get_knowledge_insights():
    """Get high-level insights from the knowledge graph"""
    if not KNOWLEDGE_GRAPH_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Knowledge graph functionality not available"
        )
    
    kg_service = get_knowledge_graph_service()
    if not kg_service:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph service not initialized"
        )
    
    try:
        insights = {}
        
        # Get dataset statistics
        dataset_query = """
        MATCH (d:Dataset) 
        RETURN count(d) as total_datasets, 
               avg(d.quality_score) as avg_quality_score,
               collect(DISTINCT d.domain) as domains
        """
        dataset_stats = kg_service.database.execute_query(dataset_query)
        insights["datasets"] = dataset_stats[0] if dataset_stats else {}
        
        # Get model performance statistics
        model_query = """
        MATCH (m:Model)
        RETURN count(m) as total_models,
               avg(m.trust_score) as avg_trust_score,
               collect(DISTINCT m.algorithm) as algorithms
        """
        model_stats = kg_service.database.execute_query(model_query)
        insights["models"] = model_stats[0] if model_stats else {}
        
        # Get project statistics
        project_query = """
        MATCH (p:Project)
        RETURN count(p) as total_projects,
               count(CASE WHEN p.status = 'completed' THEN 1 END) as completed_projects,
               collect(DISTINCT p.domain) as project_domains
        """
        project_stats = kg_service.database.execute_query(project_query)
        insights["projects"] = project_stats[0] if project_stats else {}
        
        # Get top performing features
        feature_query = """
        MATCH (f:Feature)
        WHERE f.importance_score IS NOT NULL
        RETURN f.name, f.importance_score, f.generation_method
        ORDER BY f.importance_score DESC
        LIMIT 10
        """
        top_features = kg_service.database.execute_query(feature_query)
        insights["top_features"] = top_features
        
        return {
            "status": "success",
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to generate insights: {str(e)}"
        }


@app.get("/api/knowledge/recommendations/{project_domain}")
async def get_domain_recommendations(project_domain: str, limit: int = 5):
    """Get recommendations for a specific project domain"""
    if not KNOWLEDGE_GRAPH_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Knowledge graph functionality not available"
        )
    
    kg_service = get_knowledge_graph_service()
    if not kg_service:
        raise HTTPException(
            status_code=503,
            detail="Knowledge graph service not initialized"
        )
    
    try:
        # Get successful models for this domain
        model_query = f"""
        MATCH (m:Model)-[:APPLIES_TO]->(p:Project)
        WHERE p.domain = '{project_domain}' AND p.status = 'completed'
        RETURN m.algorithm, m.performance_metrics, m.trust_score, p.name
        ORDER BY m.trust_score DESC
        LIMIT {limit}
        """
        
        successful_models = kg_service.database.execute_query(model_query)
        
        # Get important features for this domain
        feature_query = f"""
        MATCH (f:Feature)-[:TRAINED_ON]->(m:Model)-[:APPLIES_TO]->(p:Project)
        WHERE p.domain = '{project_domain}' AND f.importance_score IS NOT NULL
        RETURN f.name, avg(f.importance_score) as avg_importance, f.generation_method
        ORDER BY avg_importance DESC
        LIMIT {limit}
        """
        
        important_features = kg_service.database.execute_query(feature_query)
        
        return {
            "status": "success",
            "domain": project_domain,
            "recommendations": {
                "successful_models": successful_models,
                "important_features": important_features
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to generate recommendations: {str(e)}"
        }


@app.get("/api/knowledge/schema")
async def get_knowledge_graph_schema():
    """Get the knowledge graph schema information"""
    if not KNOWLEDGE_GRAPH_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Knowledge graph functionality not available"
        )
    
    try:
        from .knowledge_graph.schema import GraphSchema, NodeType, RelationshipType
        
        schema = GraphSchema()
        
        return {
            "status": "success",
            "schema": {
                "node_types": [node_type.value for node_type in NodeType],
                "relationship_types": [rel_type.value for rel_type in RelationshipType],
                "relationship_constraints": schema.get_relationship_constraints()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "error": f"Failed to get schema: {str(e)}"
        }

# Enhanced data processing endpoints with RAG integration

@app.post("/api/data/{session_id}/profile")
async def generate_intelligence_profile_with_rag(session_id: str, request: ProfilingRequest):
    """Generate comprehensive intelligent data profile with RAG storage"""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        df = session_store[session_id]["dataframe"]
        
        # Generate intelligence profile
        if "intelligence_profile" in session_store[session_id] and not request.deep_analysis:
            intelligence_profile = session_store[session_id]["intelligence_profile"]
        else:
            intelligence_profile = data_profiler.profile_dataset(df)
            session_store[session_id]["intelligence_profile"] = intelligence_profile
        
        # Add intelligence profile to RAG system
        rag_system.add_intelligence_profile(intelligence_profile, session_id)
        
        # Filter response based on request parameters (original logic)
        response_data = {}
        
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
        
        if request.include_domain_detection:
            domain_analysis = intelligence_profile.get('domain_analysis', {})
            response_data['domain_analysis'] = domain_analysis
        
        if request.include_relationships:
            relationship_analysis = intelligence_profile.get('relationship_analysis', {})
            response_data['relationship_analysis'] = relationship_analysis
        
        response_data['overall_recommendations'] = intelligence_profile.get('overall_recommendations', [])
        
        response_data['profiling_metadata'] = {
            'deep_analysis': request.deep_analysis,
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'profile_timestamp': datetime.now().isoformat(),
            'rag_stored': True
        }
        
        return {
            "status": "success",
            "intelligence_profile": response_data,
            "session_id": session_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Application health check endpoint."""
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
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)