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

from .core.pipeline_orchestrator import RobustPipelineOrchestrator, PipelineConfig
from .core.project_definition import ProjectDefinition, Objective, Domain, RiskLevel, Constraints
from .automator import Status
from .common.data_ingestion import ingest_data, ingest_from_url
from .data_quality.validator import DataQualityValidator
from .utils import generate_eda_report
from .intelligence.data_profiler import IntelligentDataProfiler

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
    enable_robust_pipeline: bool = True

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

# In-memory storage (in production, use Redis/database)
session_store = {}

# Initialize intelligence components
data_profiler = IntelligentDataProfiler()

# Additional models for new endpoints
class ProfilingRequest(BaseModel):
    deep_analysis: bool = True
    include_relationships: bool = True
    include_domain_detection: bool = True

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
    """Serve the main application interface."""
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
            "filename": file.filename
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
            "source_url": request.url
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

@app.post("/api/data/{session_id}/process-strategy")
async def process_data_strategy(session_id: str, strategy: StrategyConfig):
    """Strategy-driven data processing using business objectives"""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        start_time = datetime.now()
        session_data = session_store[session_id]
        df = session_data["dataframe"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            data_path = temp_file.name
        
        try:
            constraints = Constraints(
                max_latency_ms=strategy.max_latency_ms,
                max_training_hours=strategy.max_training_hours,
                min_accuracy=strategy.min_accuracy,
                protected_attributes=strategy.protected_attributes,
                interpretability_required=strategy.interpretability_required,
                compliance_rules=strategy.compliance_rules
            )
            
            project_definition = ProjectDefinition(
                objective=Objective(strategy.objective),
                domain=Domain(strategy.domain),
                risk_level=RiskLevel(strategy.risk_level),
                constraints=constraints
            )
            
            orchestrator = RobustPipelineOrchestrator(project_def=project_definition)
            
            result_dict = orchestrator.execute_pipeline(
                data_path=data_path,
                target_column=strategy.target_column,
                project_definition=project_definition
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            strategy_context = result_dict.get('strategy_context', {})
            
            return {
                "status": "success",
                "message": "Strategy-driven processing completed",
                "processing_time": processing_time,
                "strategy_applied": {
                    "objective": strategy.objective,
                    "domain": strategy.domain,
                    "objective_met": strategy_context.get('objective_met', True),
                    "constraints_satisfied": strategy_context.get('constraints_satisfied', True),
                    "trade_offs": strategy_context.get('trade_offs_made', []),
                    "business_rationale": strategy_context.get('business_rationale', '')
                },
                "results": result_dict,
                "execution_history": result_dict.get('execution_history', [])
            }
            
        finally:
            try:
                Path(data_path).unlink()
            except:
                pass
                
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Strategy processing failed: {str(e)}",
                "error_type": type(e).__name__
            }
        )

@app.post("/api/data/{session_id}/process")
async def process_data(session_id: str, config: TaskConfig):
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        start_time = datetime.now()
        session_data = session_store[session_id]
        df = session_data["dataframe"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            df.to_csv(temp_file.name, index=False)
            data_path = temp_file.name
        
        try:
            pipeline_config = PipelineConfig(
                enable_explainability=config.enable_intelligence,
                enable_bias_detection=config.enable_intelligence,
                enable_security=True,
                enable_privacy_protection=True,
                enable_compliance=True,
                enable_adaptive_learning=config.enable_intelligence,
                enable_feature_engineering=config.feature_generation_enabled,
                enable_feature_selection=config.feature_selection_enabled,
                max_memory_usage=0.8,
                enable_caching=True,
                checkpoint_enabled=True,
                auto_recovery=True
            )
            
            orchestrator = RobustPipelineOrchestrator(pipeline_config)
            
            custom_config = {
                'feature_generation_enabled': config.feature_generation_enabled,
                'feature_selection_enabled': config.feature_selection_enabled
            }
            
            result_dict = orchestrator.execute_pipeline(
                data_path=data_path,
                task_type=config.task,
                target_column=config.target_column,
                custom_config=custom_config
            )
            
        finally:
            Path(data_path).unlink()
        
        if result_dict.get('status') != 'success':
            error_msg = result_dict.get('error', 'Unknown pipeline error')
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {error_msg}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        session_store[session_id].update({
            "robust_pipeline_result": result_dict,
            "processing_config": config.dict(),
            "processing_time": processing_time
        })
        
        results = result_dict.get('results', {})
        execution_summary = result_dict.get('execution_summary', {})
        
        profiling_result = results.get('data_profiling', {})
        model_result = results.get('model_selection', {})
        security_result = results.get('security_scan', {})
        
        intelligence_summary = {
            "detected_domain": profiling_result.get('metadata', {}).get('detected_domain', 'unknown'),
            "features_analyzed": len(df.columns),
            "feature_generation_applied": config.feature_generation_enabled,
            "feature_selection_applied": config.feature_selection_enabled,
            "model_selected": model_result.get('metadata', {}).get('best_algorithm', 'N/A'),
            "model_score": model_result.get('metadata', {}).get('best_score', 0),
            "security_score": security_result.get('metadata', {}).get('security_score', 0),
            "stages_completed": execution_summary.get('stages_completed', 0),
            "total_time": execution_summary.get('total_time', processing_time),
            "original_shape": df.shape,
            "processing_successful": True
        }
        
        return ProcessingResult(
            status="success",
            message="Data processed successfully with comprehensive AI pipeline",
            data_shape=df.shape,
            processing_time=processing_time,
            intelligence_summary=intelligence_summary,
            artifacts={
                "pipeline_metadata": f"/api/data/{session_id}/download/metadata",
                "processed_data": f"/api/data/{session_id}/download/data",
                "intelligence_report": f"/api/data/{session_id}/download/intelligence",
                "model_artifacts": f"/api/data/{session_id}/download/model"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        logging.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

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
            # Check both possible locations for processed data
            processed_data = None
            if "processed_data" in session_data:
                processed_data = session_data["processed_data"]
            elif "robust_pipeline_result" in session_data:
                robust_result = session_data["robust_pipeline_result"]
                if isinstance(robust_result, dict) and "final_data" in robust_result:
                    processed_data = robust_result["final_data"]
                elif isinstance(robust_result, dict) and "processed_data" in robust_result:
                    processed_data = robust_result["processed_data"]
            
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
            if "orchestrator_result" not in session_data:
                raise HTTPException(status_code=404, detail="Pipeline not found")
            
            try:
                # Get the pipeline from orchestrator result
                orchestrator_result = session_data["orchestrator_result"]
                print(f"🔧 Orchestrator result type: {type(orchestrator_result)}")
                print(f"🔧 Orchestrator result keys: {orchestrator_result.keys() if hasattr(orchestrator_result, 'keys') else 'Not dict-like'}")
                
                # Try to get pipeline - could be in different locations
                pipeline = None
                if hasattr(orchestrator_result, 'pipeline'):
                    pipeline = orchestrator_result.pipeline
                elif isinstance(orchestrator_result, dict) and 'pipeline' in orchestrator_result:
                    pipeline = orchestrator_result['pipeline']
                elif isinstance(orchestrator_result, dict) and 'final_pipeline' in orchestrator_result:
                    pipeline = orchestrator_result['final_pipeline']
                
                if pipeline is None:
                    # If no pipeline found, create a placeholder
                    print("⚠️ No pipeline object found, creating placeholder")
                    placeholder_data = {
                        "session_id": session_id,
                        "message": "Pipeline object not available - processing may have used a different approach",
                        "orchestrator_keys": list(orchestrator_result.keys()) if isinstance(orchestrator_result, dict) else str(type(orchestrator_result)),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    return Response(
                        content=json.dumps(placeholder_data, indent=2).encode(),
                        media_type="application/json",
                        headers={"Content-Disposition": f"attachment; filename=pipeline_info_{session_id}.json"}
                    )
                
                print(f"🔧 Found pipeline object of type: {type(pipeline)}")
                
                # Test if pipeline is serializable
                try:
                    # Use a temporary file approach instead of BytesIO to avoid memory issues
                    import tempfile
                    import os
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
                        print(f"🔧 Creating temporary pipeline file: {tmp_file.name}")
                        joblib.dump(pipeline, tmp_file.name)
                        
                        # Read the file back as bytes
                        with open(tmp_file.name, 'rb') as f:
                            pipeline_bytes = f.read()
                        
                        # Clean up temp file
                        os.unlink(tmp_file.name)
                        
                        print(f"🔧 Pipeline serialization successful, size: {len(pipeline_bytes)} bytes")
                        
                        return Response(
                            content=pipeline_bytes,
                            media_type="application/octet-stream",
                            headers={
                                "Content-Disposition": f"attachment; filename=pipeline_{session_id}.joblib",
                                "Content-Length": str(len(pipeline_bytes))
                            }
                        )
                except Exception as serialize_error:
                    print(f"❌ Pipeline serialization failed: {str(serialize_error)}")
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
                print(f"❌ Pipeline download error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Pipeline serialization failed: {str(e)}")
        
        elif artifact_type == "lineage":
            if "orchestrator_result" not in session_data:
                raise HTTPException(status_code=404, detail="Lineage report not found")
            
            try:
                orchestrator_result = session_data["orchestrator_result"]
                
                # Try different ways to get lineage data
                lineage = None
                if hasattr(orchestrator_result, 'lineage_report'):
                    lineage = orchestrator_result.lineage_report
                elif isinstance(orchestrator_result, dict) and 'lineage_report' in orchestrator_result:
                    lineage = orchestrator_result['lineage_report']
                elif isinstance(orchestrator_result, dict) and 'lineage' in orchestrator_result:
                    lineage = orchestrator_result['lineage']
                else:
                    # Generate basic lineage from available data
                    lineage = {
                        "pipeline_execution": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "steps_executed": list(orchestrator_result.keys()) if isinstance(orchestrator_result, dict) else [],
                        "session_id": session_id
                    }
                
                lineage_json = json.dumps(lineage, indent=2, default=str)
                
                return Response(
                    content=lineage_json.encode(),
                    media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename=lineage_report_{session_id}.json"}
                )
            except Exception as e:
                print(f"❌ Lineage download error: {str(e)}")
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
        
        elif artifact_type == "robust-metadata":
            if "robust_pipeline_result" not in session_data:
                raise HTTPException(status_code=404, detail="Robust pipeline metadata not found")
            
            try:
                robust_result = session_data["robust_pipeline_result"]
                print(f"🔧 Robust result type: {type(robust_result)}")
                print(f"🔧 Robust result keys: {list(robust_result.keys()) if isinstance(robust_result, dict) else 'Not a dict'}")
                
                # Create a safe, serializable version of the metadata
                safe_metadata = {}
                
                if isinstance(robust_result, dict):
                    for key, value in robust_result.items():
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
                        "robust_result_type": str(type(robust_result)),
                        "robust_result_str": str(robust_result)[:1000] + "..." if len(str(robust_result)) > 1000 else str(robust_result)
                    }
                
                # Add session info
                safe_metadata["session_id"] = session_id
                safe_metadata["export_timestamp"] = datetime.now().isoformat()
                
                metadata_json = json.dumps(safe_metadata, indent=2, default=str)
                
                return Response(
                    content=metadata_json.encode(),
                    media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename=robust_pipeline_metadata_{session_id}.json"}
                )
            except Exception as e:
                print(f"❌ Metadata serialization error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Metadata export failed: {str(e)}")
        
        elif artifact_type == "enhanced-data":
            # Look for enhanced data in multiple locations
            enhanced_df = None
            
            if "enhanced_data" in session_data:
                enhanced_df = session_data["enhanced_data"]
            elif "robust_pipeline_result" in session_data:
                robust_result = session_data["robust_pipeline_result"]
                if isinstance(robust_result, dict):
                    if "enhanced_data" in robust_result:
                        enhanced_df = robust_result["enhanced_data"]
                    elif "final_data" in robust_result:
                        enhanced_df = robust_result["final_data"] 
                    elif "processed_data" in robust_result:
                        enhanced_df = robust_result["processed_data"]
            
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
        # Initialize adaptive learning system
        from .learning.adaptive_system import AdaptiveLearningSystem
        learning_system = AdaptiveLearningSystem()
        
        # Get session data for context
        if feedback.session_id in session_store:
            session_data = session_store[feedback.session_id]
            intelligence_profile = session_data.get("intelligence_profile", {})
            processing_config = session_data.get("processing_config", {})
            
            # Create mock pipeline results for learning
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
            
            # Record execution for learning
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
            # Still record feedback even without session context
            return {
                "status": "success",
                "message": "Feedback recorded (limited context)",
                "feedback_id": f"feedback_{feedback.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        
        # Check for robust pipeline results
        robust_result = session_data.get("robust_pipeline_result")
        legacy_result = session_data.get("orchestrator_result")
        
        if robust_result:
            # Extract status from robust pipeline
            execution_summary = robust_result.get('execution_summary', {})
            
            status_info = {
                "pipeline_type": "robust",
                "overall_status": robust_result.get('status', 'unknown'),
                "session_id": robust_result.get('session_id'),
                "total_stages": execution_summary.get('total_stages', 0),
                "successful_stages": execution_summary.get('successful_stages', 0),
                "total_execution_time": execution_summary.get('total_time', 0),
                "stage_details": robust_result.get('results', {}),
                "last_updated": datetime.now().isoformat()
            }
            
        elif legacy_result:
            # Extract status from legacy pipeline
            status_info = {
                "pipeline_type": "legacy",
                "overall_status": legacy_result.status.value if hasattr(legacy_result.status, 'value') else str(legacy_result.status),
                "processing_time": session_data.get("processing_time", 0),
                "data_shape": legacy_result.processed_features.shape if hasattr(legacy_result, 'processed_features') else None,
                "column_roles": legacy_result.column_roles if hasattr(legacy_result, 'column_roles') else {},
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
        robust_result = session_data.get("robust_pipeline_result")
        
        if not robust_result:
            raise HTTPException(status_code=400, detail="No robust pipeline execution found to recover")
        
        if robust_result.get('status') != 'pipeline_failure':
            return {
                "status": "success",
                "message": "Pipeline is not in failed state, no recovery needed",
                "current_status": robust_result.get('status')
            }
        
        # Attempt recovery by re-running with more conservative settings
        df = session_data["dataframe"]
        
        # Create temporary file for recovery attempt
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        df.to_csv(temp_file.name, index=False)
        
        # Initialize robust orchestrator with recovery configuration
        from .core.pipeline_orchestrator import RobustPipelineOrchestrator, PipelineConfig
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
        session_data["robust_pipeline_result"] = recovery_result
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

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)