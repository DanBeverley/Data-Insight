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
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .automator import WorkflowOrchestrator, Status
from .common.data_ingestion import ingest_data, ingest_from_url
from .data_quality.validator import DataQualityValidator
from .utils import generate_eda_report
from .intelligence.data_profiler import IntelligentDataProfiler
from .core.pipeline_orchestrator import RobustPipelineOrchestrator

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

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Data models
class TaskConfig(BaseModel):
    task: str
    target_column: Optional[str] = None
    feature_generation_enabled: bool = False
    feature_selection_enabled: bool = False
    enable_intelligence: bool = True
    enable_robust_pipeline: bool = True

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

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application interface."""
    return HTMLResponse(open("static/index.html").read())

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
                    {"name": check.name, "message": check.message, "passed": check.passed}
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
                    {"name": check.name, "message": check.message, "passed": check.passed}
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
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        df = session_store[session_id]["dataframe"]
        eda_report = generate_eda_report(df)
        
        # Store EDA report
        session_store[session_id]["eda_report"] = eda_report
        
        return {
            "status": "success",
            "report": eda_report,
            "message": "EDA report generated successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating EDA: {str(e)}")

@app.post("/api/data/{session_id}/process")
async def process_data(session_id: str, config: TaskConfig):
    """Process data with intelligent automated pipeline."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        start_time = datetime.now()
        session_data = session_store[session_id]
        df = session_data["dataframe"]
        
        # Determine which orchestrator to use
        if config.enable_robust_pipeline:
            # Use new robust pipeline orchestrator
            try:
                # Create temporary file for data path (robust orchestrator expects file path)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                df.to_csv(temp_file.name, index=False)
                
                robust_orchestrator = RobustPipelineOrchestrator()
                
                # Get intelligence profile if available
                intelligence_profile = session_data.get("intelligence_profile")
                
                # Execute robust pipeline
                robust_result = robust_orchestrator.execute_pipeline(
                    data_path=temp_file.name,
                    task_type=config.task,
                    target_column=config.target_column
                )
                
                # Clean up temp file
                Path(temp_file.name).unlink()
                
                if robust_result.get('status') != 'success':
                    raise Exception(f"Robust pipeline failed: {robust_result.get('error', 'Unknown error')}")
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # Store robust results
                session_store[session_id].update({
                    "robust_pipeline_result": robust_result,
                    "processing_config": config.dict(),
                    "processing_time": processing_time
                })
                
                # Extract intelligence summary if available
                intelligence_summary = None
                if intelligence_profile:
                    domain_analysis = intelligence_profile.get('domain_analysis', {})
                    detected_domains = domain_analysis.get('detected_domains', [])
                    intelligence_summary = {
                        "primary_domain": detected_domains[0].get('domain') if detected_domains else 'unknown',
                        "total_recommendations": len(intelligence_profile.get('overall_recommendations', [])),
                        "relationships_analyzed": len(intelligence_profile.get('relationship_analysis', {}).get('relationships', [])),
                        "feature_engineering_applied": True
                    }
                
                return ProcessingResult(
                    status="success",
                    message="Data processed with robust intelligent pipeline",
                    data_shape=df.shape,  # Simplified for now
                    processing_time=processing_time,
                    intelligence_summary=intelligence_summary,
                    artifacts={
                        "pipeline_metadata": f"/api/data/{session_id}/download/robust-metadata",
                        "processed_data": f"/api/data/{session_id}/download/data",
                        "intelligence_report": f"/api/data/{session_id}/download/intelligence"
                    }
                )
                
            except Exception as robust_error:
                # Fallback to legacy orchestrator if robust fails
                print(f"Robust pipeline failed, falling back to legacy: {robust_error}")
                config.enable_robust_pipeline = False
        
        # Use legacy orchestrator (backward compatibility)
        orchestrator = WorkflowOrchestrator(
            df=df,
            target_column=config.target_column,
            task=config.task
        )
        
        result = orchestrator.run()
        
        if result.status != Status.SUCCESS:
            raise HTTPException(status_code=500, detail=result.error_message)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store legacy results
        session_store[session_id].update({
            "orchestrator_result": result,
            "processed_data": result.processed_features,
            "aligned_target": result.aligned_target,
            "processing_config": config.dict()
        })
        
        return ProcessingResult(
            status="success",
            message="Data processed successfully (legacy pipeline)",
            data_shape=result.processed_features.shape,
            column_roles=result.column_roles,
            processing_time=processing_time,
            artifacts={
                "pipeline": f"/api/data/{session_id}/download/pipeline",
                "processed_data": f"/api/data/{session_id}/download/data",
                "lineage_report": f"/api/data/{session_id}/download/lineage"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/api/data/{session_id}/download/{artifact_type}")
async def download_artifact(session_id: str, artifact_type: str):
    """Download processing artifacts including intelligence reports."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = session_store[session_id]
    
    try:
        if artifact_type == "data":
            if "processed_data" not in session_data:
                raise HTTPException(status_code=404, detail="Processed data not found")
            
            df = session_data["processed_data"]
            if session_data.get("aligned_target") is not None:
                df = df.copy()
                df["target"] = session_data["aligned_target"]
            
            csv_data = df.to_csv(index=False)
            
            return FileResponse(
                path=None,
                filename=f"processed_data_{session_id}.csv",
                content=csv_data.encode(),
                media_type="text/csv"
            )
        
        elif artifact_type == "pipeline":
            if "orchestrator_result" not in session_data:
                raise HTTPException(status_code=404, detail="Pipeline not found")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                joblib.dump(session_data["orchestrator_result"].pipeline, tmp.name)
                return FileResponse(
                    path=tmp.name,
                    filename=f"pipeline_{session_id}.joblib",
                    media_type="application/octet-stream"
                )
        
        elif artifact_type == "lineage":
            if "orchestrator_result" not in session_data:
                raise HTTPException(status_code=404, detail="Lineage report not found")
            
            lineage = session_data["orchestrator_result"].lineage_report
            lineage_json = json.dumps(lineage, indent=2)
            
            return FileResponse(
                path=None,
                filename=f"lineage_report_{session_id}.json",
                content=lineage_json.encode(),
                media_type="application/json"
            )
        
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
            
            return FileResponse(
                path=None,
                filename=f"intelligence_report_{session_id}.json",
                content=intelligence_json.encode(),
                media_type="application/json"
            )
        
        elif artifact_type == "robust-metadata":
            if "robust_pipeline_result" not in session_data:
                raise HTTPException(status_code=404, detail="Robust pipeline metadata not found")
            
            robust_result = session_data["robust_pipeline_result"]
            metadata_json = json.dumps(robust_result, indent=2, default=str)
            
            return FileResponse(
                path=None,
                filename=f"robust_pipeline_metadata_{session_id}.json",
                content=metadata_json.encode(),
                media_type="application/json"
            )
        
        else:
            raise HTTPException(status_code=400, detail="Invalid artifact type")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)