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

class DataIngestionRequest(BaseModel):
    url: str
    data_type: str = "csv"

class ProcessingResult(BaseModel):
    status: str
    message: str
    data_shape: Optional[tuple] = None
    column_roles: Optional[Dict[str, List[str]]] = None
    processing_time: Optional[float] = None
    artifacts: Optional[Dict[str, str]] = None

# In-memory storage (in production, use Redis/database)
session_store = {}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application interface."""
    return HTMLResponse(open("static/index.html").read())

@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """Upload and validate dataset."""
    try:
        # Read uploaded file
        contents = await file.read()
        df = ingest_data(io.BytesIO(contents), filename=file.filename)
        
        if df is None:
            raise HTTPException(status_code=400, detail="Failed to parse uploaded file")
        
        # Validate data quality
        validator = DataQualityValidator(df)
        validation_report = validator.validate()
        
        # Generate session ID and store data
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_store[session_id] = {
            "dataframe": df,
            "validation_report": validation_report,
            "filename": file.filename
        }
        
        return {
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.post("/api/ingest-url")
async def ingest_from_url_endpoint(request: DataIngestionRequest):
    """Ingest data from URL."""
    try:
        df = ingest_from_url(request.url, request.data_type)
        
        if df is None:
            raise HTTPException(status_code=400, detail="Failed to fetch or parse data from URL")
        
        # Validate data quality
        validator = DataQualityValidator(df)
        validation_report = validator.validate()
        
        # Generate session ID and store data
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_store[session_id] = {
            "dataframe": df,
            "validation_report": validation_report,
            "source_url": request.url
        }
        
        return {
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
    """Process data with automated pipeline."""
    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        start_time = datetime.now()
        df = session_store[session_id]["dataframe"]
        
        # Create and run orchestrator
        orchestrator = WorkflowOrchestrator(
            df=df,
            target_column=config.target_column,
            task=config.task
        )
        
        result = orchestrator.run()
        
        if result.status != Status.SUCCESS:
            raise HTTPException(status_code=500, detail=result.error_message)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        session_store[session_id].update({
            "orchestrator_result": result,
            "processed_data": result.processed_features,
            "aligned_target": result.aligned_target,
            "processing_config": config.dict()
        })
        
        return ProcessingResult(
            status="success",
            message="Data processed successfully",
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
    """Download processing artifacts."""
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