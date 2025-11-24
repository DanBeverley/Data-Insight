"""Report generation and management endpoints"""

import uuid
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.reporting.report_generator import ReportGenerator
from src.reporting.report_repository import ReportRepository
from src.database.connection import get_database_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["reports"])


class CreateReportRequest(BaseModel):
    session_id: str
    dataset_path: str
    dataset_name: str


class ModelTrainingRequest(BaseModel):
    target_column: str


def get_db():
    """Dependency to get database session"""
    db_manager = get_database_manager()
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()


@router.post("/create")
async def create_report(request: CreateReportRequest, db: Session = Depends(get_db)):
    """
    Create a new report entry in the database.

    Returns:
        Report ID and initial status
    """
    try:
        repo = ReportRepository(db)
        report = repo.create_report(
            session_id=request.session_id, dataset_name=request.dataset_name, status="generating"
        )

        logger.info(f"[REPORT] Created report {report.id} for session {request.session_id}")

        return {"report_id": report.id, "status": report.status, "dataset_name": report.dataset_name}

    except Exception as e:
        logger.error(f"[REPORT] Failed to create report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}/stream")
async def stream_report_generation(report_id: str, db: Session = Depends(get_db)):
    """
    Stream report sections as they are generated.

    Server-Sent Events endpoint that yields report sections progressively.
    """
    repo = ReportRepository(db)
    report = repo.get_report(report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    async def generate():
        """Generate and stream report sections"""
        import json
        from src.reporting.report_generator import ReportGenerator

        generator = ReportGenerator()

        try:
            dataset_path = f"/tmp/datasets/{report.session_id}/data.csv"

            async for section in generator.generate(
                session_id=report.session_id, dataset_path=dataset_path, dataset_name=report.dataset_name
            ):
                repo.update_report_data(report_id=report_id, section=section["section"], content=section["html"])

                yield f"data: {json.dumps(section)}\n\n"

            repo.update_report_status(report_id, "completed")
            yield f"data: {json.dumps({'section': 'complete', 'status': 'completed'})}\n\n"

        except Exception as e:
            logger.error(f"[REPORT] Stream error for {report_id}: {e}")
            repo.update_report_status(report_id, "failed")
            yield f"data: {json.dumps({'section': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/{report_id}")
async def get_report(report_id: str, db: Session = Depends(get_db)):
    """Get report by ID"""
    repo = ReportRepository(db)
    report = repo.get_report(report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "id": report.id,
        "session_id": report.session_id,
        "dataset_name": report.dataset_name,
        "status": report.status,
        "report_data": report.report_data,
        "created_at": report.created_at.isoformat(),
        "updated_at": report.updated_at.isoformat(),
    }


@router.get("/session/{session_id}")
async def get_session_reports(session_id: str, db: Session = Depends(get_db)):
    """Get all reports for a session"""
    repo = ReportRepository(db)
    reports = repo.get_reports_by_session(session_id)

    return [
        {"id": r.id, "dataset_name": r.dataset_name, "status": r.status, "created_at": r.created_at.isoformat()}
        for r in reports
    ]


@router.post("/{report_id}/train-models")
async def trigger_model_training(
    report_id: str, request: ModelTrainingRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """
    Trigger model training for the report.

    This is called when the user clicks "Run Model Comparison" button.
    Training runs in the background to avoid blocking.
    """
    repo = ReportRepository(db)
    report = repo.get_report(report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    async def train_models_task():
        """Background task to train models"""
        from src.reporting.section_builders.model_arena import train_baseline_models

        try:
            dataset_path = f"/tmp/datasets/{report.session_id}/data.csv"

            models = await train_baseline_models(
                dataset_path=dataset_path, target_col=request.target_column, session_id=report.session_id
            )

            repo.update_report_data(report_id=report_id, section="model_results", content=models)

            logger.info(f"[REPORT] Model training completed for {report_id}")

        except Exception as e:
            logger.error(f"[REPORT] Model training failed for {report_id}: {e}")

    background_tasks.add_task(train_models_task)

    return {
        "status": "training_started",
        "estimated_time_seconds": 180,
        "message": "Model training started in background",
    }


@router.get("/{report_id}/models/status")
async def get_model_training_status(report_id: str, db: Session = Depends(get_db)):
    """Poll for model training completion"""
    repo = ReportRepository(db)
    report = repo.get_report(report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    has_models = "model_results" in (report.report_data or {})

    return {
        "status": "completed" if has_models else "training",
        "models_ready": has_models,
        "report_status": report.status,
    }


@router.delete("/{report_id}")
async def delete_report(report_id: str, db: Session = Depends(get_db)):
    """Delete a report and all its artifacts"""
    repo = ReportRepository(db)
    success = repo.delete_report(report_id)

    if not success:
        raise HTTPException(status_code=404, detail="Report not found")

    return {"message": "Report deleted successfully"}
