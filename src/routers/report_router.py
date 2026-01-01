"""Report generation and management endpoints"""

import uuid
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session


from src.reporting.report_repository import ReportRepository
from src.database.connection import get_database_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["reports"])


class CreateReportRequest(BaseModel):
    session_id: str
    dataset_path: str
    dataset_name: str
    user_request: Optional[str] = None


class ModelTrainingRequest(BaseModel):
    target_column: str


def get_db():
    """Dependency to get database session"""
    db_manager = get_database_manager()
    with db_manager.get_session() as db:
        yield db


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
    Uses UnifiedReportGenerator for multimodal analysis (Vision + Data).
    """
    repo = ReportRepository(db)
    report = repo.get_report(report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    async def generate():
        """Generate and stream report sections using UnifiedReportGenerator"""
        import json
        from src.reporting.unified_report_generator import UnifiedReportGenerator

        generator = UnifiedReportGenerator()

        try:
            # Get dataset path from session store
            dataset_path = None
            image_paths = []
            artifacts = []

            try:
                import builtins
                from src.api_utils.session_management import session_data_manager

                session_data = session_data_manager.get_session(report.session_id)
                if session_data:
                    dataset_path = session_data.get("dataset_path")
                    # Assuming images might be stored in session or passed differently
                    # For now, we'll check for an 'images' key in session data
                    image_paths = session_data.get("uploaded_images", [])
                    artifacts = session_data.get("artifacts", [])
            except Exception as e:
                logger.warning(f"[REPORT] Could not retrieve session data: {e}")
                dataset_path = f"/tmp/datasets/{report.session_id}/data.csv"  # Fallback

            logger.info(
                f"[REPORT] Generating Unified Report with {len(artifacts)} artifacts and {len(image_paths)} images"
            )

            # Stream comprehensive report sections
            async for section in generator.generate(
                session_id=report.session_id,
                report_type="unified_multimodal",
                dataset_path=dataset_path,
                artifacts=artifacts,
                image_paths=image_paths,
                analysis_focus=None,
                user_request=session_data.get("user_request") if session_data else None,
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


@router.get("/{session_id}/export/{format}")
async def export_report_by_session(session_id: str, format: str, db: Session = Depends(get_db)):
    from fastapi.responses import Response
    from pathlib import Path
    import glob
    from src.reporting.report_exporter import get_report_exporter

    data_reports_dir = Path("data/reports")
    html_content = None
    dataset_name = "Analysis"

    if data_reports_dir.exists():
        session_prefix = session_id[:8]
        html_files = sorted(
            [f for f in data_reports_dir.glob("*.html") if session_prefix in f.name],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if html_files:
            with open(html_files[0], "r", encoding="utf-8") as f:
                html_content = f.read()
            dataset_name = html_files[0].stem.replace("Analysis_Report_", "").replace("_", " ")

    if not html_content:
        repo = ReportRepository(db)
        reports = repo.get_reports_by_session(session_id)
        if reports:
            report = reports[0]
            report_data = report.report_data or {}
            sections = []
            section_order = ["executive_dashboard", "visual_analysis", "key_drivers", "model_arena", "artifact_gallery"]
            section_titles = {
                "executive_dashboard": "Executive Summary",
                "visual_analysis": "Visual Analysis",
                "key_drivers": "Key Drivers",
                "model_arena": "Model Performance",
                "artifact_gallery": "Generated Artifacts",
            }
            for section_key in section_order:
                if section_key in report_data:
                    sections.append(
                        {
                            "title": section_titles.get(section_key, section_key.replace("_", " ").title()),
                            "content": report_data[section_key],
                        }
                    )
            exporter = get_report_exporter()
            html_content = exporter.export_html(
                sections, title=f"Analysis Report - {report.dataset_name}", session_id=session_id
            )
            dataset_name = report.dataset_name

    if not html_content:
        raise HTTPException(status_code=404, detail="No reports found for this session")

    exporter = get_report_exporter()
    title = f"Analysis Report - {dataset_name}"

    if format == "html":
        from src.reporting.report_exporter import make_standalone_html

        standalone_content = make_standalone_html(html_content)
        return Response(
            content=standalone_content.encode("utf-8"),
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=report_{session_id[:8]}.html"},
        )

    sections = [{"title": "Report", "content": html_content}]

    format_handlers = {
        "pdf": (exporter.export_pdf, "application/pdf", "pdf"),
        "docx": (
            exporter.export_docx,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
        ),
        "txt": (exporter.export_txt, "text/plain", "txt"),
        "zip": (exporter.export_zip, "application/zip", "zip"),
    }

    if format not in format_handlers:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format: {format}. Supported: pdf, docx, html, txt, zip"
        )

    handler, media_type, ext = format_handlers[format]

    try:
        content = handler(sections, title=title, session_id=session_id)
        filename = f"report_{session_id[:8]}.{ext}"
        return Response(
            content=content, media_type=media_type, headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


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


@router.get("/{report_id}/export/{format}")
async def export_report(report_id: str, format: str, db: Session = Depends(get_db)):
    from fastapi.responses import Response
    from src.reporting.report_exporter import get_report_exporter

    repo = ReportRepository(db)
    report = repo.get_report(report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    report_data = report.report_data or {}
    sections = []

    section_order = ["executive_dashboard", "visual_analysis", "key_drivers", "model_arena", "artifact_gallery"]
    section_titles = {
        "executive_dashboard": "Executive Summary",
        "visual_analysis": "Visual Analysis",
        "key_drivers": "Key Drivers",
        "model_arena": "Model Performance",
        "artifact_gallery": "Generated Artifacts",
    }

    for section_key in section_order:
        if section_key in report_data:
            sections.append(
                {
                    "title": section_titles.get(section_key, section_key.replace("_", " ").title()),
                    "content": report_data[section_key],
                }
            )

    exporter = get_report_exporter()
    title = f"Analysis Report - {report.dataset_name}"

    format_handlers = {
        "pdf": (exporter.export_pdf, "application/pdf", "pdf"),
        "docx": (
            exporter.export_docx,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "docx",
        ),
        "xlsx": (exporter.export_xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
        "pptx": (
            exporter.export_pptx,
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "pptx",
        ),
        "html": (exporter.export_html, "text/html", "html"),
        "txt": (exporter.export_txt, "text/plain", "txt"),
        "zip": (exporter.export_zip, "application/zip", "zip"),
    }

    if format not in format_handlers:
        raise HTTPException(
            status_code=400, detail=f"Unsupported format: {format}. Supported: pdf, docx, xlsx, pptx, html, txt, zip"
        )

    handler, media_type, ext = format_handlers[format]

    try:
        content = handler(sections, title=title, session_id=report.session_id)
        filename = f"report_{report_id[:8]}.{ext}"
        return Response(
            content=content, media_type=media_type, headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/{report_id}/export-formats")
async def get_export_formats(report_id: str, db: Session = Depends(get_db)):
    repo = ReportRepository(db)
    report = repo.get_report(report_id)

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "report_id": report_id,
        "formats": [
            {
                "id": "html",
                "label": "HTML (Interactive)",
                "icon": "globe",
                "description": "For web viewing with interactivity",
            },
            {"id": "pdf", "label": "PDF Document", "icon": "file-text", "description": "For sharing and printing"},
            {
                "id": "docx",
                "label": "Word Document",
                "icon": "file-edit",
                "description": "For editing in Microsoft Word",
            },
            {"id": "xlsx", "label": "Excel Spreadsheet", "icon": "table", "description": "For data analysis"},
            {"id": "pptx", "label": "PowerPoint", "icon": "presentation", "description": "For presentations"},
            {"id": "txt", "label": "Plain Text", "icon": "file", "description": "For simple text viewing"},
            {"id": "zip", "label": "ZIP Bundle", "icon": "archive", "description": "HTML + artifacts bundled"},
        ],
    }
