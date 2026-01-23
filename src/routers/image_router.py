"""Image Analysis Router - Upload and analyze images as datasets"""

import os
import uuid
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from src.database.connection import get_database_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/image", tags=["image-analysis"])


def get_db():
    db_manager = get_database_manager()
    with db_manager.get_session() as db:
        yield db


ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/upload")
async def upload_image_for_analysis(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    analysis_type: str = Form("auto"),  # auto, chart, table, document
):
    """
    Upload an image for data extraction and analysis.

    Args:
        file: Image file to analyze
        session_id: Current session ID
        analysis_type: Type of analysis (auto-detect by default)

    Returns:
        Image ID and initial processing status
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}")

    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large. Max 10MB.")

    image_id = str(uuid.uuid4())[:8]

    upload_dir = Path(f"data/uploads/{session_id}/images")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_ext = Path(file.filename).suffix or ".png"
    image_path = upload_dir / f"{image_id}{file_ext}"

    with open(image_path, "wb") as f:
        f.write(content)

    logger.info(f"[IMAGE] Uploaded image {image_id} for session {session_id}")

    return {
        "image_id": image_id,
        "filename": file.filename,
        "path": str(image_path),
        "analysis_type": analysis_type,
        "status": "uploaded",
        "message": "Image uploaded. Call /analyze to extract data.",
    }


@router.post("/{session_id}/analyze/{image_id}")
async def analyze_image(session_id: str, image_id: str, analysis_type: str = "auto"):
    """
    Trigger vision model analysis to extract data from image.

    Returns extracted data preview for user confirmation.
    """
    from src.services.image_analyzer import ImageAnalyzer

    upload_dir = Path(f"data/uploads/{session_id}/images")
    image_files = list(upload_dir.glob(f"{image_id}.*"))

    if not image_files:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = image_files[0]

    try:
        analyzer = ImageAnalyzer()
        result = await analyzer.extract_data(
            image_path=str(image_path), analysis_type=analysis_type, session_id=session_id
        )

        logger.info(f"[IMAGE] Extracted data from {image_id}: {result.get('row_count', 0)} rows")

        return {
            "image_id": image_id,
            "status": "extracted",
            "image_type": result.get("image_type"),
            "title": result.get("title"),
            "columns": result.get("columns", []),
            "preview_rows": result.get("preview_rows", []),
            "row_count": result.get("row_count", 0),
            "confidence": result.get("confidence", 0.0),
            "notes": result.get("notes"),
        }
    except Exception as e:
        logger.error(f"[IMAGE] Analysis failed for {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/{session_id}/confirm/{image_id}")
async def confirm_extracted_data(session_id: str, image_id: str, adjustments: Optional[dict] = None):
    """
    Confirm extracted data and create DataFrame for analysis.

    User can optionally provide adjustments to column names or data.
    """
    from src.services.image_analyzer import ImageAnalyzer
    from src.api_utils.session_management import session_data_manager

    upload_dir = Path(f"data/uploads/{session_id}/images")
    image_files = list(upload_dir.glob(f"{image_id}.*"))

    if not image_files:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        analyzer = ImageAnalyzer()
        df_path = await analyzer.create_dataframe(image_id=image_id, session_id=session_id, adjustments=adjustments)

        session_data_manager.update_session(
            session_id, {"dataset_path": df_path, "dataset_source": "image_extraction", "source_image_id": image_id}
        )

        logger.info(f"[IMAGE] Created DataFrame from {image_id} at {df_path}")

        return {"status": "confirmed", "dataset_path": df_path, "message": "Data extracted and ready for analysis"}
    except Exception as e:
        logger.error(f"[IMAGE] Confirmation failed for {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Confirmation failed: {str(e)}")


@router.get("/{session_id}/images")
async def list_session_images(session_id: str):
    """List all uploaded images for a session"""
    upload_dir = Path(f"data/uploads/{session_id}/images")

    if not upload_dir.exists():
        return {"images": []}

    images = []
    for img_path in upload_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            images.append(
                {
                    "image_id": img_path.stem,
                    "filename": img_path.name,
                    "path": str(img_path),
                    "size": img_path.stat().st_size,
                    "created_at": datetime.fromtimestamp(img_path.stat().st_ctime).isoformat(),
                }
            )

    return {"images": sorted(images, key=lambda x: x["created_at"], reverse=True)}
