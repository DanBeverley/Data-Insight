"""
Hierarchical dataset upload router

Handles folder-based dataset uploads with:
- Folder scanning and indexing
- Smart selection UI
- Progressive loading with progress tracking
- Multiple loading strategies
"""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import asyncio

from ..data_ingestion.hierarchical_scanner import HierarchicalDatasetScanner, DatasetIndex
from ..data_ingestion.hierarchical_loader import HierarchicalDatasetLoader, LoadingProgress
from ..api_utils.upload_handler import enhance_with_agent_profile, load_data_to_agent_sandbox
from ..reporting.report_repository import ReportRepository
from ..database.connection import get_database_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload/hierarchical", tags=["hierarchical_upload"])


class ScanFolderRequest(BaseModel):
    """Request to scan a hierarchical folder"""

    folder_path: str = Field(..., description="Absolute path to folder")
    max_depth: int = Field(3, description="Maximum folder depth to scan")
    include_schema: bool = Field(True, description="Include schema samples")


class LoadDatasetRequest(BaseModel):
    """Request to load selected dataset"""

    session_id: str
    root_path: str
    selection: Dict[str, Any] = Field(..., description="Selection criteria with folders, filters, strategy")


# Store active scan results
_scan_cache: Dict[str, DatasetIndex] = {}


@router.post("/scan")
async def scan_hierarchical_folder(request: ScanFolderRequest) -> JSONResponse:
    """
    Scan a hierarchical folder and return index

    Returns complete folder structure, file counts, schemas, and
    smart selection suggestions.
    """
    try:
        folder_path = Path(request.folder_path)

        if not folder_path.exists():
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")

        if not folder_path.is_dir():
            raise HTTPException(status_code=400, detail="Path must be a directory")

        logger.info(f"Scanning hierarchical dataset at: {folder_path}")

        scanner = HierarchicalDatasetScanner()
        index = scanner.scan_folder(folder_path, max_depth=request.max_depth, include_schema=request.include_schema)

        suggestions = scanner.get_selection_suggestions(index, max_files=1000, max_size_mb=500)

        scan_id = str(folder_path)
        _scan_cache[scan_id] = index

        return JSONResponse(
            {"success": True, "scan_id": scan_id, "index": scanner.to_dict(index), "suggestions": suggestions}
        )

    except Exception as e:
        logger.error(f"Scan failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_hierarchical_dataset(
    request: LoadDatasetRequest, background_tasks: BackgroundTasks
) -> StreamingResponse:
    """
    Load selected dataset with progress streaming

    Streams progress updates via Server-Sent Events, then
    loads data into session and E2B sandbox.
    """

    async def generate_progress():
        try:
            root_path = Path(request.root_path)
            loader = HierarchicalDatasetLoader(max_workers=4)

            async for progress in _async_load_wrapper(loader, root_path, request.selection):
                if isinstance(progress, LoadingProgress):
                    progress_data = {
                        "type": "progress",
                        "status": progress.status,
                        "current_file": progress.current_file,
                        "loaded": progress.loaded_files,
                        "total": progress.total_files,
                        "percent": progress.progress_percent,
                        "size_mb": progress.current_size_mb,
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"

                    await asyncio.sleep(0.01)

                else:
                    result = progress

                    if not result.success:
                        error_data = {"type": "error", "message": result.error}
                        yield f"data: {json.dumps(error_data)}\n\n"
                        return
                    logger.info(f"Creating automatic report for session {request.session_id}")
                    try:
                        db_manager = get_database_manager()
                        db = db_manager.get_session()
                        repo = ReportRepository(db)

                        report = repo.create_report(
                            session_id=request.session_id, dataset_name=dataset_name, status="generating"
                        )

                        logger.info(f"Report {report.id} created for dataset {dataset_name}")

                        report_created_data = {
                            "type": "report_created",
                            "report_id": report.id,
                            "dataset_name": dataset_name,
                        }
                        yield f"data: {json.dumps(report_created_data)}\n\n"

                        db.close()

                        # [INTELLIGENCE UPGRADE] Run profiling immediately
                        logger.info(f"Starting automatic profiling for session {request.session_id}")
                        try:
                            # Run in executor to avoid blocking the stream
                            loop = asyncio.get_event_loop()
                            # We need to get the dataframe from session or reload it.
                            # Since loader doesn't return DF directly here, we rely on session_store or reload.
                            # But wait, the loader *should* have loaded it into session?
                            # Let's check HierarchicalDatasetLoader.

                            # Assuming loader saves to session/disk.
                            # We can use enhance_with_agent_profile if we have the DF.
                            # For now, let's trigger the background profiling task if possible.

                            from ..api import run_profiling_background

                            # We need the DF. Let's get it from session.
                            from ..api_utils.session_management import session_data_manager

                            session_data = session_data_manager.get_session(request.session_id)
                            if session_data and "dataframe" in session_data:
                                df = session_data["dataframe"]
                                # Fire and forget profiling
                                asyncio.create_task(run_profiling_background(df, request.session_id, dataset_name))
                                yield f"data: {json.dumps({'type': 'progress', 'message': 'Profiling started...'})}\n\n"
                        except Exception as prof_e:
                            logger.error(f"Failed to trigger profiling: {prof_e}")

                    except Exception as report_error:
                        logger.error(f"Failed to create report: {report_error}")

                    complete_data = {
                        "type": "complete",
                        "shape": result.shape,
                        "size_mb": result.size_mb,
                        "files_loaded": result.files_loaded,
                        "files_failed": result.files_failed,
                        "loading_time": result.loading_time_seconds,
                        "metadata": result.metadata,
                    }
                    yield f"data: {json.dumps(complete_data)}\n\n"

        except Exception as e:
            logger.error(f"Loading failed: {e}", exc_info=True)
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"},
    )


async def _async_load_wrapper(loader, root_path, selection):
    """Wrap synchronous generator in async context"""
    loop = asyncio.get_event_loop()

    def sync_generator():
        for item in loader.load_dataset(root_path, selection):
            yield item

    gen = sync_generator()

    while True:
        try:
            item = await loop.run_in_executor(None, next, gen)
            yield item
        except StopIteration as e:
            if e.value:
                yield e.value
            break


@router.get("/suggestions")
async def get_loading_suggestions(root_path: str, max_files: int = 1000, max_size_mb: int = 500) -> JSONResponse:
    """Get smart loading suggestions for a scanned folder"""
    try:
        scan_id = root_path

        if scan_id not in _scan_cache:
            raise HTTPException(status_code=404, detail="Folder not scanned. Call /scan first.")

        index = _scan_cache[scan_id]
        scanner = HierarchicalDatasetScanner()

        suggestions = scanner.get_selection_suggestions(index, max_files=max_files, max_size_mb=max_size_mb)

        return JSONResponse({"success": True, "suggestions": suggestions})

    except Exception as e:
        logger.error(f"Suggestions failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/{scan_id}")
async def clear_scan_cache(scan_id: str) -> JSONResponse:
    """Clear cached scan results"""
    if scan_id in _scan_cache:
        del _scan_cache[scan_id]
        return JSONResponse({"success": True, "message": "Cache cleared"})

    raise HTTPException(status_code=404, detail="Scan not found in cache")
