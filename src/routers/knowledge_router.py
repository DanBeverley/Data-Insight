"""Knowledge Store API Router - Endpoints for managing per-session knowledge."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
import logging
import shutil
from pathlib import Path

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)


@router.get("/{session_id}")
async def list_knowledge_items(session_id: str):
    """List all items in a session's knowledge store."""
    from data_scientist_chatbot.app.utils.knowledge_store import KnowledgeStore

    try:
        store = KnowledgeStore(session_id)
        items = store.list_items()
        return {"session_id": session_id, "count": len(items), "items": items}
    except Exception as e:
        logger.error(f"[KNOWLEDGE_API] List failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}")
async def upload_to_knowledge(session_id: str, file: UploadFile = File(...)):
    """Upload a file directly to knowledge store."""
    from data_scientist_chatbot.app.utils.knowledge_store import KnowledgeStore

    try:
        temp_dir = Path("data/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / file.filename

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        store = KnowledgeStore(session_id)
        doc_id = store.add_file(str(temp_path), source_type="user_upload")

        temp_path.unlink(missing_ok=True)

        if not doc_id:
            raise HTTPException(status_code=400, detail="Failed to ingest file")

        return {"status": "success", "doc_id": doc_id, "filename": file.filename}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[KNOWLEDGE_API] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}/{doc_id}")
async def delete_knowledge_item(session_id: str, doc_id: str):
    """Delete an item from knowledge store."""
    from data_scientist_chatbot.app.utils.knowledge_store import KnowledgeStore

    try:
        store = KnowledgeStore(session_id)
        success = store.delete_item(doc_id)

        if not success:
            raise HTTPException(status_code=404, detail="Item not found or delete failed")

        return {"status": "deleted", "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[KNOWLEDGE_API] Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/query")
async def query_knowledge(session_id: str, q: str, k: int = 5):
    """Query knowledge store for relevant items."""
    from data_scientist_chatbot.app.utils.knowledge_store import KnowledgeStore

    try:
        store = KnowledgeStore(session_id)
        results = store.query(q, k=k)
        return {"session_id": session_id, "query": q, "count": len(results), "results": results}
    except Exception as e:
        logger.error(f"[KNOWLEDGE_API] Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
