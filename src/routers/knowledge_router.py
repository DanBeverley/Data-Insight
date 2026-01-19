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
    from data_scientist_chatbot.app.utils.knowledge_store import get_knowledge_store

    try:
        store = get_knowledge_store(session_id)
        items = store.list_items()
        return {"session_id": session_id, "count": len(items), "items": items}
    except Exception as e:
        logger.error(f"[KNOWLEDGE_API] List failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}")
async def upload_to_knowledge(session_id: str, file: UploadFile = File(...)):
    """Upload a file directly to knowledge store."""
    from data_scientist_chatbot.app.utils.knowledge_store import get_knowledge_store
    from data_scientist_chatbot.app.utils.dataset_registry import DatasetRegistry
    import pandas as pd

    DATA_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}

    try:
        temp_dir = Path("data/temp_uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / file.filename

        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        suffix = Path(file.filename).suffix.lower()
        is_data_file = suffix in DATA_EXTENSIONS

        doc_id = None
        registry_registered = False

        # For data files: register in DatasetRegistry + index summary to RAG
        if is_data_file:
            registry = DatasetRegistry(session_id)

            # Load to get row/column counts
            try:
                if suffix == ".csv":
                    df = pd.read_csv(temp_path)
                elif suffix == ".tsv":
                    df = pd.read_csv(temp_path, sep="\t")
                elif suffix in [".xlsx", ".xls"]:
                    df = pd.read_excel(temp_path)
                elif suffix == ".json":
                    df = pd.read_json(temp_path)
                elif suffix == ".parquet":
                    df = pd.read_parquet(temp_path)
                else:
                    df = pd.DataFrame()

                rows, cols = df.shape
            except Exception:
                rows, cols = 0, 0

            registry.register(file.filename, str(temp_path), rows, cols)
            registry_registered = True

            # Index basic info to RAG for semantic search
            store = get_knowledge_store(session_id)
            summary = f"Dataset: {file.filename}\nRows: {rows}, Columns: {cols}"
            if cols > 0:
                summary += f"\nColumn Names: {', '.join(df.columns.tolist()[:20])}"
            doc_id = store.add_document(summary, "dataset", file.filename)
            registry.mark_rag_indexed(file.filename)
        else:
            # Non-data files: just index to RAG
            store = get_knowledge_store(session_id)
            doc_id = store.add_file(str(temp_path), source_type="user_upload")

        temp_path.unlink(missing_ok=True)

        if not doc_id and not registry_registered:
            raise HTTPException(status_code=400, detail="Failed to ingest file")

        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": file.filename,
            "registered_as_dataset": registry_registered,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[KNOWLEDGE_API] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}/{doc_id}")
async def delete_knowledge_item(session_id: str, doc_id: str):
    """Delete an item from knowledge store."""
    from data_scientist_chatbot.app.utils.knowledge_store import get_knowledge_store

    try:
        store = get_knowledge_store(session_id)
        success = store.delete_item(doc_id)

        if not success:
            raise HTTPException(status_code=404, detail="Item not found or delete failed")

        return {"status": "deleted", "doc_id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[KNOWLEDGE_API] Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/doc/{doc_id}")
async def get_knowledge_document(session_id: str, doc_id: str):
    """Get full content of a specific knowledge document."""
    from data_scientist_chatbot.app.utils.knowledge_store import get_knowledge_store

    try:
        store = get_knowledge_store(session_id)
        doc = store.get_document(doc_id)

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "id": doc["id"],
            "content": doc["content"],
            "source_name": doc["metadata"].get("source_name", "Unknown"),
            "source": doc["metadata"].get("source", "unknown"),
            "added_at": doc["metadata"].get("added_at", ""),
            "inject_to_context": doc["metadata"].get("inject_to_context", False),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[KNOWLEDGE_API] Get document failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/query")
async def query_knowledge(session_id: str, q: str, k: int = 5):
    """Query knowledge store for relevant items."""
    from data_scientist_chatbot.app.utils.knowledge_store import get_knowledge_store

    try:
        store = get_knowledge_store(session_id)
        results = store.query(q, k=k)
        return {"session_id": session_id, "query": q, "count": len(results), "results": results}
    except Exception as e:
        logger.error(f"[KNOWLEDGE_API] Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{session_id}/{doc_id}")
async def update_knowledge_item(session_id: str, doc_id: str, inject_to_context: bool):
    """Update item metadata (toggle context injection)."""
    from data_scientist_chatbot.app.utils.knowledge_store import get_knowledge_store

    try:
        store = get_knowledge_store(session_id)
        success = store.update_item_metadata(doc_id, {"inject_to_context": inject_to_context})

        if not success:
            raise HTTPException(status_code=404, detail="Item not found")

        return {"status": "updated", "doc_id": doc_id, "inject_to_context": inject_to_context}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[KNOWLEDGE_API] Update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
