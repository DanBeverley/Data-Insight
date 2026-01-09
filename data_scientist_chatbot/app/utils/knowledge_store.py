"""Knowledge Store - Persistent RAG storage for user documents and research."""

import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = Path("data/knowledge")


class KnowledgeStore:
    """Per-session persistent vector store for documents and research."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.storage_path = KNOWLEDGE_BASE_DIR / session_id
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.storage_path))
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        try:
            self.collection = self.client.get_or_create_collection(name="knowledge", embedding_function=self.ef)
            logger.info(f"[KNOWLEDGE] Initialized store for session {session_id}")
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Failed to create collection: {e}")
            self.collection = None

    def _generate_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def add_document(
        self, content: str, source: str = "user", source_name: str = "", metadata: Optional[Dict] = None
    ) -> str:
        """Add a text document to knowledge store."""
        if not self.collection:
            return ""

        doc_id = f"doc_{self._generate_id(content)}"
        doc_metadata = {
            "source": source,
            "source_name": source_name or "Unknown",
            "added_at": datetime.now().isoformat(),
            "content_preview": content[:200],
            **(metadata or {}),
        }

        try:
            self.collection.add(documents=[content], metadatas=[doc_metadata], ids=[doc_id])
            logger.info(f"[KNOWLEDGE] Added document {doc_id} from {source}")
            return doc_id
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Failed to add document: {e}")
            return ""

    def add_file(self, file_path: str, source_type: str = "user_upload") -> str:
        """Add a file to knowledge store by extracting its text content."""
        path = Path(file_path)
        if not path.exists():
            logger.error(f"[KNOWLEDGE] File not found: {file_path}")
            return ""

        try:
            content = self._extract_file_content(path)
            if not content:
                return ""

            return self.add_document(
                content=content,
                source=source_type,
                source_name=path.name,
                metadata={"file_path": str(file_path), "file_type": path.suffix},
            )
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Failed to process file {file_path}: {e}")
            return ""

    def _extract_file_content(self, path: Path) -> str:
        """Extract text content from various file types."""
        suffix = path.suffix.lower()

        if suffix in [".txt", ".md", ".py", ".json", ".csv"]:
            return path.read_text(encoding="utf-8", errors="ignore")

        if suffix == ".pdf":
            try:
                import fitz

                doc = fitz.open(path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except ImportError:
                logger.warning("[KNOWLEDGE] PyMuPDF not installed for PDF extraction")
                return ""

        logger.warning(f"[KNOWLEDGE] Unsupported file type: {suffix}")
        return ""

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Query knowledge store for relevant documents."""
        if not self.collection:
            return []

        try:
            results = self.collection.query(query_texts=[text], n_results=k)

            items = []
            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    items.append(
                        {
                            "content": doc,
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "id": results["ids"][0][i] if results["ids"] else "",
                        }
                    )
            return items

        except Exception as e:
            logger.error(f"[KNOWLEDGE] Query failed: {e}")
            return []

    def list_items(self) -> List[Dict[str, Any]]:
        """List all items in knowledge store."""
        if not self.collection:
            return []

        try:
            results = self.collection.get(include=["metadatas"])
            items = []
            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                items.append(
                    {
                        "id": doc_id,
                        "source": meta.get("source", "unknown"),
                        "source_name": meta.get("source_name", "Unknown"),
                        "added_at": meta.get("added_at", ""),
                        "content_preview": meta.get("content_preview", ""),
                    }
                )
            return items
        except Exception as e:
            logger.error(f"[KNOWLEDGE] List failed: {e}")
            return []

    def delete_item(self, doc_id: str) -> bool:
        """Delete an item from knowledge store."""
        if not self.collection:
            return False

        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"[KNOWLEDGE] Deleted {doc_id}")
            return True
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Delete failed: {e}")
            return False

    def count(self) -> int:
        """Get total items in store."""
        if not self.collection:
            return 0
        return self.collection.count()
