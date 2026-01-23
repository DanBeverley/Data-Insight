"""Model loader for retrieving models from blob storage"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Handles loading models from blob storage with local caching
    """

    def __init__(self, cloud_storage, registry_service, cache_dir: Optional[Path] = None):
        """
        Args:
            cloud_storage: Cloud storage service instance (R2StorageService)
            registry_service: ModelRegistryService instance
            cache_dir: Local cache directory (defaults to data/model_cache)
        """
        self.blob_service = cloud_storage
        self.registry = registry_service

        if cache_dir is None:
            from src.config import settings

            storage_config = settings.get("object_storage", {})
            cache_path = storage_config.get("cache_dir", "data/model_cache")
            self.cache_dir = Path(cache_path)
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_model_for_session(
        self, session_id: str, model_type: Optional[str] = None, model_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load model for a session, downloading from blob storage if needed

        Args:
            session_id: Session ID
            model_type: Optional model type filter
            model_id: Optional specific model ID

        Returns:
            Dict with local_path, model_id, metadata or None if not found
        """
        try:
            # Find model in registry
            if model_id:
                model_info = self.registry.get_model_download_info(model_id)
            else:
                model = self.registry.find_model(session_id=session_id, model_type=model_type)
                if not model:
                    logger.warning(f"No model found for session {session_id}, type {model_type}")
                    return None

                model_info = self.registry.get_model_download_info(model["model_id"])

            # Check local cache
            cached_path = self._get_cache_path(model_info["model_id"])

            if cached_path.exists():
                # Verify checksum
                actual_checksum = self.blob_service._compute_sha256(cached_path)
                if actual_checksum == model_info["checksum"]:
                    logger.info(f"Using cached model: {model_info['model_id']}")
                    return {
                        "local_path": str(cached_path),
                        "model_id": model_info["model_id"],
                        "model_type": model_info["model_type"],
                        "framework": model_info.get("framework"),
                        "from_cache": True,
                    }
                else:
                    logger.warning(f"Cache checksum mismatch for {model_info['model_id']}, re-downloading")
                    cached_path.unlink()

            # Download from blob storage
            logger.info(f"Downloading model from blob storage: {model_info['blob_path']}")
            download_result = self.blob_service.download_file(
                blob_path=model_info["blob_path"], local_path=cached_path, verify_checksum=True
            )

            logger.info(f"Model downloaded and verified: {model_info['model_id']}")

            return {
                "local_path": str(cached_path),
                "model_id": model_info["model_id"],
                "model_type": model_info["model_type"],
                "framework": model_info.get("framework"),
                "from_cache": False,
                "size_bytes": download_result["size_bytes"],
            }

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def upload_model_to_sandbox(
        self,
        sandbox,
        session_id: str,
        model_type: Optional[str] = None,
        model_id: Optional[str] = None,
        sandbox_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Load model and upload to E2B sandbox for use in code execution

        Args:
            sandbox: E2B Sandbox instance
            session_id: Session ID
            model_type: Optional model type filter
            model_id: Optional specific model ID
            sandbox_path: Optional custom path in sandbox (default: /home/user/{filename})

        Returns:
            Sandbox path to uploaded model or None if failed
        """
        try:
            # Load model locally first
            model_data = self.load_model_for_session(session_id=session_id, model_type=model_type, model_id=model_id)

            if not model_data:
                logger.error("Model not found for upload to sandbox")
                return None

            local_path = Path(model_data["local_path"])
            filename = local_path.name

            # Determine sandbox path
            if sandbox_path is None:
                sandbox_path = f"/home/user/{filename}"

            # Read local file
            with open(local_path, "rb") as f:
                file_content = f.read()

            # Upload to sandbox
            sandbox.files.write(sandbox_path, file_content)

            logger.info(f"Uploaded model to sandbox: {sandbox_path} ({len(file_content)} bytes)")

            return sandbox_path

        except Exception as e:
            logger.error(f"Failed to upload model to sandbox: {e}")
            return None

    def _get_cache_path(self, model_id: str) -> Path:
        """Get local cache path for a model"""
        # Use first 8 chars of model_id as subdirectory to avoid too many files in one dir
        subdir = self.cache_dir / model_id[:8]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{model_id}.model"

    def clear_cache(self, max_size_mb: Optional[int] = None) -> int:
        """
        Clear old cached models based on size limit

        Args:
            max_size_mb: Maximum cache size in MB (from config if not provided)

        Returns:
            Number of files deleted
        """
        try:
            if max_size_mb is None:
                from src.config import settings

                storage_config = settings.get("object_storage", {})
                max_size_mb = storage_config.get("cache_max_size_mb", 500)

            max_size_bytes = max_size_mb * 1024 * 1024

            # Get all cached files with their sizes and access times
            cache_files = []
            total_size = 0

            for file_path in self.cache_dir.rglob("*.model"):
                if file_path.is_file():
                    stat = file_path.stat()
                    cache_files.append({"path": file_path, "size": stat.st_size, "atime": stat.st_atime})
                    total_size += stat.st_size

            if total_size <= max_size_bytes:
                logger.info(f"Cache size {total_size / 1024 / 1024:.2f} MB within limit")
                return 0

            # Sort by access time (oldest first)
            cache_files.sort(key=lambda x: x["atime"])

            # Delete oldest files until within limit
            deleted_count = 0
            for file_info in cache_files:
                if total_size <= max_size_bytes:
                    break

                file_info["path"].unlink()
                total_size -= file_info["size"]
                deleted_count += 1
                logger.info(f"Deleted cached model: {file_info['path'].name}")

            logger.info(f"Cache cleanup: deleted {deleted_count} files, size now {total_size / 1024 / 1024:.2f} MB")
            return deleted_count

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0
