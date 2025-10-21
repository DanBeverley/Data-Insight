"""Model registry service for tracking models in object storage"""

import uuid
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelRegistryService:
    """Manages model registry database operations"""

    def __init__(self, db_service):
        """
        Args:
            db_service: Database service instance for executing queries
        """
        self.db = db_service

    def register_model(
        self,
        session_id: str,
        dataset_hash: str,
        model_type: str,
        blob_path: str,
        blob_url: str,
        file_checksum: str,
        file_size_bytes: int,
        hyperparameters: Optional[Dict] = None,
        training_metrics: Optional[Dict] = None,
        user_id: Optional[str] = None,
        framework: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Register a trained model in the registry

        Args:
            session_id: Session ID where model was trained
            dataset_hash: Hash of dataset used for training
            model_type: Type/algorithm of model (e.g., "linear_regression", "random_forest")
            blob_path: Path to model in blob storage
            blob_url: Full URL to blob
            file_checksum: SHA256 checksum of model file
            file_size_bytes: Size of model file in bytes
            hyperparameters: Model hyperparameters used
            training_metrics: Training metrics (e.g., {"r2": 0.95, "mse": 0.02})
            user_id: Optional user ID
            framework: ML framework used (e.g., "scikit-learn", "tensorflow")
            dependencies: List of required packages

        Returns:
            model_id: Unique identifier for registered model
        """
        model_id = str(uuid.uuid4())

        # Check for existing model with same characteristics
        existing_version = self._get_latest_version(
            session_id=session_id,
            dataset_hash=dataset_hash,
            model_type=model_type
        )
        version = (existing_version or 0) + 1

        model_data = {
            "model_id": model_id,
            "session_id": session_id,
            "user_id": user_id,
            "dataset_hash": dataset_hash,
            "model_type": model_type,
            "hyperparameters": hyperparameters or {},
            "blob_path": blob_path,
            "blob_url": blob_url,
            "file_checksum": file_checksum,
            "file_size_bytes": file_size_bytes,
            "training_metrics": training_metrics or {},
            "model_version": version,
            "framework": framework,
            "dependencies": dependencies or [],
            "is_active": True,
            "created_at": datetime.utcnow(),
            "access_count": 0
        }

        self.db.insert_model_registry(model_data)
        logger.info(f"Registered model {model_id} (v{version}): {model_type} for session {session_id}")

        return model_id

    def find_model(
        self,
        session_id: str,
        dataset_hash: Optional[str] = None,
        model_type: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a model by criteria

        Args:
            session_id: Session ID
            dataset_hash: Optional dataset hash filter
            model_type: Optional model type filter
            model_id: Optional specific model ID

        Returns:
            Model record dict or None if not found
        """
        if model_id:
            return self.db.get_model_by_id(model_id)

        models = self.db.query_models(
            session_id=session_id,
            dataset_hash=dataset_hash,
            model_type=model_type,
            is_active=True
        )

        if not models:
            return None

        # Return most recent model
        return max(models, key=lambda m: m.get("created_at", ""))

    def get_model_download_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get download information for a model and update access stats

        Args:
            model_id: Model ID

        Returns:
            Dict with blob_path, checksum, size, etc.

        Raises:
            ValueError: If model not found
        """
        model = self.db.get_model_by_id(model_id)

        if not model:
            raise ValueError(f"Model not found: {model_id}")

        # Update access tracking
        self.db.increment_model_access(model_id)

        return {
            "model_id": model_id,
            "blob_path": model["blob_path"],
            "blob_url": model["blob_url"],
            "checksum": model["file_checksum"],
            "size_bytes": model["file_size_bytes"],
            "model_type": model["model_type"],
            "framework": model.get("framework"),
            "dependencies": model.get("dependencies", [])
        }

    def list_session_models(self, session_id: str) -> List[Dict[str, Any]]:
        """
        List all models for a session

        Args:
            session_id: Session ID

        Returns:
            List of model records sorted by creation time
        """
        models = self.db.query_models(session_id=session_id, is_active=True)
        return sorted(models, key=lambda m: m.get("created_at", ""), reverse=True)

    def deactivate_model(self, model_id: str) -> None:
        """Mark model as inactive (soft delete)"""
        self.db.update_model_status(model_id, is_active=False)
        logger.info(f"Deactivated model {model_id}")

    def _get_latest_version(
        self,
        session_id: str,
        dataset_hash: str,
        model_type: str
    ) -> Optional[int]:
        """Get latest version number for a model configuration"""
        models = self.db.query_models(
            session_id=session_id,
            dataset_hash=dataset_hash,
            model_type=model_type
        )

        if not models:
            return None

        return max(m.get("model_version", 1) for m in models)

    @staticmethod
    def compute_dataset_hash(dataset_path: Path) -> str:
        """
        Compute hash of dataset file for tracking

        Args:
            dataset_path: Path to dataset file

        Returns:
            SHA256 hash string
        """
        sha256_hash = hashlib.sha256()

        with open(dataset_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()[:16]  # Use first 16 chars for brevity
