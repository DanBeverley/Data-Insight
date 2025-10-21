"""Azure Blob Storage service for model and artifact persistence"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import logging

try:
    from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class BlobStorageService:
    """Manages model and artifact uploads/downloads to Azure Blob Storage"""

    def __init__(
        self,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        container_name: str = "datainsight-models"
    ):
        if not AZURE_AVAILABLE:
            raise ImportError("azure-storage-blob not installed. Run: pip install azure-storage-blob azure-identity")

        self.account_name = account_name or os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = account_key or  os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        self.container_name = container_name

        if not self.account_name:
            raise ValueError("Azure storage account name not configured")

        if self.account_key:
            connection_string = f"DefaultEndpointsProtocol=https;AccountName={self.account_name};AccountKey={self.account_key};EndpointSuffix=core.windows.net"
            self.client = BlobServiceClient.from_connection_string(connection_string)
        else:
            credential = DefaultAzureCredential()
            account_url = f"https://{self.account_name}.blob.core.windows.net"
            self.client = BlobServiceClient(account_url=account_url, credential=credential)

        self.container_client = self.client.get_container_client(self.container_name)
        self._ensure_container_exists()

    def _ensure_container_exists(self) -> None:
        """Create container if it doesn't exist"""
        try:
            self.container_client.get_container_properties()
        except Exception:
            try:
                self.container_client.create_container()
                logger.info(f"Created container: {self.container_name}")
            except Exception as e:
                logger.error(f"Failed to create container: {e}")
                raise

    def upload_file(
        self,
        local_path: Path,
        blob_path: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload file to blob storage with SHA256 checksum

        Args:
            local_path: Path to local file
            blob_path: Destination path in blob storage (e.g., "user123/models/model.pkl")
            metadata: Optional metadata to attach to blob

        Returns:
            Dict with upload info including blob_url, checksum, size
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        checksum = self._compute_sha256(local_path)
        file_size = local_path.stat().st_size

        upload_metadata = metadata or {}
        upload_metadata.update({
            "sha256": checksum,
            "uploaded_at": datetime.utcnow().isoformat(),
            "original_name": local_path.name
        })

        blob_client = self.container_client.get_blob_client(blob_path)

        with open(local_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                metadata=upload_metadata
            )

        logger.info(f"Uploaded {local_path.name} to {blob_path} (SHA256: {checksum[:8]}...)")

        return {
            "blob_path": blob_path,
            "blob_url": blob_client.url,
            "checksum": checksum,
            "size_bytes": file_size,
            "uploaded_at": upload_metadata["uploaded_at"]
        }

    def download_file(
        self,
        blob_path: str,
        local_path: Path,
        verify_checksum: bool = True
    ) -> Dict[str, Any]:
        """
        Download file from blob storage with optional checksum verification

        Args:
            blob_path: Source path in blob storage
            local_path: Destination path on local filesystem
            verify_checksum: Whether to verify SHA256 checksum

        Returns:
            Dict with download info including checksum match status
        """
        blob_client = self.container_client.get_blob_client(blob_path)

        if not blob_client.exists():
            raise FileNotFoundError(f"Blob not found: {blob_path}")

        local_path.parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

        result = {
            "blob_path": blob_path,
            "local_path": str(local_path),
            "size_bytes": local_path.stat().st_size
        }

        if verify_checksum:
            properties = blob_client.get_blob_properties()
            expected_checksum = properties.metadata.get("sha256")

            if expected_checksum:
                actual_checksum = self._compute_sha256(local_path)
                checksum_match = (actual_checksum == expected_checksum)
                result["checksum_verified"] = checksum_match

                if not checksum_match:
                    logger.error(f"Checksum mismatch for {blob_path}: expected {expected_checksum[:8]}..., got {actual_checksum[:8]}...")
                    raise ValueError("Checksum verification failed")

                logger.info(f"Downloaded {blob_path} with verified checksum")
            else:
                result["checksum_verified"] = None
                logger.warning(f"No checksum metadata found for {blob_path}")

        return result

    def get_presigned_url(
        self,
        blob_path: str,
        expiry_hours: int = 24,
        permissions: str = "r"
    ) -> str:
        """
        Generate pre-signed SAS URL for temporary access

        Args:
            blob_path: Path to blob
            expiry_hours: Hours until URL expires
            permissions: Permissions string ('r' for read, 'rw' for read-write)

        Returns:
            Pre-signed URL string
        """
        if not self.account_key:
            raise ValueError("Account key required for SAS token generation")

        blob_client = self.container_client.get_blob_client(blob_path)

        sas_permissions = BlobSasPermissions(read=True)
        if 'w' in permissions:
            sas_permissions.write = True

        sas_token = generate_blob_sas(
            account_name=self.account_name,
            container_name=self.container_name,
            blob_name=blob_path,
            account_key=self.account_key,
            permission=sas_permissions,
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours)
        )

        return f"{blob_client.url}?{sas_token}"

    def blob_exists(self, blob_path: str) -> bool:
        """Check if blob exists in storage"""
        blob_client = self.container_client.get_blob_client(blob_path)
        return blob_client.exists()

    def get_blob_metadata(self, blob_path: str) -> Dict[str, Any]:
        """Retrieve blob metadata"""
        blob_client = self.container_client.get_blob_client(blob_path)
        properties = blob_client.get_blob_properties()

        return {
            "blob_path": blob_path,
            "size_bytes": properties.size,
            "content_type": properties.content_settings.content_type,
            "last_modified": properties.last_modified.isoformat(),
            "metadata": properties.metadata
        }

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()
