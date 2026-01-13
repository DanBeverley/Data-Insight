"""Cloudflare R2 Storage service for model and artifact persistence (S3-compatible)"""

import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config

    R2_AVAILABLE = True
except ImportError:
    R2_AVAILABLE = False

logger = logging.getLogger(__name__)


class R2StorageService:
    """Manages model and artifact uploads/downloads to Cloudflare R2 (S3-compatible API)"""

    def __init__(
        self,
        account_id: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        bucket_name: str = "Quorvix-artifacts",
        public_url: Optional[str] = None,
    ):
        if not R2_AVAILABLE:
            raise ImportError("boto3 not installed. Run: pip install boto3")

        self.account_id = account_id or os.getenv("R2_ACCOUNT_ID")
        self.access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME", "Quorvix-artifacts")
        self.public_url = public_url or os.getenv("R2_PUBLIC_URL")

        if not self.account_id or not self.access_key_id or not self.secret_access_key:
            raise ValueError(
                "R2 credentials not configured. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY in .env"
            )

        endpoint_url = f"https://{self.account_id}.r2.cloudflarestorage.com"

        config = Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=120,
        )

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=config,
            region_name="auto",
        )

        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.debug(f"Bucket {self.bucket_name} already exists")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Successfully created bucket: {self.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket {self.bucket_name}: {create_error}")
                    logger.error("Cloud storage will not be available. Ensure R2 credentials are configured.")
                    raise
            else:
                logger.error(f"Failed to check bucket {self.bucket_name}: {e}")
                raise

    def upload_file(
        self, local_path: Path, blob_path: str, metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Upload file to R2 storage with SHA256 checksum

        Args:
            local_path: Path to local file
            blob_path: Destination path in R2 (e.g., "user123/models/model.pkl")
            metadata: Optional metadata to attach to object

        Returns:
            Dict with upload info including blob_url, checksum, size
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        checksum = self._compute_sha256(local_path)
        file_size = local_path.stat().st_size

        upload_metadata = metadata or {}
        upload_metadata.update(
            {"sha256": checksum, "uploaded_at": datetime.utcnow().isoformat(), "original_name": local_path.name}
        )

        extra_args = {"Metadata": upload_metadata, "ContentType": self._get_content_type(local_path)}

        try:
            self.s3_client.upload_file(str(local_path), self.bucket_name, blob_path, ExtraArgs=extra_args)

            blob_url = self._get_blob_url(blob_path)
            logger.debug(f"Uploaded {local_path.name} to R2: {blob_path} (SHA256: {checksum[:8]}...)")

            return {
                "blob_path": blob_path,
                "blob_url": blob_url,
                "checksum": checksum,
                "size_bytes": file_size,
                "uploaded_at": upload_metadata["uploaded_at"],
            }
        except ClientError as e:
            logger.error(f"Failed to upload {local_path} to R2: {e}")
            raise

    def download_file(self, blob_path: str, local_path: Path, verify_checksum: bool = True) -> Dict[str, Any]:
        """
        Download file from R2 storage with optional checksum verification

        Args:
            blob_path: Source path in R2 storage
            local_path: Destination path on local filesystem
            verify_checksum: Whether to verify SHA256 checksum

        Returns:
            Dict with download info including checksum match status
        """
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)

            self.s3_client.download_file(self.bucket_name, blob_path, str(local_path))

            result = {"blob_path": blob_path, "local_path": str(local_path), "size_bytes": local_path.stat().st_size}

            if verify_checksum:
                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=blob_path)
                metadata = response.get("Metadata", {})
                expected_checksum = metadata.get("sha256")

                if expected_checksum:
                    actual_checksum = self._compute_sha256(local_path)
                    checksum_match = actual_checksum == expected_checksum
                    result["checksum_verified"] = checksum_match

                    if not checksum_match:
                        logger.error(
                            f"Checksum mismatch for {blob_path}: expected {expected_checksum[:8]}..., got {actual_checksum[:8]}..."
                        )
                        raise ValueError("Checksum verification failed")

                    logger.debug(f"Downloaded {blob_path} with verified checksum")
                else:
                    result["checksum_verified"] = None
                    logger.warning(f"No checksum metadata found for {blob_path}")

            return result

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(f"Object not found in R2: {blob_path}")
            logger.error(f"Failed to download {blob_path} from R2: {e}")
            raise

    def list_blobs(self, prefix: str = "", max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List objects in R2 bucket with optional prefix filter

        Args:
            prefix: Prefix to filter objects (e.g., "user123/models/")
            max_results: Maximum number of results to return

        Returns:
            List of dicts with object info (key, size, last_modified)
        """
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            blobs = []
            for page in page_iterator:
                if "Contents" not in page:
                    break

                for obj in page["Contents"]:
                    blobs.append(
                        {
                            "name": obj["Key"],
                            "size_bytes": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                            "url": self._get_blob_url(obj["Key"]),
                        }
                    )

                    if max_results and len(blobs) >= max_results:
                        break

                if max_results and len(blobs) >= max_results:
                    break

            return blobs

        except ClientError as e:
            logger.error(f"Failed to list objects in R2: {e}")
            raise

    def delete_blob(self, blob_path: str) -> bool:
        """
        Delete object from R2 storage

        Args:
            blob_path: Path to object in R2

        Returns:
            True if deleted successfully
        """
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=blob_path)
            logger.info(f"Deleted {blob_path} from R2")
            return True
        except ClientError as e:
            logger.error(f"Failed to delete {blob_path} from R2: {e}")
            return False

    def get_blob_url(self, blob_path: str, expires_in: int = 3600) -> str:
        """
        Generate presigned URL for temporary access

        Args:
            blob_path: Path to object in R2
            expires_in: URL expiration in seconds (default 1 hour)

        Returns:
            Presigned URL string
        """
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object", Params={"Bucket": self.bucket_name, "Key": blob_path}, ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {blob_path}: {e}")
            raise

    def _get_blob_url(self, blob_path: str) -> str:
        """Get public or presigned URL for blob"""
        if self.public_url:
            return f"{self.public_url.rstrip('/')}/{blob_path}"
        else:
            return f"https://{self.account_id}.r2.cloudflarestorage.com/{self.bucket_name}/{blob_path}"

    def _get_content_type(self, file_path: Path) -> str:
        """Determine content type from file extension"""
        import mimetypes

        content_type, _ = mimetypes.guess_type(str(file_path))
        return content_type or "application/octet-stream"

    def blob_exists(self, blob_path: str) -> bool:
        """Check if object exists in R2"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=blob_path)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def get_blob_metadata(self, blob_path: str) -> Dict[str, Any]:
        """Retrieve object metadata from R2"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=blob_path)

            return {
                "blob_path": blob_path,
                "size_bytes": response["ContentLength"],
                "content_type": response.get("ContentType", "application/octet-stream"),
                "last_modified": response["LastModified"].isoformat(),
                "metadata": response.get("Metadata", {}),
            }
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                raise FileNotFoundError(f"Object not found in R2: {blob_path}")
            raise

    def get_bucket_stats(self) -> Dict[str, Any]:
        """
        Get bucket usage statistics.

        Returns:
            Dict with used_gb, total_gb, file_count, bucket_name
        """
        try:
            # List all objects to calculate total size
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name)

            total_bytes = 0
            file_count = 0

            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        total_bytes += obj["Size"]
                        file_count += 1

            # Convert bytes to GB
            used_gb = total_bytes / (1024**3)

            return {
                "used_gb": round(used_gb, 2),
                "total_gb": 100,  # Can be configured based on plan
                "file_count": file_count,
                "bucket_name": self.bucket_name,
                "total_bytes": total_bytes,
            }

        except ClientError as e:
            logger.error(f"Failed to get bucket stats: {e}")
            raise

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


R2Service = R2StorageService  # Alias for backwards compatibility
