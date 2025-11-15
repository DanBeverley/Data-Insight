"""Unified cloud storage interface using Cloudflare R2"""

import os
import logging

logger = logging.getLogger(__name__)


def get_cloud_storage(bucket_name: str = "datainsight-artifacts"):
    """
    Factory function to get Cloudflare R2 storage service.

    Args:
        bucket_name: Name of the R2 bucket (single bucket for all artifacts)

    Returns:
        R2StorageService instance or None if not configured
    """
    r2_account_id = os.getenv("R2_ACCOUNT_ID")

    if r2_account_id:
        try:
            from .r2_service import R2StorageService

            logger.info("Using Cloudflare R2 for cloud storage")
            return R2StorageService(bucket_name=bucket_name)
        except Exception as e:
            logger.error(f"Failed to initialize R2 storage: {e}")
            logger.warning("Falling back to local storage only")
            return None
    else:
        logger.info("No cloud storage configured - using local storage only")
        return None


def is_cloud_storage_available() -> bool:
    """Check if R2 cloud storage is configured and available"""
    return os.getenv("R2_ACCOUNT_ID") is not None
