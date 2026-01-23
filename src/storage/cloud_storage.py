"""Unified cloud storage interface using Cloudflare R2"""

import os
import logging

logger = logging.getLogger(__name__)

_r2_service_cache = None


def get_cloud_storage(bucket_name: str = None):
    """
    Factory function to get Cloudflare R2 storage service (singleton).

    Args:
        bucket_name: Name of the R2 bucket (single bucket for all artifacts)

    Returns:
        R2StorageService instance or None if not configured
    """
    global _r2_service_cache

    if _r2_service_cache is not None:
        return _r2_service_cache

    r2_account_id = os.getenv("R2_ACCOUNT_ID")

    if r2_account_id:
        try:
            from .r2_service import R2StorageService

            logger.info("Initializing Cloudflare R2 storage (one-time)")
            _r2_service_cache = R2StorageService(bucket_name=bucket_name)
            return _r2_service_cache
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
