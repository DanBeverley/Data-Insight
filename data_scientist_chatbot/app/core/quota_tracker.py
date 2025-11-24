"""GPU quota tracking for AWS SageMaker and Azure ML"""

import os
from typing import Dict, Optional
from datetime import datetime, timedelta

try:
    from .logger import logger
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from logger import logger


class QuotaTracker:
    """
    Track GPU quota usage for cloud providers
    Caches results to avoid excessive API calls
    """

    def __init__(self):
        self.cache_ttl = timedelta(minutes=5)
        self._cache = {}

    def get_quota_status(self) -> Dict[str, Dict]:
        """
        Get quota status for all configured providers

        Returns:
            Dict with provider keys containing usage info
        """
        status = {}

        # Check AWS if credentials configured
        if os.getenv("AWS_ACCESS_KEY_ID"):
            status["aws"] = self._get_aws_quota()

        # Check Azure if credentials configured
        if os.getenv("AZURE_SUBSCRIPTION_ID"):
            status["azure"] = self._get_azure_quota()

        return status

    def _get_aws_quota(self) -> Dict:
        """Get AWS SageMaker quota and usage"""
        cache_key = "aws_quota"

        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            import boto3

            # Service Quotas client for limits
            sq_client = boto3.client("service-quotas", region_name=os.getenv("AWS_REGION", "us-east-1"))

            # SageMaker client for usage
            sm_client = boto3.client("sagemaker", region_name=os.getenv("AWS_REGION", "us-east-1"))

            # Get training job quota (GPU hours)
            try:
                quota_response = sq_client.get_service_quota(
                    ServiceCode="sagemaker", QuotaCode="L-47A926C0"  # Training job quota code
                )
                max_hours = quota_response["Quota"]["Value"]
            except Exception:
                max_hours = 100  # Default fallback

            # Approximate current usage by listing recent jobs
            try:
                jobs_response = sm_client.list_training_jobs(
                    MaxResults=50, SortBy="CreationTime", SortOrder="Descending"
                )

                # Estimate hours used (very rough approximation)
                used_hours = len(jobs_response.get("TrainingJobSummaries", [])) * 0.5  # Assume 30min avg per job
            except Exception:
                used_hours = 0

            # Get network bandwidth (placeholder - real impl would use CloudWatch)
            bandwidth = "N/A"

            result = {
                "provider": "AWS",
                "used": round(used_hours, 1),
                "total": int(max_hours),
                "unit": "hrs",
                "bandwidth": bandwidth,
                "available": max_hours - used_hours > 0,
                "timestamp": datetime.now().isoformat(),
            }

            self._update_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"[QuotaTracker] AWS quota check failed: {e}")
            return {
                "provider": "AWS",
                "used": 0,
                "total": 0,
                "unit": "hrs",
                "bandwidth": "N/A",
                "available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _get_azure_quota(self) -> Dict:
        """Get Azure ML quota and usage"""
        cache_key = "azure_quota"

        if self._is_cached(cache_key):
            return self._cache[cache_key]["data"]

        try:
            from azure.mgmt.compute import ComputeManagementClient
            from azure.identity import DefaultAzureCredential

            subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
            location = os.getenv("AZURE_LOCATION", "eastus")

            credential = DefaultAzureCredential()
            compute_client = ComputeManagementClient(credential, subscription_id)

            # Get compute usage/quota
            usage = compute_client.usage.list(location)

            # Find GPU-related quota (NC series for GPUs)
            gpu_usage = None
            for item in usage:
                if "standardNC" in item.name.value.lower() or "gpu" in item.name.value.lower():
                    gpu_usage = item
                    break

            if gpu_usage:
                used = gpu_usage.current_value
                total = gpu_usage.limit
            else:
                # Fallback values
                used = 0
                total = 100

            # Bandwidth placeholder
            bandwidth = "N/A"

            result = {
                "provider": "Azure",
                "used": used,
                "total": total,
                "unit": "cores",
                "bandwidth": bandwidth,
                "available": used < total,
                "timestamp": datetime.now().isoformat(),
            }

            self._update_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"[QuotaTracker] Azure quota check failed: {e}")
            return {
                "provider": "Azure",
                "used": 0,
                "total": 0,
                "unit": "cores",
                "bandwidth": "N/A",
                "available": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _is_cached(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self._cache:
            return False

        cache_entry = self._cache[key]
        age = datetime.now() - cache_entry["cached_at"]
        return age < self.cache_ttl

    def _update_cache(self, key: str, data: Dict):
        """Update cache with new data"""
        self._cache[key] = {"data": data, "cached_at": datetime.now()}

    def has_available_quota(self, provider: str) -> bool:
        """
        Quick check if provider has available quota

        Args:
            provider: 'aws' or 'azure'

        Returns:
            True if quota available
        """
        status = self.get_quota_status()

        if provider.lower() not in status:
            return False

        provider_status = status[provider.lower()]
        return provider_status.get("available", False)


# Global instance
quota_tracker = QuotaTracker()
