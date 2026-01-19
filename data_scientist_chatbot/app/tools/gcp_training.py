"""
GCP Vertex AI GPU Training Module

Trains models on GCP Vertex AI with T4 GPU, saves to R2 storage.
"""

import os
import sys
import json
import uuid
import tempfile
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add project paths
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from .logger import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


def gcp_gpu_train(code: str, session_id: str, user_format: Optional[str] = None) -> str:
    """
    Train models on GCP Vertex AI using T4 GPU.

    Args:
        code: Python training code to execute
        session_id: Session identifier for tracking
        user_format: Optional user-specified model save format

    Returns:
        Status message with model path or error
    """
    try:
        from google.cloud import aiplatform
        from google.cloud import storage as gcs_storage
    except ImportError:
        return "Error: google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform google-cloud-storage"

    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION", "us-central1")
    staging_bucket = os.getenv("GCP_STAGING_BUCKET")

    if not project_id:
        return "Error: GCP_PROJECT_ID not configured in .env"
    if not staging_bucket:
        return "Error: GCP_STAGING_BUCKET not configured in .env"

    # Regions with T4 GPU availability
    t4_regions = {
        "us-central1",
        "us-east1",
        "us-east4",
        "us-west1",
        "us-west2",
        "us-west4",
        "europe-west1",
        "europe-west2",
        "europe-west4",
        "asia-east1",
        "asia-northeast1",
        "asia-southeast1",
        "australia-southeast1",
    }

    if region not in t4_regions:
        return f"Error: Region '{region}' may not support T4 GPU. Recommended: us-central1. Update GCP_REGION in .env"

    # Set up credentials
    service_account_json = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if service_account_json and os.path.exists(service_account_json):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json
        logger.info(f"[GCP_GPU] Using service account: {service_account_json}")
    else:
        logger.info("[GCP_GPU] Using Application Default Credentials (ADC)")

    logger.info(f"[GCP_GPU] Initializing Vertex AI for project={project_id}, region={region}")

    try:
        aiplatform.init(project=project_id, location=region, staging_bucket=f"gs://{staging_bucket}")
    except Exception as e:
        return f"Error: Failed to initialize Vertex AI: {e}"

    # Create temporary directory for training script
    job_id = f"quorvix-{session_id[:8]}-{uuid.uuid4().hex[:6]}"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Load GPU wrapper
        wrapper_path = Path(__file__).parent / "core" / "gpu_wrapper.py"
        if not wrapper_path.exists():
            wrapper_path = Path(__file__).parent.parent / "core" / "gpu_wrapper.py"

        if wrapper_path.exists():
            with open(wrapper_path, "r") as f:
                wrapper_code = f.read()
        else:
            logger.warning("[GCP_GPU] gpu_wrapper.py not found, using inline wrapper")
            wrapper_code = _get_inline_wrapper()

        # Create wrapped training script
        wrapped_script = f'''
"""GCP Vertex AI Training Script - Auto-generated"""
import os
import sys
import json

# GPU Wrapper Code
{wrapper_code}

# User Training Code
user_code = """
{code}
"""

if __name__ == "__main__":
    output_dir = os.environ.get("AIP_MODEL_DIR", "/tmp/model")
    os.makedirs(output_dir, exist_ok=True)
    
    result = train_wrapper(
        user_code=user_code,
        output_dir=output_dir,
        user_format={repr(user_format)}
    )
    
    print(f"Training complete: {{result}}")
    
    # Write result to file for retrieval
    with open(os.path.join(output_dir, "training_result.json"), "w") as f:
        json.dump(result, f)
'''

        script_path = Path(tmpdir) / "train.py"
        with open(script_path, "w") as f:
            f.write(wrapped_script)

        # Also create requirements.txt
        requirements = """
scikit-learn>=1.0
pandas>=1.3
numpy>=1.20
joblib>=1.0
torch>=2.0
"""
        req_path = Path(tmpdir) / "requirements.txt"
        with open(req_path, "w") as f:
            f.write(requirements.strip())

        logger.info(f"[GCP_GPU] Created training script at {script_path}")

        try:
            # Convert Windows path to forward slashes for SDK compatibility
            script_path_str = str(script_path).replace("\\", "/")

            # Use TensorFlow GPU container for python package training with GPU
            # TF containers support GPU and can run sklearn/pandas code
            job = aiplatform.CustomJob.from_local_script(
                display_name=job_id,
                script_path=script_path_str,
                container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12.py310:latest",
                requirements=["scikit-learn>=1.0", "pandas>=1.3", "numpy>=1.20", "joblib>=1.0"],
                machine_type="n1-standard-8",
                accelerator_type="NVIDIA_TESLA_V100",
                accelerator_count=1,
                replica_count=1,
                staging_bucket=f"gs://{staging_bucket}",
            )

            logger.info(f"[GCP_GPU] Submitting training job: {job_id}")

            # Run the job (synchronous - waits for completion)
            job.run(sync=True)

            logger.info(f"[GCP_GPU] Training job completed: {job.state}")

            if job.state.name == "JOB_STATE_SUCCEEDED":
                # Download model from GCS staging bucket
                model_gcs_path = f"gs://{staging_bucket}/aiplatform-custom-training/{job_id}/model"
                local_model_dir = Path(tmpdir) / "downloaded_model"
                local_model_dir.mkdir(exist_ok=True)

                # Download model files from GCS
                gcs_client = gcs_storage.Client(project=project_id)
                bucket = gcs_client.bucket(staging_bucket)

                model_files = []
                prefix = f"aiplatform-custom-training/{job_id}/model/"
                blobs = bucket.list_blobs(prefix=prefix)

                for blob in blobs:
                    filename = blob.name.replace(prefix, "")
                    if filename:
                        local_path = local_model_dir / filename
                        blob.download_to_filename(str(local_path))
                        model_files.append(str(local_path))
                        logger.info(f"[GCP_GPU] Downloaded: {filename}")

                if not model_files:
                    return f"Training completed but no model files found in {model_gcs_path}"

                # Upload to R2
                r2_urls = _upload_to_r2(session_id, model_files)

                # Track artifact
                _track_artifact(session_id, model_files, r2_urls)

                return f"Model trained successfully on GCP T4 GPU. Files: {', '.join(Path(f).name for f in model_files)}. Saved to R2."
            else:
                return f"Training job failed with state: {job.state.name}"

        except Exception as e:
            logger.error(f"[GCP_GPU] Training job failed: {e}")
            return f"Error: GCP training failed: {e}"

    return "Error: Unknown error in GCP training"


def _upload_to_r2(session_id: str, model_files: list) -> list:
    """Upload model files to R2 storage"""
    try:
        from src.storage.cloud_storage import get_cloud_storage

        r2 = get_cloud_storage()
        if not r2:
            logger.warning("[GCP_GPU] R2 not configured, skipping cloud upload")
            return []

        urls = []
        for file_path in model_files:
            filename = Path(file_path).name
            blob_path = f"models/{session_id}/{filename}"

            result = r2.upload_file(
                local_path=Path(file_path),
                blob_path=blob_path,
                metadata={"source": "gcp_vertex_ai", "session_id": session_id},
            )

            if result:
                urls.append(result.get("blob_url", blob_path))
                logger.info(f"[GCP_GPU] Uploaded to R2: {blob_path}")

        return urls

    except Exception as e:
        logger.error(f"[GCP_GPU] R2 upload failed: {e}")
        return []


def _track_artifact(session_id: str, model_files: list, r2_urls: list):
    """Track model as artifact"""
    try:
        from src.api_utils.artifact_tracker import get_artifact_tracker

        tracker = get_artifact_tracker()

        for i, file_path in enumerate(model_files):
            filename = Path(file_path).name
            r2_url = r2_urls[i] if i < len(r2_urls) else f"/static/models/{session_id}/{filename}"

            tracker.add_artifact(
                session_id=session_id,
                filename=filename,
                file_path=r2_url,
                description=f"Trained model (GCP Vertex AI T4 GPU)",
                metadata={"type": "model", "environment": "gpu_gcp", "timestamp": datetime.now().isoformat()},
            )
            logger.info(f"[GCP_GPU] Artifact tracked: {filename}")

    except Exception as e:
        logger.error(f"[GCP_GPU] Artifact tracking failed: {e}")


def _get_inline_wrapper() -> str:
    """Fallback inline wrapper if gpu_wrapper.py not found"""
    return '''
import pickle
import json
from pathlib import Path

def train_wrapper(user_code, output_dir, user_format=None):
    """Simple training wrapper"""
    exec_globals = {}
    exec(user_code, exec_globals)
    
    model = exec_globals.get("model")
    if model is None:
        raise ValueError("No trained model found. Assign your model to variable 'model'.")
    
    model_path = Path(output_dir) / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    metadata = {"format": ".pkl", "model_path": str(model_path)}
    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    return metadata
'''


# For testing
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    test_code = """
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)
print(f"Trained model: coef={model.coef_}, intercept={model.intercept_}")
"""

    result = gcp_gpu_train(test_code, "test-session-123")
    print(f"Result: {result}")
