"""
GCP GPU Training End-to-End Test

This test actually submits a job to GCP Vertex AI.
WARNING: This will incur GCP charges (~$0.01-0.05 for a short test).

Run with: py -3.11 tests/test_gcp_gpu_e2e.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "data_scientist_chatbot" / "app"))
sys.path.insert(0, str(project_root / "data_scientist_chatbot" / "app" / "tools"))


def load_env():
    """Load .env file manually"""
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def test_gcp_gpu_training():
    """
    End-to-end test for GCP Vertex AI GPU training.
    Submits a real training job to verify the integration works.
    """
    print("\n" + "=" * 60)
    print("GCP VERTEX AI GPU TRAINING - END-TO-END TEST")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")

    # Check prerequisites
    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION")
    bucket = os.getenv("GCP_STAGING_BUCKET")

    print("\n📋 Prerequisites Check:")
    print(f"  GCP_PROJECT_ID: {project_id}")
    print(f"  GCP_REGION: {region}")
    print(f"  GCP_STAGING_BUCKET: {bucket}")

    if not all([project_id, region, bucket]):
        print("\n❌ Missing GCP configuration in .env")
        return False

    # Simple training code for testing
    test_code = """
import numpy as np
from sklearn.linear_model import LinearRegression

print("Starting simple model training...")

# Create simple dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2.1, 4.0, 5.9, 8.1, 10.0, 12.1, 13.9, 16.0, 18.1, 19.9])

# Train model
model = LinearRegression()
model.fit(X, y)

print(f"Model trained successfully!")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"R² Score: {model.score(X, y):.4f}")
"""

    session_id = f"test-gpu-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"\n🚀 Submitting GPU training job...")
    print(f"  Session ID: {session_id}")

    try:
        from gcp_training import gcp_gpu_train

        result = gcp_gpu_train(code=test_code, session_id=session_id, user_format=None)

        print(f"\n📊 Result:")
        print(f"  {result}")

        if "successfully" in result.lower():
            print("\n✅ GPU TRAINING TEST PASSED!")
            return True
        elif "error" in result.lower():
            print(f"\n❌ GPU TRAINING TEST FAILED: {result}")
            return False
        else:
            print(f"\n⚠️ Unexpected result: {result}")
            return False

    except Exception as e:
        print(f"\n❌ Exception during training: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_cpu_fallback():
    """
    Test that CPU fallback works when GPU is not available.
    """
    print("\n" + "=" * 60)
    print("CPU FALLBACK TEST")
    print("=" * 60)

    try:
        from core.training_executor import TrainingExecutor

        executor = TrainingExecutor()

        simple_code = """
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3]])
y = np.array([1, 2, 3])
model = LinearRegression().fit(X, y)
print(f"Model trained: coef={model.coef_}")
"""

        session_id = f"test-cpu-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        print(f"Testing CPU execution for session: {session_id}")

        result = executor.execute_training(
            code=simple_code, session_id=session_id, user_request="Train a simple model", model_type="linear_regression"
        )

        print(f"Execution environment: {result.get('execution_environment', 'unknown')}")
        print(f"Success: {result.get('success', False)}")

        if result.get("success"):
            print("\n✅ CPU FALLBACK TEST PASSED!")
            return True
        else:
            print(f"\n❌ CPU TEST FAILED: {result.get('stderr', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    load_env()

    print("\n" + "=" * 60)
    print("GCP GPU TRAINING - INTEGRATION TEST SUITE")
    print("=" * 60)
    print("\n⚠️  WARNING: GPU test will submit a real job to GCP Vertex AI")
    print("   This may incur charges (~$0.01-0.05 for a short test)")

    response = input("\nProceed with tests? (y/n): ").strip().lower()

    if response != "y":
        print("Tests cancelled.")
        return

    results = {}

    # Test 1: CPU Fallback (safe, no GCP charges)
    results["CPU Fallback"] = test_cpu_fallback()

    # Test 2: GPU Training (real GCP call)
    gpu_response = input("\nRun GPU training test? (y/n): ").strip().lower()
    if gpu_response == "y":
        results["GPU Training"] = test_gcp_gpu_training()
    else:
        print("GPU test skipped.")
        results["GPU Training"] = None

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        if passed is None:
            status = "⏭️ SKIPPED"
        elif passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  {status} - {test_name}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
