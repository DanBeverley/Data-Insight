"""
GPU Training Integration Test Script

Tests GCP Vertex AI GPU training capabilities.
Run with: python tests/test_gpu_training.py
"""

import os
import sys
from pathlib import Path

# Add project root to path FIRST
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "data_scientist_chatbot" / "app"))
sys.path.insert(0, str(project_root / "data_scientist_chatbot" / "app" / "tools"))
sys.path.insert(0, str(project_root / "data_scientist_chatbot" / "app" / "core"))


def test_gcp_credentials():
    """Test if GCP credentials are configured"""
    print("\n" + "=" * 50)
    print("GCP CREDENTIALS CHECK")
    print("=" * 50)

    required_vars = ["GCP_PROJECT_ID", "GCP_REGION", "GCP_STAGING_BUCKET"]

    all_configured = True
    for var in required_vars:
        value = os.getenv(var, "").strip()
        status = "✅" if value else "❌"
        print(f"  {status} {var}: {value if value else 'NOT SET'}")
        if not value:
            all_configured = False

    return all_configured


def test_gcp_sdk_import():
    """Test if GCP SDK is installed"""
    print("\n" + "=" * 50)
    print("GCP SDK IMPORT TEST")
    print("=" * 50)

    try:
        # Suppress warnings during import
        import warnings

        warnings.filterwarnings("ignore")

        from google.cloud import aiplatform
        from google.cloud import storage

        print("  ✅ google.cloud.aiplatform imported")
        print("  ✅ google.cloud.storage imported")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        print("  💡 Install: pip install google-cloud-aiplatform google-cloud-storage")
        return False


def test_gpu_wrapper():
    """Test if gpu_wrapper.py works correctly"""
    print("\n" + "=" * 50)
    print("GPU WRAPPER TEST")
    print("=" * 50)

    try:
        from gpu_wrapper import ModelFormatDetector

        detector = ModelFormatDetector()
        print("  ✅ ModelFormatDetector initialized")

        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np

            X = np.array([[1], [2], [3]])
            y = np.array([1, 2, 3])
            model = LinearRegression().fit(X, y)

            model_module = type(model).__module__
            print(f"  ✅ Model type detected: {model_module}")

            for key in detector.format_map:
                if key in model_module:
                    ext, _ = detector.format_map[key]
                    print(f"  ✅ Would save as: {ext}")
                    break

            return True
        except ImportError:
            print("  ⚠️ sklearn not installed, skipping model test")
            return True

    except Exception as e:
        print(f"  ❌ GPU wrapper test failed: {e}")
        return False


def test_gcp_training_module():
    """Test if gcp_training module is correct"""
    print("\n" + "=" * 50)
    print("GCP TRAINING MODULE TEST")
    print("=" * 50)

    try:
        from gcp_training import gcp_gpu_train

        print("  ✅ gcp_gpu_train function imported")

        # Check function signature
        import inspect

        sig = inspect.signature(gcp_gpu_train)
        params = list(sig.parameters.keys())
        if "code" in params and "session_id" in params:
            print("  ✅ Function signature correct")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_r2_storage():
    """Test R2 storage availability"""
    print("\n" + "=" * 50)
    print("R2 STORAGE TEST")
    print("=" * 50)

    r2_account = os.getenv("R2_ACCOUNT_ID", "").strip()

    if not r2_account:
        print("  ❌ R2_ACCOUNT_ID not configured")
        return False

    print(f"  ✅ R2_ACCOUNT_ID: {r2_account[:8]}...")

    try:
        from storage.cloud_storage import get_cloud_storage

        r2 = get_cloud_storage()
        if r2:
            print("  ✅ R2 service initialized")
            return True
        else:
            print("  ⚠️ R2 service returned None (check credentials)")
            return False
    except Exception as e:
        print(f"  ❌ R2 initialization failed: {e}")
        return False


def run_all_tests():
    """Run all tests and summarize"""
    print("\n" + "=" * 60)
    print("GCP GPU TRAINING TEST SUITE")
    print("=" * 60)

    results = {}

    results["GCP Credentials"] = test_gcp_credentials()
    results["GCP SDK"] = test_gcp_sdk_import()
    results["GPU Wrapper"] = test_gpu_wrapper()
    results["GCP Training Module"] = test_gcp_training_module()
    results["R2 Storage"] = test_r2_storage()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {test_name}")

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"\n  Total: {passed_count}/{total_count} tests passed")

    return all(results.values())


if __name__ == "__main__":
    # Load .env manually
    env_path = project_root / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

    success = run_all_tests()
    sys.exit(0 if success else 1)
