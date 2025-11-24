import os
import sys
import uuid
import json
import logging
import io
from typing import Dict, Any, TYPE_CHECKING
from pathlib import Path
from dotenv import load_dotenv
from langsmith import traceable

if TYPE_CHECKING:
    from e2b_code_interpreter import Sandbox

logger = logging.getLogger(__name__)

try:
    from langchain.tools import tool
except ImportError:
    logger.warning("langchain.tools not available - tool decorator will be disabled")

    def tool(func):
        return func


project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
load_dotenv(env_file)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from performance_monitor import PerformanceMonitor
except ImportError:
    try:
        from .performance_monitor import PerformanceMonitor
    except ImportError:
        raise ImportError("Could not import PerformanceMonitor")

# Use builtins for persistent storage across module reloads
import builtins

if not hasattr(builtins, "_persistent_sandboxes"):
    builtins._persistent_sandboxes = {}
    print(f"DEBUG: Initialized persistent sandbox storage")
session_sandboxes = builtins._persistent_sandboxes
print(f"DEBUG: tools.py module loaded, using persistent sandboxes: {id(session_sandboxes)}")
performance_monitor = PerformanceMonitor()


def get_sandbox(session_id: str) -> "Sandbox":
    from e2b_code_interpreter import Sandbox

    if session_id not in session_sandboxes:
        print(f"DEBUG: Creating new sandbox for session {session_id}")
        sandbox = Sandbox.create(timeout=600)
        session_sandboxes[session_id] = sandbox
        _reload_dataset_if_available(sandbox, session_id)
    else:
        print(f"DEBUG: Reusing existing sandbox for session {session_id}")
        try:
            session_sandboxes[session_id].run_code("print('health_check')", timeout=5)
        except Exception as e:
            print(f"DEBUG: Sandbox health check failed, recreating: {e}")
            try:
                session_sandboxes[session_id].close()
            except:
                pass
            sandbox = Sandbox.create()
            session_sandboxes[session_id] = sandbox
            _reload_dataset_if_available(sandbox, session_id)
    return session_sandboxes[session_id]


def _reload_dataset_if_available(sandbox: "Sandbox", session_id: str):
    try:
        import builtins

        session_store = getattr(builtins, "_session_store", None)
        if session_store and session_id in session_store and "dataframe" in session_store[session_id]:
            df_original = session_store[session_id]["dataframe"]

            df_to_upload = (
                df_original.sample(n=min(50000, len(df_original)), random_state=42)
                if len(df_original) > 50000
                else df_original
            )

            parquet_buffer = io.BytesIO()
            df_to_upload.to_parquet(parquet_buffer, engine="pyarrow", compression="snappy")
            parquet_bytes = parquet_buffer.getvalue()

            print(f"DEBUG: Uploading {len(parquet_bytes) / 1024 / 1024:.1f}MB Parquet ({len(df_to_upload)} rows)")

            try:
                sandbox.filesystem.write_bytes("/tmp/dataset.parquet", parquet_bytes)
            except AttributeError:
                sandbox.files.write("/tmp/dataset.parquet", parquet_bytes)

            check_and_install_code = """
import subprocess
import sys
from packaging import version

try:
    import pandas as pd
    import pyarrow as pa
    pandas_ok = version.parse(pd.__version__) >= version.parse('2.2.0')
    pyarrow_ok = version.parse(pa.__version__) >= version.parse('17.0.0')

    if not (pandas_ok and pyarrow_ok):
        raise ImportError("Versions too old")
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', '-q', 'pandas==2.2.3', 'pyarrow==17.0.0'])
    print("DEBUG: Upgraded pandas to 2.2.3 and pyarrow to 17.0.0")
else:
    print(f"DEBUG: Pandas {pd.__version__} and PyArrow {pa.__version__} already installed")
"""
            sandbox.run_code(check_and_install_code, timeout=60)

    except Exception as e:
        print(f"DEBUG: Parquet reload failed, falling back to CSV: {e}")
        try:
            import builtins

            session_store = getattr(builtins, "_session_store", None)
            if session_store and session_id in session_store and "dataframe" in session_store[session_id]:
                df = session_store[session_id]["dataframe"]
                df_sample = df.sample(n=min(20000, len(df)), random_state=42) if len(df) > 20000 else df
                csv_data = df_sample.to_csv(index=False)

                reload_code = f"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

csv_data = '''{csv_data}'''
df = pd.read_csv(StringIO(csv_data))
print(f"Dataset reloaded: {{df.shape}} shape, {{len(df.columns)}} columns")
"""
                result = sandbox.run_code(reload_code, timeout=60)
        except Exception as fallback_error:
            print(f"DEBUG: Could not reload dataset for session {session_id}: {fallback_error}")


@traceable(name="sandbox_execution", tags=["tool", "e2b"])
@performance_monitor.cache_result(ttl=600, key_prefix="sandbox_exec")
@performance_monitor.time_function("sandbox", "code_execution")
def execute_python_in_sandbox(code: str, session_id: str) -> Dict[str, Any]:
    """
    Executes Python code in a stateful, secure sandbox for a specific session.
    Enhanced with performance monitoring and intelligent caching.
    Routes training code to GPU when appropriate.
    """
    if not session_id:
        performance_monitor.record_metric(
            session_id="unknown", metric_name="sandbox_error", value=1.0, context={"error": "missing_session_id"}
        )
        return {"success": False, "stderr": "Session ID is missing."}

    # Check if training decision exists for this session
    import builtins

    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
        session_data = builtins._session_store[session_id]
        training_decision = session_data.get("training_decision")

        if training_decision and training_decision.get("environment") == "gpu":
            print(f"[execute_python_in_sandbox] Using pre-decided GPU execution: {training_decision.get('reasoning')}")

            # Import training executor
            sys.path.append(os.path.join(os.path.dirname(__file__), "core"))
            from core.training_executor import training_executor

            # Clear the decision to avoid re-use
            session_data["training_decision"] = None
            session_data["training_environment"] = None

            # Execute using the pre-made decision (skip re-deciding)
            from core.training_decision import TrainingDecision

            decision_obj = TrainingDecision(
                environment=training_decision["environment"],
                reasoning=training_decision["reasoning"],
                confidence=training_decision["confidence"],
            )

            # Route directly based on decision
            if decision_obj.environment == "gpu":
                result = training_executor._execute_on_gpu(
                    code=code, session_id=session_id, user_format=None, decision=decision_obj
                )
            else:
                result = training_executor._execute_on_cpu(code=code, session_id=session_id, decision=decision_obj)

            print(f"[execute_python_in_sandbox] Training execution complete: {result.get('execution_environment')}")
            return result

    sandbox = get_sandbox(session_id)
    plot_urls = []
    model_urls = []
    dataset_urls = []

    try:
        import psutil
        import sys
        import os

        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
        from src.mlops.monitoring import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()
        # Record initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()

        import builtins

        session_store = getattr(builtins, "_session_store", None)

        is_plotting = any(pattern in code for pattern in ["plt.", "sns.", ".plot(", ".hist(", "matplotlib", "seaborn"])
        patterns_found = [p for p in ["plt.", "sns.", ".plot(", ".hist(", "matplotlib", "seaborn"] if p in code]
        print(f"DEBUG: Plot detection - code contains: {patterns_found}")
        if "df.corr()" in code:
            code = code.replace("df.corr()", "df.select_dtypes(include=[np.number]).corr()")

        import re

        verbose_patterns_found = []
        lines = code.split("\n")
        cleaned_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]

            if "import pandas" in line or "from pandas" in line:
                cleaned_lines.append("# pandas already imported as pd")
                verbose_patterns_found.append("import pandas")
                i += 1
                continue

            if re.match(r"^\s*df\s*=\s*pd\.read_", line):
                cleaned_lines.append("# df already loaded from /tmp/dataset.parquet")
                verbose_patterns_found.append("df = pd.read_*()")
                i += 1
                continue

            if ".value_counts()" in line and "print" in line and ".head(" not in line:
                cleaned_lines.append(line.replace(".value_counts()", ".value_counts().head(10)"))
                verbose_patterns_found.append("value_counts()")
                i += 1
                continue

            if "print(df[" in line and ".value_counts()" in line:
                if ".head(" not in line:
                    cleaned_lines.append(line.replace(".value_counts()", ".value_counts().head(10)"))
                    verbose_patterns_found.append("df[col].value_counts()")
                    i += 1
                    continue

            cleaned_lines.append(line)
            i += 1

        code = "\n".join(cleaned_lines)

        if verbose_patterns_found:
            print(f"DEBUG: Suppressed patterns: {', '.join(set(verbose_patterns_found))}")

        indented_code = "\n".join("    " + line if line.strip() else "" for line in code.split("\n"))

        df_rows = (
            len(session_store[session_id]["dataframe"])
            if session_store and session_id in session_store and "dataframe" in session_store[session_id]
            else 0
        )
        sampling_code = ""
        if is_plotting and df_rows > 50000:
            sampling_code = """# Auto-sampling for large dataset visualization
if 'df' in locals() and len(df) > 50000:
    print("Dataset has " + str(len(df)) + " rows - sampling 50,000 rows for visualization performance")
    df_plot = df.sample(n=min(50000, len(df)), random_state=42).sort_index()
else:
    df_plot = df if 'df' in locals() else None
"""

        enhanced_code = f"""import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import glob

# Apply pandas output limits
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)
pd.set_option('display.max_colwidth', 50)

df = pd.read_parquet('/tmp/dataset.parquet')

{sampling_code}

before_pngs = set(glob.glob('*.png') + glob.glob('/tmp/*.png'))

model_exts = ['*.pkl', '*.joblib', '*.h5', '*.pt', '*.pth', '*.json', '*.txt', '*.cbm', '*.onnx']
before_models = set()
for ext in model_exts:
    before_models.update(glob.glob(ext))
    before_models.update(glob.glob('/tmp/' + ext))

try:
{indented_code}
except Exception as e:
    print("Execution error: " + type(e).__name__ + ": " + str(e))
    import traceback
    traceback.print_exc()

try:
    unsaved_figures = plt.get_fignums()
    if unsaved_figures:
        print(f"WARNING: {{len(unsaved_figures)}} matplotlib figure(s) created but not saved. Use plt.savefig('descriptive_name.png') to save plots.")
        for fig_num in unsaved_figures:
            plt.close(fig_num)
except:
    pass

after_pngs = set(glob.glob('*.png') + glob.glob('/tmp/*.png'))
new_pngs = after_pngs - before_pngs

for png in new_pngs:
    if os.path.exists(png) and os.path.getsize(png) > 100:
        print("PLOT_SAVED:" + os.path.basename(png))

after_models = set()
for ext in model_exts:
    after_models.update(glob.glob(ext))
    after_models.update(glob.glob('/tmp/' + ext))

new_models = after_models - before_models
for model_file in new_models:
    if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
        print("MODEL_SAVED:" + os.path.basename(model_file))
"""

        timeout = 600 if df_rows > 200000 else 300 if df_rows > 100000 else 180 if df_rows > 10000 else 90

        print(f"DEBUG: Executing code with timeout={timeout}s")

        try:
            result = sandbox.run_code(enhanced_code, timeout=timeout)
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            if "RemoteProtocolError" in error_type or "incomplete chunked read" in error_msg:
                print(f"DEBUG: Stdout overflow detected - connection dropped during streaming")
                print(f"DEBUG: This usually means generated output exceeded HTTP buffer limits")
                print(f"DEBUG: Falling back to minimal output code...")

                minimal_code = """
print("Code execution completed but output was too large to stream.")
print("Dataset shape:", df.shape if 'df' in locals() else 'N/A')
print("Columns:", list(df.columns) if 'df' in locals() else 'N/A')
"""
                try:
                    result = sandbox.run_code(minimal_code, timeout=30)
                except:
                    return {
                        "success": False,
                        "stderr": "Output too large - exceeded streaming limits",
                        "stdout": "Execution completed but output was truncated",
                        "plots": [],
                        "models": [],
                    }

            elif "TimeoutException" in error_type or "timed out" in error_msg.lower():
                print(f"DEBUG: Execution timeout after {timeout}s")
                print(f"DEBUG: Code may have hung or be computationally intensive")
                return {
                    "success": False,
                    "stderr": f"Execution exceeded {timeout}s timeout",
                    "stdout": "Code execution took too long and was terminated",
                    "plots": [],
                    "models": [],
                }
            else:
                raise

        if hasattr(result, "error") and result.error:
            error_msg = f"{result.error.name}: {result.error.value}"
            print(f"ERROR: E2B execution failed: {error_msg}")
            performance_monitor.record_metric(
                session_id=session_id, metric_name="sandbox_error", value=1.0, context={"error": error_msg}
            )
            return {"success": False, "stderr": error_msg, "stdout": "", "plots": [], "models": [], "datasets": []}

        performance_monitor.record_metric(
            session_id=session_id, metric_name="sandbox_success", value=1.0, context={"code_length": len(code)}
        )
        if hasattr(result, "logs"):
            stdout_lines = result.logs.stdout if hasattr(result.logs, "stdout") else []
            stderr_lines = result.logs.stderr if hasattr(result.logs, "stderr") else []
        elif hasattr(result, "stdout"):
            stdout_lines = result.stdout if result.stdout else []
            stderr_lines = result.stderr if hasattr(result, "stderr") and result.stderr else []
        else:
            stdout_lines = []
            stderr_lines = []
            print(f"WARNING: Unexpected result structure, trying to extract from string representation")
            result_str = str(result)
            print(f"DEBUG: Result string: {result_str}")
        if isinstance(stdout_lines, str):
            stdout_content = stdout_lines
        elif isinstance(stdout_lines, list):
            stdout_content = "\n".join(stdout_lines) if stdout_lines else ""
        else:
            stdout_content = str(stdout_lines) if stdout_lines else ""
        if isinstance(stderr_lines, str):
            stderr_content = stderr_lines
        elif isinstance(stderr_lines, list):
            stderr_content = "\n".join(stderr_lines) if stderr_lines else ""
        else:
            stderr_content = str(stderr_lines) if stderr_lines else ""

        stdout_size = len(stdout_content)
        if stdout_size > 50000:
            print(f"DEBUG: Large stdout detected ({stdout_size} bytes) - truncating to prevent display issues")
            stdout_content = (
                stdout_content[:25000]
                + f"\n\n... [Output truncated: {stdout_size - 50000} bytes omitted] ...\n\n"
                + stdout_content[-25000:]
            )

        from pathlib import Path

        if stdout_content:
            import os
            from pathlib import Path

            current_path = Path(__file__).resolve()
            project_root = current_path
            while project_root.name != "Data-Insight" and project_root.parent != project_root:
                project_root = project_root.parent

            processed_files = set()

            static_dir = project_root / "static" / "plots"
            static_dir.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Static dir set to: {static_dir}")

            for line in stdout_content.split("\n"):
                if line.startswith("PLOT_SAVED:"):
                    sandbox_filename = line.split(":")[1].strip()

                    if sandbox_filename in processed_files:
                        print(f"Skipping duplicate: {sandbox_filename}")
                        continue
                    processed_files.add(sandbox_filename)

                    local_path = static_dir / sandbox_filename
                    print(f"Attempting to download {sandbox_filename} to {local_path}")

                    potential_paths = [
                        sandbox_filename,
                        f"/home/user/{sandbox_filename}",
                        f"/tmp/{sandbox_filename}",
                        f"./{sandbox_filename}",
                    ]

                    file_downloaded = False
                    for path in potential_paths:
                        try:
                            print(f"Trying to download from sandbox path: {path}")
                            # Use the correct E2B method for binary file operations
                            file_content_bytes = sandbox.files.read(path, format="bytes")
                            print(f"Downloaded {len(file_content_bytes) if file_content_bytes else 0} bytes")

                            if file_content_bytes and len(file_content_bytes) > 8:
                                # Check PNG header
                                if file_content_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                                    with open(local_path, "wb") as f:
                                        f.write(file_content_bytes)
                                    print(f"Successfully wrote to {local_path}")
                                    if local_path.exists() and local_path.stat().st_size > 0:
                                        web_url = f"/static/plots/{sandbox_filename}"
                                        plot_url = web_url
                                        file_downloaded = True
                                        logger.info(f"File verified: {sandbox_filename}")

                                        blob_path = None
                                        blob_url = None
                                        presigned_url = None

                                        try:
                                            import sys

                                            src_path = str(project_root / "src")
                                            if src_path not in sys.path:
                                                sys.path.insert(0, src_path)

                                            from src.config import settings
                                            from src.storage.cloud_storage import get_cloud_storage

                                            storage_config = settings.get("object_storage", {})
                                            if storage_config.get("enabled") and storage_config.get("upload_to_cloud"):
                                                bucket_name = storage_config.get("bucket_name", "datainsight-artifacts")
                                                folder_prefix = storage_config.get("folders", {}).get("plots", "plots")
                                                cloud_storage = get_cloud_storage(bucket_name=bucket_name)

                                                if cloud_storage:
                                                    blob_path = f"{folder_prefix}/{session_id}/{sandbox_filename}"
                                                    upload_result = cloud_storage.upload_file(
                                                        local_path=local_path,
                                                        blob_path=blob_path,
                                                        metadata={"session_id": session_id, "type": "plot"},
                                                    )
                                                    blob_url = upload_result["blob_url"]
                                                    presigned_url = cloud_storage.get_blob_url(
                                                        blob_path=blob_path,
                                                        expires_in=storage_config.get("presigned_url_expiry_hours", 24)
                                                        * 3600,
                                                    )
                                                plot_url = presigned_url
                                                logger.info(f"Uploaded plot to cloud: {blob_path}")

                                                if not storage_config.get("keep_local_copy"):
                                                    local_path.unlink()
                                        except Exception as cloud_error:
                                            logger.warning(f"Cloud upload failed: {cloud_error}, using local URL")

                                        plot_urls.append(plot_url)

                                        try:
                                            from src.api_utils.artifact_tracker import get_artifact_tracker

                                            tracker = get_artifact_tracker()
                                            tracker.add_artifact(
                                                session_id=session_id,
                                                filename=sandbox_filename,
                                                file_path=web_url,
                                                description="Generated visualization",
                                                metadata={"type": "plot", "format": "png"},
                                                blob_path=blob_path,
                                                blob_url=blob_url,
                                                presigned_url=presigned_url,
                                            )
                                            logger.info(f"Artifact tracked: {sandbox_filename}")
                                        except Exception as e:
                                            logger.error(f"Artifact tracking error: {e}")
                                        break
                                    else:
                                        print(f"File verification failed for {local_path}")
                                else:
                                    print(f"Invalid PNG header: {file_content_bytes[:8]}")
                            else:
                                print(f"No valid content downloaded from {path}")
                        except Exception as e:
                            print(f"Download failed for {path}: {e}")
                            continue
                    if not file_downloaded:
                        print(f"Failed to download {sandbox_filename} from any path")
                    pass

            # Extract model files
            models_dir = project_root / "static" / "models" / session_id
            models_dir.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Models dir set to: {models_dir}")

            model_extensions = [".pkl", ".joblib", ".h5", ".pt", ".pth", ".json", ".txt", ".cbm", ".onnx"]

            for line in stdout_content.split("\n"):
                if line.startswith("MODEL_SAVED:"):
                    sandbox_filename = line.split(":")[1].strip()

                    if sandbox_filename in processed_files:
                        print(f"Skipping duplicate model: {sandbox_filename}")
                        continue
                    processed_files.add(sandbox_filename)

                    local_path = models_dir / sandbox_filename
                    print(f"Attempting to download model {sandbox_filename} to {local_path}")

                    potential_paths = [
                        sandbox_filename,
                        f"/home/user/{sandbox_filename}",
                        f"/tmp/{sandbox_filename}",
                        f"./{sandbox_filename}",
                    ]

                    file_downloaded = False
                    for path in potential_paths:
                        try:
                            print(f"Trying to download model from sandbox path: {path}")
                            file_content_bytes = sandbox.files.read(path, format="bytes")
                            print(f"Downloaded {len(file_content_bytes) if file_content_bytes else 0} bytes")

                            if file_content_bytes and len(file_content_bytes) > 0:
                                with open(local_path, "wb") as f:
                                    f.write(file_content_bytes)
                                print(f"Successfully wrote model to {local_path}")

                                if local_path.exists() and local_path.stat().st_size > 0:
                                    web_url = f"/static/models/{session_id}/{sandbox_filename}"
                                    model_url = web_url
                                    file_downloaded = True
                                    logger.info(f"Model file verified: {sandbox_filename}")

                                    file_ext = Path(sandbox_filename).suffix
                                    model_id = None
                                    blob_path_var = None
                                    blob_url_var = None
                                    presigned_url_var = None

                                    try:
                                        src_path = str(project_root / "src")
                                        if src_path not in sys.path:
                                            sys.path.insert(0, src_path)

                                        from src.storage.cloud_storage import get_cloud_storage
                                        from src.storage.model_registry import ModelRegistryService
                                        from src.database.service import get_database_service
                                        from src.config import settings

                                        storage_config = settings.get("object_storage", {})
                                        if storage_config.get("enabled") and storage_config.get("upload_to_cloud"):
                                            bucket_name = storage_config.get("bucket_name", "datainsight-artifacts")
                                            folder_prefix = storage_config.get("folders", {}).get("models", "models")
                                            cloud_storage = get_cloud_storage(bucket_name=bucket_name)

                                            if cloud_storage:
                                                blob_path_var = f"{folder_prefix}/{session_id}/{sandbox_filename}"
                                                upload_result = cloud_storage.upload_file(
                                                    local_path=local_path,
                                                    blob_path=blob_path_var,
                                                    metadata={
                                                        "session_id": session_id,
                                                        "model_type": sandbox_filename.replace(file_ext, ""),
                                                        "environment": "cpu",
                                                    },
                                                )
                                                blob_url_var = upload_result["blob_url"]
                                                presigned_url_var = cloud_storage.get_blob_url(
                                                    blob_path=blob_path_var,
                                                    expires_in=storage_config.get("presigned_url_expiry_hours", 24)
                                                    * 3600,
                                                )
                                            model_url = presigned_url_var

                                            db_service = get_database_service()
                                            registry = ModelRegistryService(db_service)

                                            dataset_hash = "unknown"
                                            try:
                                                import builtins

                                                session_store = getattr(builtins, "_session_store", None)
                                                if session_store and session_id in session_store:
                                                    dataset_path = session_store[session_id].get("dataset_path")
                                                    if dataset_path and Path(dataset_path).exists():
                                                        dataset_hash = ModelRegistryService.compute_dataset_hash(
                                                            Path(dataset_path)
                                                        )
                                            except Exception:
                                                pass

                                            model_id = registry.register_model(
                                                session_id=session_id,
                                                dataset_hash=dataset_hash,
                                                model_type=sandbox_filename.replace(file_ext, ""),
                                                blob_path=upload_result["blob_path"],
                                                blob_url=upload_result["blob_url"],
                                                file_checksum=upload_result["checksum"],
                                                file_size_bytes=upload_result["size_bytes"],
                                                framework="scikit-learn",
                                                dependencies=["scikit-learn", "pandas", "numpy"],
                                            )
                                            logger.info(
                                                f"Model uploaded to cloud: {blob_path_var}, registry ID: {model_id}"
                                            )

                                            if not storage_config.get("keep_local_copy"):
                                                local_path.unlink()
                                    except Exception as cloud_error:
                                        logger.warning(f"Cloud upload failed: {cloud_error}, using local URL")

                                    model_urls.append(model_url)

                                    try:
                                        from src.api_utils.artifact_tracker import get_artifact_tracker

                                        tracker = get_artifact_tracker()
                                        tracker.add_artifact(
                                            session_id=session_id,
                                            filename=sandbox_filename,
                                            file_path=web_url,
                                            description=f"Trained model (CPU)",
                                            metadata={
                                                "type": "model",
                                                "format": file_ext,
                                                "environment": "cpu",
                                                "model_id": model_id,
                                            },
                                            blob_path=blob_path_var,
                                            blob_url=blob_url_var,
                                            presigned_url=presigned_url_var,
                                        )
                                        logger.info(f"Model artifact tracked: {sandbox_filename}")
                                    except Exception as e:
                                        logger.error(f"Model artifact tracking error: {e}")

                                    break
                                else:
                                    print(f"Model file verification failed for {local_path}")
                            else:
                                print(f"No valid content downloaded from {path}")
                        except Exception as e:
                            print(f"Model download failed for {path}: {e}")
                            continue

                    if not file_downloaded:
                        print(f"Failed to download model {sandbox_filename} from any path")

            # Extract processed dataset files
            datasets_dir = project_root / "static" / "datasets" / session_id
            datasets_dir.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Datasets dir set to: {datasets_dir}")

            for line in stdout_content.split("\n"):
                if line.startswith("DATASET_SAVED:"):
                    sandbox_filename = line.split(":", 1)[1].strip()

                    if sandbox_filename in processed_files:
                        print(f"Skipping duplicate dataset: {sandbox_filename}")
                        continue
                    processed_files.add(sandbox_filename)

                    local_path = datasets_dir / sandbox_filename
                    print(f"Attempting to download dataset {sandbox_filename} to {local_path}")

                    potential_paths = [
                        sandbox_filename,
                        f"/home/user/{sandbox_filename}",
                        f"/tmp/{sandbox_filename}",
                        f"./{sandbox_filename}",
                    ]

                    file_downloaded = False
                    for path in potential_paths:
                        try:
                            print(f"Trying to download dataset from sandbox path: {path}")
                            file_content_bytes = sandbox.files.read(path, format="bytes")
                            print(f"Downloaded {len(file_content_bytes) if file_content_bytes else 0} bytes")

                            if file_content_bytes and len(file_content_bytes) > 0:
                                with open(local_path, "wb") as f:
                                    f.write(file_content_bytes)
                                print(f"Successfully wrote dataset to {local_path}")

                                if local_path.exists() and local_path.stat().st_size > 0:
                                    web_url = f"/static/datasets/{session_id}/{sandbox_filename}"
                                    dataset_url = web_url
                                    file_downloaded = True
                                    logger.info(f"Dataset file verified: {sandbox_filename}")

                                    file_ext = Path(sandbox_filename).suffix
                                    blob_path_ds = None
                                    blob_url_ds = None
                                    presigned_url_ds = None

                                    try:
                                        src_path = str(project_root / "src")
                                        if src_path not in sys.path:
                                            sys.path.insert(0, src_path)

                                        from src.config import settings
                                        from src.storage.cloud_storage import get_cloud_storage

                                        storage_config = settings.get("object_storage", {})
                                        if storage_config.get("enabled") and storage_config.get("upload_to_cloud"):
                                            bucket_name = storage_config.get("bucket_name", "datainsight-artifacts")
                                            folder_prefix = storage_config.get("folders", {}).get(
                                                "datasets", "datasets"
                                            )
                                            cloud_storage = get_cloud_storage(bucket_name=bucket_name)

                                            if cloud_storage:
                                                blob_path_ds = f"{folder_prefix}/{session_id}/{sandbox_filename}"
                                                upload_result = cloud_storage.upload_file(
                                                    local_path=local_path,
                                                    blob_path=blob_path_ds,
                                                    metadata={"session_id": session_id, "type": "processed_dataset"},
                                                )
                                                blob_url_ds = upload_result["blob_url"]
                                                presigned_url_ds = cloud_storage.get_blob_url(
                                                    blob_path=blob_path_ds,
                                                    expires_in=storage_config.get("presigned_url_expiry_hours", 24)
                                                    * 3600,
                                                )
                                            dataset_url = presigned_url_ds
                                            logger.info(f"Uploaded dataset to cloud: {blob_path_ds}")

                                            if not storage_config.get("keep_local_copy"):
                                                local_path.unlink()
                                    except Exception as cloud_error:
                                        logger.warning(f"Cloud upload failed: {cloud_error}, using local URL")

                                    dataset_urls.append(dataset_url)

                                    try:
                                        from src.api_utils.artifact_tracker import get_artifact_tracker

                                        tracker = get_artifact_tracker()
                                        tracker.add_artifact(
                                            session_id=session_id,
                                            filename=sandbox_filename,
                                            file_path=web_url,
                                            description=f"Processed dataset",
                                            metadata={
                                                "type": "processed_dataset",
                                                "format": file_ext,
                                                "size_bytes": local_path.stat().st_size,
                                            },
                                            blob_path=blob_path_ds,
                                            blob_url=blob_url_ds,
                                            presigned_url=presigned_url_ds,
                                        )
                                        logger.info(f"Dataset artifact tracked: {sandbox_filename}")
                                    except Exception as e:
                                        logger.error(f"Dataset artifact tracking error: {e}")

                                    break
                                else:
                                    print(f"Dataset file verification failed for {local_path}")
                            else:
                                print(f"No valid content downloaded from {path}")
                        except Exception as e:
                            print(f"Dataset download failed for {path}: {e}")
                            continue

                    if not file_downloaded:
                        print(f"Failed to download dataset {sandbox_filename} from any path")

            # Filesystem fallback: Scan for orphaned dataset files without markers
            try:
                print(f"DEBUG: Scanning for orphaned dataset files...")
                downloaded_files = set([url.split("/")[-1] for url in dataset_urls])

                for ext in [".csv", ".parquet", ".xlsx"]:
                    try:
                        sandbox_files = sandbox.files.list(f"/tmp/*{ext}")
                        if sandbox_files:
                            for file_info in sandbox_files:
                                filename = file_info if isinstance(file_info, str) else file_info.get("name", "")
                                if filename and filename not in downloaded_files and filename != "dataset.parquet":
                                    print(f"DEBUG: Found orphaned dataset: {filename}")
                                    local_path = datasets_dir / filename
                                    try:
                                        file_content = sandbox.files.read(f"/tmp/{filename}", format="bytes")
                                        if file_content and len(file_content) > 0:
                                            with open(local_path, "wb") as f:
                                                f.write(file_content)

                                            if local_path.exists() and local_path.stat().st_size > 0:
                                                web_url = f"/static/datasets/{session_id}/{filename}"
                                                dataset_urls.append(web_url)
                                                print(f"DEBUG: Orphaned dataset downloaded: {filename}")

                                                try:
                                                    from src.api_utils.artifact_tracker import get_artifact_tracker

                                                    tracker = get_artifact_tracker()
                                                    tracker.add_artifact(
                                                        session_id=session_id,
                                                        filename=filename,
                                                        file_path=web_url,
                                                        description="Processed dataset (auto-detected)",
                                                        metadata={
                                                            "type": "processed_dataset",
                                                            "format": ext,
                                                            "auto_detected": True,
                                                        },
                                                    )
                                                except Exception as e:
                                                    print(f"DEBUG: Orphaned dataset tracking failed: {e}")
                                    except Exception as e:
                                        print(f"DEBUG: Failed to download orphaned dataset {filename}: {e}")
                    except Exception as e:
                        print(f"DEBUG: Filesystem scan for {ext} failed: {e}")
            except Exception as e:
                print(f"DEBUG: Filesystem fallback scanning failed: {e}")

        if isinstance(stdout_lines, str):
            clean_stdout = "\n".join(
                [
                    line
                    for line in stdout_lines.split("\n")
                    if not line.startswith("PLOT_SAVED:")
                    and not line.startswith("MODEL_SAVED:")
                    and not line.startswith("DATASET_SAVED:")
                ]
            )
        else:
            clean_stdout = "\n".join(
                [
                    line
                    for line in stdout_lines
                    if not line.startswith("PLOT_SAVED:")
                    and not line.startswith("MODEL_SAVED:")
                    and not line.startswith("DATASET_SAVED:")
                ]
            )
        try:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = process.cpu_percent()
            memory_usage = final_memory - initial_memory
            monitor.record_metric(
                deployment_id=session_id,
                metric_type=MetricType.MEMORY_USAGE,
                value=memory_usage,
                metadata={"code_length": len(code)},
            )
            if memory_usage > 500:  # MB threshold
                print(f"HIGH MEMORY USAGE: {memory_usage:.2f} MB")
        except Exception as monitor_error:
            print(f"Resource monitoring failed: {monitor_error}")

        has_python_traceback = "Traceback (most recent call last):" in stderr_content and any(
            line.strip().startswith("File ") or line.strip().startswith("  File ")
            for line in stderr_content.split("\n")
        )

        has_explicit_error = (
            "Execution error:" in clean_stdout and ":" in clean_stdout.split("Execution error:")[-1][:100]
        )

        execution_success = not (
            has_python_traceback or has_explicit_error or (hasattr(result, "error") and result.error)
        )

        return {
            "success": execution_success,
            "stdout": clean_stdout,
            "stderr": stderr_content,
            "plots": plot_urls,
            "models": model_urls,
            "datasets": dataset_urls,
            "files": [],
        }
    except Exception as e:
        print(f"DEBUG: Exception in execute_python_in_sandbox: {e}")
        print(f"DEBUG: Exception type: {type(e)}")
        import traceback

        print(f"DEBUG: Traceback: {traceback.format_exc()}")

        error_str = str(e)
        if "disconnected" in error_str.lower() or "timeout" in error_str.lower():
            try:
                if session_id in session_sandboxes:
                    session_sandboxes[session_id].close()
                    del session_sandboxes[session_id]

                new_sandbox = get_sandbox(session_id)
                result = new_sandbox.run_code(code, timeout=timeout)

                performance_monitor.record_metric(
                    session_id=session_id,
                    metric_name="sandbox_recovery_success",
                    value=1.0,
                    context={"original_error": error_str},
                )

                retry_clean_stdout = ""
                try:
                    if hasattr(result, "logs") and hasattr(result.logs, "stdout"):
                        if result.logs.stdout:
                            retry_clean_stdout = "\n".join(result.logs.stdout)
                    elif hasattr(result, "stdout"):
                        retry_clean_stdout = result.stdout
                    else:
                        result_str = str(result)
                        if "stdout: [" in result_str:
                            import re

                            stdout_match = re.search(r"stdout: \[(.*?)\]", result_str, re.DOTALL)
                            if stdout_match:
                                stdout_content = stdout_match.group(1)
                                stdout_lines = re.findall(r'"([^"]*)"', stdout_content)
                                retry_clean_stdout = "\n".join(stdout_lines).replace("\\n", "\n")
                except:
                    retry_clean_stdout = str(result)

                return {
                    "success": True,
                    "stdout": retry_clean_stdout,
                    "stderr": "",
                    "plots": [],
                    "models": [],
                    "datasets": [],
                    "files": [],
                }
            except Exception as retry_e:
                error_str = f"Sandbox connection failed. Original: {error_str}, Retry: {str(retry_e)}"

        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="sandbox_failure",
            value=1.0,
            context={"error": error_str, "code_length": len(code)},
        )
        return {
            "success": False,
            "stdout": "",
            "stderr": error_str,
            "plots": [],
            "models": [],
            "datasets": [],
            "files": [],
        }


def close_sandbox_session(session_id: str):
    """Closes and cleans up a session's sandbox."""
    if session_id in session_sandboxes:
        session_sandboxes[session_id].close()
        del session_sandboxes[session_id]


def azure_gpu_train(code: str, session_id: str, user_format: str = None) -> str:
    """
    Train models on Azure GPU clusters with smart format detection.
    Submits Python training code to Azure ML compute cluster with intelligent model saving.

    Args:
        code: Python training code to execute on the GPU cluster
        session_id: Session identifier for tracking the training run
        user_format: Optional user-specified save format (e.g., "onnx", "joblib")
    """
    try:
        from azureml.core import Workspace, Experiment, ScriptRunConfig
        from azure.storage.blob import BlobServiceClient

        ws = Workspace.from_config()

        wrapper_path = os.path.join(os.path.dirname(__file__), "core", "gpu_wrapper.py")
        with open(wrapper_path, "r") as f:
            wrapper_code = f.read()

        wrapped_script = f"""
{wrapper_code}

# User training code
user_code = '''
{code}
'''

# Execute with train wrapper
result = train_wrapper(
    user_code=user_code,
    output_dir='/opt/ml/model',
    user_format={repr(user_format)}
)

print(f"Training complete: {{result}}")
"""

        script_path = f"/tmp/azure_train_{uuid.uuid4()}.py"
        with open(script_path, "w") as f:
            f.write(wrapped_script)

        exp = Experiment(workspace=ws, name=f"train_{session_id}")
        config = ScriptRunConfig(
            source_directory=".", script=script_path, compute_target="gpu_cluster", environment="pytorch-env"
        )

        run = exp.submit(config)
        run.wait_for_completion(show_output=True)

        if run.get_status() == "Completed":
            metadata_path = f"/tmp/metadata_{session_id}.json"
            run.download_file("outputs/metadata.json", metadata_path)

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            model_format = metadata.get("format", ".pkl")
            model_filename = f"model{model_format}"
            local_model_path = f"/tmp/{session_id}_{model_filename}"

            run.download_file(f"outputs/{model_filename}", local_model_path)

            import shutil

            static_models_dir = project_root / "static" / "models" / session_id
            static_models_dir.mkdir(parents=True, exist_ok=True)
            static_model_path = static_models_dir / model_filename
            shutil.copy(local_model_path, static_model_path)

            blob_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONN_STR"))
            blob = blob_client.get_blob_client(container="models", blob=f"{session_id}/{model_filename}")

            with open(local_model_path, "rb") as f:
                blob.upload_blob(f, overwrite=True)

            try:
                import sys

                src_path = str(project_root / "src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from src.api_utils.artifact_tracker import get_artifact_tracker

                tracker = get_artifact_tracker()
                web_url = f"/static/models/{session_id}/{model_filename}"
                tracker.add_artifact(
                    session_id=session_id,
                    filename=model_filename,
                    file_path=web_url,
                    description=f"Trained model (Azure GPU)",
                    metadata={"type": "model", "format": model_format, "environment": "gpu_azure"},
                )
                logger.info(f"Model artifact tracked: {model_filename}")
            except Exception as e:
                logger.error(f"Model artifact tracking error: {e}")

            return f"Model trained successfully: {blob.url} (format: {model_format}, local: {static_model_path})"
        else:
            error_details = run.get_details().get("error", "Unknown error")
            return f"Training failed: {error_details}"

    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


@tool
def aws_gpu_train(code: str, session_id: str, user_format: str = None) -> str:
    """
    Train models on AWS SageMaker GPU with smart format detection.
    Submits Python training code to SageMaker processing job with intelligent model saving.

    Args:
        code: Python training code to execute on SageMaker
        session_id: Session identifier for tracking the training run
        user_format: Optional user-specified save format (e.g., "onnx", "joblib")
    """
    try:
        import boto3
        from sagemaker import Session
        from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

        s3_bucket = os.getenv("S3_BUCKET")
        sagemaker_role = os.getenv("SAGEMAKER_ROLE")
        s3 = boto3.client("s3")
        session = Session()

        # Read gpu_wrapper.py for bundling
        wrapper_path = os.path.join(os.path.dirname(__file__), "core", "gpu_wrapper.py")
        with open(wrapper_path, "r") as f:
            wrapper_code = f.read()

        # Create wrapper script
        wrapped_script = f"""
                        {wrapper_code}

                        # User training code
                        user_code = '''
                        {code}
                        '''

                        # Execute with train wrapper
                        result = train_wrapper(
                            user_code=user_code,
                            output_dir='/opt/ml/model',
                            user_format={repr(user_format)}
                        )

                        print(f"Training complete: {{result}}")
                        """

        # Upload script to S3
        script_key = f"scripts/train_{session_id}_{uuid.uuid4()}.py"
        s3.put_object(Bucket=s3_bucket, Key=script_key, Body=wrapped_script.encode("utf-8"))

        # Configure SageMaker processor
        processor = ScriptProcessor(
            command=["python3"],
            image_uri=os.getenv(
                "SAGEMAKER_IMAGE_URI", "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310"
            ),
            role=sagemaker_role,
            instance_count=1,
            instance_type="ml.g4dn.xlarge",
            sagemaker_session=session,
            volume_size_in_gb=30,
        )

        # Run training job
        processor.run(
            code=f"s3://{s3_bucket}/{script_key}",
            outputs=[
                ProcessingOutput(
                    output_name="model", source="/opt/ml/model", destination=f"s3://{s3_bucket}/models/{session_id}"
                )
            ],
        )

        # Download metadata to find actual model format
        metadata_key = f"models/{session_id}/metadata.json"
        metadata_local = f"/tmp/metadata_{session_id}.json"
        s3.download_file(s3_bucket, metadata_key, metadata_local)

        with open(metadata_local, "r") as f:
            metadata = json.load(f)

        model_format = metadata.get("format", ".pkl")
        model_filename = f"model{model_format}"

        # Download model with detected format
        model_key = f"models/{session_id}/{model_filename}"
        local_model_path = f"/tmp/{session_id}_{model_filename}"
        s3.download_file(s3_bucket, model_key, local_model_path)

        # Move to static/models for artifact system
        import shutil

        static_models_dir = Path(__file__).resolve().parent.parent.parent / "static" / "models" / session_id
        static_models_dir.mkdir(parents=True, exist_ok=True)
        static_model_path = static_models_dir / model_filename
        shutil.copy(local_model_path, static_model_path)

        # Track in artifact system
        try:
            import sys

            project_root = Path(__file__).resolve().parent.parent.parent
            src_path = str(project_root / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            from src.api_utils.artifact_tracker import get_artifact_tracker

            tracker = get_artifact_tracker()
            web_url = f"/static/models/{session_id}/{model_filename}"
            tracker.add_artifact(
                session_id=session_id,
                filename=model_filename,
                file_path=web_url,
                description=f"Trained model (AWS GPU)",
                metadata={"type": "model", "format": model_format, "environment": "gpu_aws"},
            )
            logger.info(f"Model artifact tracked: {model_filename}")
        except Exception as e:
            logger.error(f"Model artifact tracking error: {e}")

        s3_url = f"s3://{s3_bucket}/{model_key}"
        return f"Model trained successfully: {static_model_path} (S3: {s3_url}, format: {model_format})"

    except Exception as e:
        return f"Error: {str(e)}"
