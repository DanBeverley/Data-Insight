import os
import sys
import uuid
import json
import logging
from typing import Dict, Any, TYPE_CHECKING
from pathlib import Path
from dotenv import load_dotenv

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
        sandbox = Sandbox.create(timeout=300)
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
    """Attempt to reload dataset into sandbox if available in session store"""
    try:
        import builtins

        session_store = getattr(builtins, "_session_store", None)
        if session_store and session_id in session_store and "dataframe" in session_store[session_id]:
            df = session_store[session_id]["dataframe"]
            csv_data = df.to_csv(index=False)
            reload_code = f"""
                            import pandas as pd
                            import numpy as np
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            from io import StringIO

                            # Reload dataset from session
                            csv_data = '''{csv_data}'''
                            df = pd.read_csv(StringIO(csv_data))
                            print(f"Dataset reloaded: {{df.shape}} shape, {{len(df.columns)}} columns")
                            """
            result = sandbox.run_code(reload_code, timeout=10)
    except Exception as e:
        print(f"DEBUG: Could not reload dataset for session {session_id}: {e}")


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
        dataset_load_code = ""

        if session_store and session_id in session_store and "dataframe" in session_store[session_id]:
            df = session_store[session_id]["dataframe"]
            csv_data = df.to_csv(index=False)
            dataset_load_code = f"""
                                from io import StringIO
                                csv_data = '''{csv_data}'''
                                df = pd.read_csv(StringIO(csv_data))
                                """
        is_plotting = any(pattern in code for pattern in ["plt.", "sns.", ".plot(", ".hist(", "matplotlib", "seaborn"])
        patterns_found = [p for p in ["plt.", "sns.", ".plot(", ".hist(", "matplotlib", "seaborn"] if p in code]
        print(f"DEBUG: Plot detection - code contains: {patterns_found}")
        if "df.corr()" in code:
            code = code.replace("df.corr()", "df.select_dtypes(include=[np.number]).corr()")

        indented_code = "\n".join("    " + line if line.strip() else "" for line in code.split("\n"))
        indented_dataset_load = "\n".join(line.strip() for line in dataset_load_code.split("\n") if line.strip())

        enhanced_code = f"""import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import os
import glob

def use_noop(*args, **kwargs):
    pass

matplotlib.use = use_noop
plt.use = use_noop

{indented_dataset_load}

before_pngs = set(glob.glob('*.png') + glob.glob('/tmp/*.png'))

# Track model files before execution
model_exts = ['*.pkl', '*.joblib', '*.h5', '*.pt', '*.pth', '*.json', '*.txt', '*.cbm', '*.onnx']
before_models = set()
for ext in model_exts:
    before_models.update(glob.glob(ext))
    before_models.update(glob.glob(f'/tmp/{{ext}}'))

try:
{indented_code}
except Exception as e:
    print(f"Execution error: {{type(e).__name__}}: {{e}}")
    import traceback
    traceback.print_exc()

after_pngs = set(glob.glob('*.png') + glob.glob('/tmp/*.png'))
new_pngs = after_pngs - before_pngs
processed = set()

for fig_num in plt.get_fignums():
    import uuid
    filename = f"plot_{{uuid.uuid4().hex[:8]}}.png"
    plt.figure(fig_num)
    plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig_num)
    if os.path.exists(filename):
        print(f"PLOT_SAVED:{{filename}}")
        processed.add(filename)

for png in new_pngs:
    if png not in processed and os.path.exists(png) and os.path.getsize(png) > 100:
        print(f"PLOT_SAVED:{{os.path.basename(png)}}")

# Detect new model files
after_models = set()
for ext in model_exts:
    after_models.update(glob.glob(ext))
    after_models.update(glob.glob(f'/tmp/{{ext}}'))

new_models = after_models - before_models
for model_file in new_models:
    if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
        print(f"MODEL_SAVED:{{os.path.basename(model_file)}}")
"""
        result = sandbox.run_code(enhanced_code, timeout=30)

        if hasattr(result, "error") and result.error:
            error_msg = f"{result.error.name}: {result.error.value}"
            print(f"ERROR: E2B execution failed: {error_msg}")
            performance_monitor.record_metric(
                session_id=session_id, metric_name="sandbox_error", value=1.0, context={"error": error_msg}
            )
            return {"success": False, "stderr": error_msg, "stdout": "", "plots": [], "models": []}

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

        from pathlib import Path

        current_path = Path(__file__).resolve()
        project_root = current_path
        while project_root.name != "Data-Insight" and project_root.parent != project_root:
            project_root = project_root.parent
        static_dir = project_root / "static" / "plots"
        static_dir.mkdir(parents=True, exist_ok=True)
        if stdout_content:
            import os
            from pathlib import Path

            current_path = Path(__file__).resolve()
            project_root = current_path
            while project_root.name != "Data-Insight" and project_root.parent != project_root:
                project_root = project_root.parent
            static_dir = project_root / "static" / "plots"
            static_dir.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Static dir set to: {static_dir}")
            for line in stdout_content.split("\n"):
                if line.startswith("PLOT_SAVED:"):
                    sandbox_filename = line.split(":")[1].strip()
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
                                        plot_urls.append(web_url)
                                        file_downloaded = True
                                        logger.info(f"File verified and URL added: {web_url}")
                                        try:
                                            import sys

                                            src_path = str(project_root / "src")
                                            if src_path not in sys.path:
                                                sys.path.insert(0, src_path)
                                            from src.api_utils.artifact_tracker import get_artifact_tracker

                                            tracker = get_artifact_tracker()
                                            tracker.add_artifact(
                                                session_id=session_id,
                                                filename=sandbox_filename,
                                                file_path=web_url,
                                                description="Generated visualization",
                                                metadata={"type": "plot", "format": "png"},
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
                                    model_urls.append(web_url)
                                    file_downloaded = True
                                    logger.info(f"Model file verified and URL added: {web_url}")

                                    try:
                                        src_path = str(project_root / "src")
                                        if src_path not in sys.path:
                                            sys.path.insert(0, src_path)
                                        from src.api_utils.artifact_tracker import get_artifact_tracker

                                        tracker = get_artifact_tracker()

                                        # Determine model format from extension
                                        file_ext = Path(sandbox_filename).suffix

                                        # Upload model to blob storage and register
                                        model_id = None
                                        try:
                                            from src.storage.blob_service import BlobStorageService
                                            from src.storage.model_registry import ModelRegistryService
                                            from src.database.service import get_database_service
                                            from src.config import settings

                                            storage_config = settings.get("object_storage", {})
                                            if storage_config.get("enabled", False):
                                                blob_service = BlobStorageService(
                                                    container_name=storage_config.get(
                                                        "container_name", "datainsight-models"
                                                    )
                                                )

                                                # Upload to blob storage
                                                blob_path = f"{session_id}/models/{sandbox_filename}"
                                                upload_result = blob_service.upload_file(
                                                    local_path=local_path,
                                                    blob_path=blob_path,
                                                    metadata={
                                                        "session_id": session_id,
                                                        "model_type": sandbox_filename.replace(file_ext, ""),
                                                        "environment": "cpu",
                                                    },
                                                )

                                                # Register in model registry
                                                db_service = get_database_service()
                                                registry = ModelRegistryService(db_service)

                                                # Compute dataset hash if available
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

                                                print(f"Model uploaded to blob storage and registered: {model_id}")
                                        except Exception as blob_error:
                                            print(
                                                f"Blob storage upload failed (continuing with local storage): {blob_error}"
                                            )

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

        if isinstance(stdout_lines, str):
            clean_stdout = "\n".join(
                [
                    line
                    for line in stdout_lines.split("\n")
                    if not line.startswith("PLOT_SAVED:") and not line.startswith("MODEL_SAVED:")
                ]
            )
        else:
            clean_stdout = "\n".join(
                [
                    line
                    for line in stdout_lines
                    if not line.startswith("PLOT_SAVED:") and not line.startswith("MODEL_SAVED:")
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
        return {
            "success": True,
            "stdout": clean_stdout,
            "stderr": stderr_content,
            "plots": plot_urls,
            "models": model_urls,
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
                result = new_sandbox.run_code(code, timeout=30)

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
        return {"success": False, "stdout": "", "stderr": error_str, "plots": [], "models": [], "files": []}


def close_sandbox_session(session_id: str):
    """Closes and cleans up a session's sandbox."""
    if session_id in session_sandboxes:
        session_sandboxes[session_id].close()
        del session_sandboxes[session_id]


@tool
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

        # Read gpu_wrapper.py for bundling
        wrapper_path = os.path.join(os.path.dirname(__file__), "core", "gpu_wrapper.py")
        with open(wrapper_path, "r") as f:
            wrapper_code = f.read()

        # Create wrapper script that calls train_wrapper
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
            # Download metadata to find actual model format
            metadata_path = f"/tmp/metadata_{session_id}.json"
            run.download_file("outputs/metadata.json", metadata_path)

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            model_format = metadata.get("format", ".pkl")
            model_filename = f"model{model_format}"
            local_model_path = f"/tmp/{session_id}_{model_filename}"

            # Download model with detected format
            run.download_file(f"outputs/{model_filename}", local_model_path)

            # Move to static/models for artifact system
            import shutil

            static_models_dir = project_root / "static" / "models" / session_id
            static_models_dir.mkdir(parents=True, exist_ok=True)
            static_model_path = static_models_dir / model_filename
            shutil.copy(local_model_path, static_model_path)

            # Upload to Azure Blob Storage
            blob_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONN_STR"))
            blob = blob_client.get_blob_client(container="models", blob=f"{session_id}/{model_filename}")

            with open(local_model_path, "rb") as f:
                blob.upload_blob(f, overwrite=True)

            # Track in artifact system
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
