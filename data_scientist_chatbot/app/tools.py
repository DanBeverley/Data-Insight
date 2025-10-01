import os
import sys
import uuid
from typing import Dict, Any
from pathlib import Path
from e2b_code_interpreter import Sandbox
from dotenv import load_dotenv

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
if not hasattr(builtins, '_persistent_sandboxes'):
    builtins._persistent_sandboxes = {}
    print(f"DEBUG: Initialized persistent sandbox storage")
session_sandboxes = builtins._persistent_sandboxes
print(f"DEBUG: tools.py module loaded, using persistent sandboxes: {id(session_sandboxes)}")
performance_monitor = PerformanceMonitor()

def get_sandbox(session_id: str) -> Sandbox:
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

def _reload_dataset_if_available(sandbox: Sandbox, session_id: str):
    """Attempt to reload dataset into sandbox if available in session store"""
    try:
        import builtins
        session_store = getattr(builtins, '_session_store', None)
        
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
            print(f"DEBUG: Dataset reload {'successful' if result else 'failed'} for session {session_id}")
        else:
            print(f"DEBUG: No session data found for {session_id}, skipping reload")
    except Exception as e:
        print(f"DEBUG: Could not reload dataset for session {session_id}: {e}")

@performance_monitor.cache_result(ttl=600, key_prefix="sandbox_exec")
@performance_monitor.time_function("sandbox", "code_execution")
def execute_python_in_sandbox(code: str, session_id: str) -> Dict[str, Any]:
    """
    Executes Python code in a stateful, secure sandbox for a specific session.
    Enhanced with performance monitoring and intelligent caching.
    """
    if not session_id:
        performance_monitor.record_metric(
            session_id="unknown",
            metric_name="sandbox_error",
            value=1.0,
            context={"error": "missing_session_id"}
        )
        return {"success": False, "stderr": "Session ID is missing."}

    sandbox = get_sandbox(session_id)
    plot_urls = []

    try:
        print("DEBUG: Starting sandbox execution try block")
        import psutil
        import sys
        import os  
        print("DEBUG: Imported os successfully")
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        print("DEBUG: Added src to path")
        from src.mlops.monitoring import PerformanceMonitor, MetricType

        monitor = PerformanceMonitor()

        # Record initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()

        is_plotting = any(pattern in code for pattern in [
            'plt.', 'sns.', '.plot(', '.hist(', 'matplotlib', 'seaborn'])
        patterns_found = [p for p in ["plt.", "sns.", ".plot(", ".hist(", "matplotlib", "seaborn"] if p in code]
        print(f"DEBUG: Plot detection - code contains: {patterns_found}")
        if 'df.corr()' in code:
            code = code.replace('df.corr()', 'df.select_dtypes(include=[np.number]).corr()')
        
        if is_plotting:
            import re
            code = re.sub(r'plt\.show\(\)', '', code)
            code = re.sub(r'plt\.savefig\([^)]+\)', '', code)
            code = re.sub(r'matplotlib\.use\([^)]+\)', '', code)
            enhanced_code = f"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import uuid
import sys
import os
import glob

print("DEBUG_WRAPPER: Starting enhanced code execution")
sys.stdout.flush()

try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    print(f"DEBUG_WRAPPER: User code error: {{type(e).__name__}}: {{e}}")
    import traceback
    print("DEBUG_WRAPPER: Traceback:")
    traceback.print_exc()
    sys.stdout.flush()

print("DEBUG_WRAPPER: User code completed, checking for figures...")
sys.stdout.flush()
fig_nums = plt.get_fignums()
print(f"DEBUG_WRAPPER: Found {{len(fig_nums)}} matplotlib figures: {{fig_nums}}")
sys.stdout.flush()

if fig_nums:
    print(f"DEBUG_WRAPPER: Processing {{len(fig_nums)}} figures")
    sys.stdout.flush()
    for fig_num in fig_nums:
        try:
            plt.figure(fig_num)
            plot_filename = f"plot_{{uuid.uuid4().hex[:8]}}.png"
            print(f"DEBUG_WRAPPER: Saving figure {{fig_num}} to {{plot_filename}}")
            sys.stdout.flush()

            plt.savefig(plot_filename, format='png', dpi=150, bbox_inches='tight')

            if os.path.exists(plot_filename):
                file_size = os.path.getsize(plot_filename)
                print(f"DEBUG_WRAPPER: File created, size {{file_size}} bytes")
                print(f"PLOT_SAVED:{{plot_filename}}")
                sys.stdout.flush()
            else:
                print(f"DEBUG_WRAPPER: ERROR - File not created after savefig")
                sys.stdout.flush()

            plt.close(fig_num)
        except Exception as e:
            print(f"DEBUG_WRAPPER: Exception saving figure {{fig_num}}: {{e}}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
else:
    print("DEBUG_WRAPPER: No matplotlib figures, checking for orphaned PNGs")
    sys.stdout.flush()
    all_pngs = glob.glob('*.png') + glob.glob('/tmp/*.png')
    if all_pngs:
        print(f"DEBUG_WRAPPER: Found PNG files: {{all_pngs}}")
        sys.stdout.flush()
        for png_path in all_pngs:
            if os.path.getsize(png_path) > 100:
                print(f"PLOT_SAVED:{{os.path.basename(png_path)}}")
                sys.stdout.flush()
    else:
        print("DEBUG_WRAPPER: No figures or PNG files found")
        sys.stdout.flush()
"""
        else:
            # For non-plotting code, execute directly
            enhanced_code = f"""
import pandas as pd
import numpy as np

# Execute the user code directly
{code}
"""
        
        print(f"DEBUG: About to run code in sandbox. Enhanced code length: {len(enhanced_code)}")
        print(f"DEBUG: First 500 chars of enhanced code: {enhanced_code[:500]}")
        result = sandbox.run_code(enhanced_code, timeout=30)
        print("DEBUG: Sandbox execution completed")
        print(f"DEBUG: Result type: {type(result)}")
        print(f"DEBUG: Result has logs: {hasattr(result, 'logs')}")
        
        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="sandbox_success",
            value=1.0,
            context={"code_length": len(code)}
        )
        
        stdout_lines = result.logs.stdout if hasattr(result, 'logs') and hasattr(result.logs, 'stdout') else []
        stderr_lines = result.logs.stderr if hasattr(result, 'logs') and hasattr(result.logs, 'stderr') else []

        stdout_content = '\n'.join(stdout_lines)
        stderr_content = '\n'.join(stderr_lines)

        print(f"DEBUG: stdout_lines count: {len(stdout_lines)}")
        print(f"DEBUG: Full stdout_content: {stdout_content}")
        print(f"DEBUG: Full stderr_content: {stderr_content if stderr_content else 'EMPTY'}")

        from pathlib import Path
        current_path = Path(__file__).resolve()
        project_root = current_path
        while project_root.name != "Data-Insight" and project_root.parent != project_root:
            project_root = project_root.parent
        static_dir = project_root / "static" / "plots"
        static_dir.mkdir(parents=True, exist_ok=True)

        print(f"DEBUG: Checking if stdout_content exists: {bool(stdout_content)}")
        if stdout_content:
            print("DEBUG: Processing stdout content for plot downloads")
            import os
            from pathlib import Path

            current_path = Path(__file__).resolve()
            project_root = current_path
            while project_root.name != "Data-Insight" and project_root.parent != project_root:
                project_root = project_root.parent

            static_dir = project_root / "static" / "plots"
            static_dir.mkdir(parents=True, exist_ok=True)
            print(f"DEBUG: Static dir set to: {static_dir}")
            
            for line in stdout_content.split('\n'):
                if line.startswith("PLOT_SAVED:"):
                    sandbox_filename = line.split(":")[1].strip()
                    local_path = static_dir / sandbox_filename
                    print(f"Attempting to download {sandbox_filename} to {local_path}")
                    
                    potential_paths = [
                        sandbox_filename, 
                        f"/home/user/{sandbox_filename}",  
                        f"/tmp/{sandbox_filename}",  
                        f"./{sandbox_filename}" 
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
                                if file_content_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                                    with open(local_path, "wb") as f:
                                        f.write(file_content_bytes)
                                    print(f"Successfully wrote to {local_path}")
                                    
                                    # Verify the saved file
                                    if local_path.exists() and local_path.stat().st_size > 0:
                                        plot_urls.append(f"/static/plots/{sandbox_filename}")
                                        file_downloaded = True
                                        print(f"File verified and URL added: /static/plots/{sandbox_filename}")
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

        clean_stdout = "\n".join([line for line in stdout_lines if not line.startswith("PLOT_SAVED:")])

        # Record final resource usage
        try:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = process.cpu_percent()
            memory_usage = final_memory - initial_memory

            monitor.record_metric(
                deployment_id=session_id,
                metric_type=MetricType.MEMORY_USAGE,
                value=memory_usage,
                metadata={"code_length": len(code)}
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
            "files": []
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
                    context={"original_error": error_str}
                )
                
                retry_clean_stdout = ""
                try:
                    if hasattr(result, 'logs') and hasattr(result.logs, 'stdout'):
                        if result.logs.stdout:
                            retry_clean_stdout = '\n'.join(result.logs.stdout)
                    elif hasattr(result, 'stdout'):
                        retry_clean_stdout = result.stdout
                    else:
                        result_str = str(result)
                        if 'stdout: [' in result_str:
                            import re
                            stdout_match = re.search(r'stdout: \[(.*?)\]', result_str, re.DOTALL)
                            if stdout_match:
                                stdout_content = stdout_match.group(1)
                                stdout_lines = re.findall(r'"([^"]*)"', stdout_content)
                                retry_clean_stdout = '\n'.join(stdout_lines).replace('\\n', '\n')
                except:
                    retry_clean_stdout = str(result)
                
                return {
                    "success": True,
                    "stdout": retry_clean_stdout,
                    "stderr": "",
                    "plots": [],
                    "files": []
                }
            except Exception as retry_e:
                error_str = f"Sandbox connection failed. Original: {error_str}, Retry: {str(retry_e)}"
        
        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="sandbox_failure",
            value=1.0,
            context={"error": error_str, "code_length": len(code)}
        )
        return {
            "success": False,
            "stdout": "",
            "stderr": error_str,
            "plots": [],
            "files": []
        }

def close_sandbox_session(session_id: str):
    """Closes and cleans up a session's sandbox."""
    if session_id in session_sandboxes:
        session_sandboxes[session_id].close()
        del session_sandboxes[session_id]

def python_repl(code: str) -> Dict[str, Any]:
    """Legacy function - use execute_python_in_sandbox directly"""
    return {"success": False, "stderr": "Use execute_python_in_sandbox with session_id"}