import os
from typing import Dict, Any
from pathlib import Path
from e2b_code_interpreter import Sandbox
from .performance_monitor import PerformanceMonitor

session_sandboxes = {}
performance_monitor = PerformanceMonitor()

def get_sandbox(session_id: str) -> Sandbox:
    if session_id not in session_sandboxes:
        sandbox = Sandbox.create()
        session_sandboxes[session_id] = sandbox
    else:
        try:
            session_sandboxes[session_id].run_code("print('test')", timeout=5)
        except:
            print(f"DEBUG: Recreating expired sandbox for session {session_id}")
            sandbox = Sandbox.create()
            session_sandboxes[session_id] = sandbox
    return session_sandboxes[session_id]

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
    try:
        plot_save_code = ""
        if any(keyword in code for keyword in ["plt.show()", "fig.show()", "sns.heatmap()", "plt.figure", "df.plot"]):
            plot_save_code = f"""
        import matplotlib.pyplot as plt
        import os
        import uuid
        plt.close('all')

        if plt.get_fignums():
            fig = plt.gcf()
            plot_filename = f"plot_{session_id}_{{uuid.uuid4().hex[:8]}}.png"
            fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
            print(f"PLOT_SAVED:{{plot_filename}}") # Use a clear separator
            plt.close(fig)
        """
        if 'df.corr()' in code:
            code = code.replace('df.corr()', 'df.select_dtypes(include=[\'number\']).corr()')
        
        enhanced_code = code + plot_save_code
        
        if '.head(' in code or '.tail(' in code or 'print(df' in code:
            enhanced_code = enhanced_code + '\n\n# Enhanced dataframe display\ntry:\n    if "df" in locals() and hasattr(df, "shape"):\n        print(f"\\n📊 DataFrame Info: {df.shape[0]} rows × {df.shape[1]} columns")\nexcept: pass'
        
        result = sandbox.run_code(enhanced_code, timeout=30)

        performance_monitor.record_metric(
            session_id=session_id,
            metric_name="sandbox_success",
            value=1.0,
            context={"code_length": len(code)}
        )
        plot_files = []
        other_files = []
        try:
            files_result = sandbox.run_code("import os; print('\\n'.join(os.listdir('.')))", timeout=5)
            files_list = []
            if files_result:
                if hasattr(files_result, 'logs') and hasattr(files_result.logs, 'stdout'):
                    stdout_content = '\n'.join(files_result.logs.stdout) if files_result.logs.stdout else ""
                    files_list = stdout_content.strip().split('\n') if stdout_content else []
                elif hasattr(files_result, 'stdout'):
                    files_list = files_result.stdout.strip().split('\n') if files_result.stdout else []
            
            for file in files_list:
                if file and file.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf')):
                    plot_files.append(file)
                elif file and not file.endswith(('.py', '.pyc')):
                    other_files.append(file)
        except Exception as e:
            pass

        plot_urls = []
        if plot_files:
            import os
            from pathlib import Path
            
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            static_dir = project_root / "static" / "plots"
            static_dir.mkdir(parents=True, exist_ok=True)
            
            for plot_file in plot_files:
                try:
                    file_content = None
                    
                    try:
                        b64_result = sandbox.run_code(f"""
                        import base64
                        with open('{plot_file}', 'rb') as f:
                            content = f.read()
                            b64_content = base64.b64encode(content).decode('ascii')
                            print(b64_content)
                        """, timeout=10)
                        if b64_result and hasattr(b64_result, 'logs') and b64_result.logs.stdout:
                            import base64
                            b64_string = '\n'.join(b64_result.logs.stdout).strip()
                            file_content = base64.b64decode(b64_string)
                        else:
                            continue
                    except Exception as e:
                        continue
                    
                    if file_content:
                        local_path = static_dir / plot_file
                        
                        try:
                            with open(local_path, 'wb') as f:
                                f.write(file_content)
                            
                            if os.path.exists(local_path):
                                plot_url = f"/static/plots/{plot_file}"
                                plot_urls.append(plot_url)
                        except Exception as write_err:
                            pass
                        
                except Exception as e:
                    pass

        clean_stdout = ""
        clean_stderr = ""
        
        try:
            if hasattr(result, 'logs') and hasattr(result.logs, 'stdout'):
                if result.logs.stdout:
                    clean_stdout = '\n'.join(result.logs.stdout)
            elif hasattr(result, 'stdout'):
                clean_stdout = result.stdout
            else:
                result_str = str(result)
                if 'stdout: [' in result_str:
                    import re
                    stdout_match = re.search(r'stdout: \[(.*?)\]', result_str, re.DOTALL)
                    if stdout_match:
                        stdout_content = stdout_match.group(1)
                        stdout_lines = re.findall(r'"([^"]*)"', stdout_content)
                        clean_stdout = '\n'.join(stdout_lines).replace('\\n', '\n')
                        
            if hasattr(result, 'logs') and hasattr(result.logs, 'stderr'):
                if result.logs.stderr:
                    clean_stderr = '\n'.join(result.logs.stderr)
            elif hasattr(result, 'stderr'):
                clean_stderr = result.stderr
                
        except Exception as parse_error:
            clean_stdout = str(result)
        return {
            "success": True,
            "stdout": clean_stdout,
            "stderr": clean_stderr,
            "plots": plot_urls,
            "files": other_files
        }
    except Exception as e:
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

# Backward compatibility
def python_repl(code: str) -> Dict[str, Any]:
    """Legacy function - use execute_python_in_sandbox directly"""
    return {"success": False, "stderr": "Use execute_python_in_sandbox with session_id"}