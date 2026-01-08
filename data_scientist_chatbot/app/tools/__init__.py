"""Tool execution and parsing components"""

import sys
import os
import importlib.util

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tools_file_path = os.path.join(parent_dir, "tools.py")

if os.path.exists(tools_file_path):
    spec = importlib.util.spec_from_file_location("tools_module", tools_file_path)
    tools_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools_module)
    execute_python_in_sandbox = tools_module.execute_python_in_sandbox

    # execute_tool doesn't exist in tools.py, create a stub
    if hasattr(tools_module, "execute_tool"):
        execute_tool = tools_module.execute_tool
    else:

        def execute_tool(tool_name, tool_args, session_id):
            """
            Execute a tool by name with arguments for a specific session.
            """
            # Special handling for zip_artifacts to run as python code in sandbox
            if tool_name == "zip_artifacts":
                artifact_ids = tool_args.get("artifact_ids", [])
                description = tool_args.get("description", "artifacts")
                code = f"""
import zipfile
import os
import glob
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
zip_filename = f"analysis_artifacts_{{timestamp}}.zip"

# Find files to zip
files_to_zip = []
# If specific IDs were passed, we'd filter here. For now, grab generated content.
files_to_zip.extend(glob.glob("*.png"))
files_to_zip.extend(glob.glob("*.csv"))
files_to_zip.extend(glob.glob("*.json"))
files_to_zip.extend(glob.glob("*.pkl"))

# Exclude the zip itself if it exists
if zip_filename in files_to_zip:
    files_to_zip.remove(zip_filename)

if not files_to_zip:
    print("No artifacts found to zip.")
else:
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in files_to_zip:
            zipf.write(file)
            print(f"Added {{file}} to archive")
    
    print(f"DATASET_SAVED:{{zip_filename}}")
    print(f"Created archive {{zip_filename}} with {{len(files_to_zip)}} files.")
"""
                return execute_python_in_sandbox(code, session_id)

            # Handle submit_dashboard_insights
            if tool_name == "submit_dashboard_insights":
                try:
                    import builtins

                    if not hasattr(builtins, "_session_store"):
                        return "Error: Session store not initialized."

                    if session_id not in builtins._session_store:
                        builtins._session_store[session_id] = {}

                    insights = tool_args.get("insights", [])
                    builtins._session_store[session_id]["agent_insights"] = insights
                    return "Insights successfully submitted to dashboard."
                except Exception as e:
                    return f"Error submitting insights: {str(e)}"

            # Handle delegate_coding_task
            if tool_name == "delegate_coding_task":
                task_desc = tool_args.get("task_description", "No description provided.")
                return f"{task_desc}"

            # Handle web_search tool
            if tool_name == "web_search":
                import asyncio
                from .web_search import web_search as do_web_search
                import builtins

                query = tool_args.get("query", "")
                search_config = {}
                if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                    search_config = builtins._session_store[session_id].get("search_config", {})

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            result = pool.submit(asyncio.run, do_web_search(query, search_config)).result()
                    else:
                        result = asyncio.run(do_web_search(query, search_config))
                    return result
                except Exception as e:
                    return f"Web search error: {str(e)}"

            # Default delegation for other tools (assuming they are python code tools or handled elsewhere)
            # If tool_args has 'code', it's likely python_code_interpreter
            if "code" in tool_args:
                return execute_python_in_sandbox(tool_args["code"], session_id)

            # Fallback/Error for unknown tools
            return f"Error: Tool '{tool_name}' not implemented in execute_tool wrapper."

else:

    def execute_python_in_sandbox(*args, **kwargs):
        raise NotImplementedError("execute_python_in_sandbox not available")

    def execute_tool(*args, **kwargs):
        raise NotImplementedError("execute_tool not available")


from .tool_definitions import delegate_coding_task
from .report_generation_tool import generate_comprehensive_report
from .dataset_explorer import inspect_dataset, list_files, load_file, combine_files

if hasattr(tools_module, "refresh_sandbox_data"):
    refresh_sandbox_data = tools_module.refresh_sandbox_data
else:

    def refresh_sandbox_data(*args, **kwargs):
        return False


__all__ = [
    "execute_python_in_sandbox",
    "execute_tool",
    "delegate_coding_task",
    "generate_comprehensive_report",
    "refresh_sandbox_data",
    "inspect_dataset",
    "list_files",
    "load_file",
    "combine_files",
]
