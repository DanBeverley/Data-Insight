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

        def execute_tool(*args, **kwargs):
            # Delegate to execute_python_in_sandbox
            return execute_python_in_sandbox(*args, **kwargs)

else:

    def execute_python_in_sandbox(*args, **kwargs):
        raise NotImplementedError("execute_python_in_sandbox not available")

    def execute_tool(*args, **kwargs):
        raise NotImplementedError("execute_tool not available")


__all__ = ["execute_python_in_sandbox", "execute_tool"]
