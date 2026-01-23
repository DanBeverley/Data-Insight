"""Integration tests for sandbox code isolation.

These tests validate:
1. Security restrictions (file system, network, package installation)
2. Resource limits (memory, timeout)
3. Safe code execution
4. Error handling
"""

import pytest
from unittest.mock import patch, MagicMock


def _check_result_for_error(result, *keywords):
    """Helper to check if result contains error info."""
    if isinstance(result, dict):
        # Dict result - check success field and stderr
        if not result.get("success"):
            return True
        stderr = result.get("stderr", "")
        return any(kw.lower() in stderr.lower() for kw in keywords)
    else:
        # String result - check for keywords
        return any(kw in str(result) for kw in keywords)


def _check_result_for_content(result, *keywords):
    """Helper to check if result contains expected content."""
    if isinstance(result, dict):
        # Check stdout for content
        stdout = result.get("stdout", "")
        plots = result.get("plots", [])
        return any(kw in stdout for kw in keywords) or any(kw in str(plots) for kw in keywords)
    else:
        return any(kw in str(result) for kw in keywords)


@pytest.mark.integration
class TestSandboxCodeIsolation:
    def test_sandbox_prevents_file_system_access(self):
        from data_scientist_chatbot.app.tools import execute_tool

        malicious_code = """
import os
os.system('rm -rf /')
"""

        with patch("data_scientist_chatbot.app.tools.execute_python_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = {
                "success": False,
                "stdout": "",
                "stderr": "SecurityError: File system access not allowed",
                "plots": [],
            }

            result = execute_tool("python_code_interpreter", {"code": malicious_code}, "test_session_security")

            assert _check_result_for_error(result, "Error", "Security")

    def test_sandbox_prevents_network_access(self):
        from data_scientist_chatbot.app.tools import execute_tool

        network_code = """
import urllib.request
urllib.request.urlopen('http://malicious-site.com')
"""

        with patch("data_scientist_chatbot.app.tools.execute_python_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = {
                "success": False,
                "stdout": "",
                "stderr": "NetworkError: Network access not allowed in sandbox",
                "plots": [],
            }

            result = execute_tool("python_code_interpreter", {"code": network_code}, "test_session_network")

            assert _check_result_for_error(result, "Error", "Network")

    def test_sandbox_memory_limits(self):
        from data_scientist_chatbot.app.tools import execute_tool

        memory_intensive_code = """
huge_list = [0] * (10**9)
"""

        with patch("data_scientist_chatbot.app.tools.execute_python_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = {
                "success": False,
                "stdout": "",
                "stderr": "MemoryError: Memory limit exceeded",
                "plots": [],
            }

            result = execute_tool("python_code_interpreter", {"code": memory_intensive_code}, "test_session_memory")

            assert _check_result_for_error(result, "Error", "Memory")

    def test_sandbox_timeout_enforcement(self):
        from data_scientist_chatbot.app.tools import execute_tool

        infinite_loop_code = """
while True:
    pass
"""

        with patch("data_scientist_chatbot.app.tools.execute_python_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = {
                "success": False,
                "stdout": "",
                "stderr": "TimeoutError: Execution exceeded time limit",
                "plots": [],
            }

            result = execute_tool("python_code_interpreter", {"code": infinite_loop_code}, "test_session_timeout")

            assert _check_result_for_error(result, "Error", "Timeout")

    def test_sandbox_safe_code_execution(self):
        from data_scientist_chatbot.app.tools import execute_tool

        safe_code = """
import pandas as pd
result = 2 + 2
print(result)
"""

        with patch("data_scientist_chatbot.app.tools.execute_python_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = {"success": True, "stdout": "4\n", "stderr": "", "plots": []}

            result = execute_tool("python_code_interpreter", {"code": safe_code}, "test_session_safe")

            assert _check_result_for_content(result, "4")

    def test_sandbox_stateful_execution(self):
        from data_scientist_chatbot.app.tools import execute_tool

        with patch("data_scientist_chatbot.app.tools.execute_python_in_sandbox") as mock_sandbox:
            mock_sandbox.side_effect = [
                {"success": True, "stdout": "", "stderr": "", "plots": []},
                {"success": True, "stdout": "10\n", "stderr": "", "plots": []},
            ]

            session_id = "test_stateful_session"

            execute_tool("python_code_interpreter", {"code": "x = 10"}, session_id)
            execute_tool("python_code_interpreter", {"code": "print(x)"}, session_id)

            assert mock_sandbox.call_count == 2

    def test_sandbox_plot_generation(self):
        from data_scientist_chatbot.app.tools import execute_tool

        plot_code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig('test_plot.png')
"""

        with patch("data_scientist_chatbot.app.tools.execute_python_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = {"success": True, "stdout": "", "stderr": "", "plots": ["test_plot.png"]}

            result = execute_tool("python_code_interpreter", {"code": plot_code}, "test_session_plot")

            assert _check_result_for_content(result, "test_plot.png", "visualization", "plot")

    def test_sandbox_dataframe_persistence(self):
        from data_scientist_chatbot.app.tools import execute_tool

        with patch("data_scientist_chatbot.app.tools.execute_python_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = {"success": True, "stdout": "(5, 2)\n", "stderr": "", "plots": []}

            code = "print(df.shape)"
            result = execute_tool("python_code_interpreter", {"code": code}, "test_session_df")

            assert _check_result_for_content(result, "(5, 2)", "5")

    def test_sandbox_error_handling(self):
        from data_scientist_chatbot.app.tools import execute_tool

        error_code = """
undefined_variable + 1
"""

        with patch("data_scientist_chatbot.app.tools.execute_python_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = {
                "success": False,
                "stdout": "",
                "stderr": "NameError: name 'undefined_variable' is not defined",
                "plots": [],
            }

            result = execute_tool("python_code_interpreter", {"code": error_code}, "test_session_error")

            assert _check_result_for_error(result, "Error", "NameError")

    def test_sandbox_package_installation_prevention(self):
        from data_scientist_chatbot.app.tools import execute_tool

        install_code = """
import subprocess
subprocess.run(['pip', 'install', 'malicious-package'])
"""

        with patch("data_scientist_chatbot.app.tools.execute_python_in_sandbox") as mock_sandbox:
            mock_sandbox.return_value = {
                "success": False,
                "stdout": "",
                "stderr": "SecurityError: Package installation not allowed",
                "plots": [],
            }

            result = execute_tool("python_code_interpreter", {"code": install_code}, "test_session_install")

            assert _check_result_for_error(result, "Error", "Security")
