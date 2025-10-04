import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.integration
class TestSandboxCodeIsolation:

    def test_sandbox_prevents_file_system_access(self):
        from data_scientist_chatbot.app.tools.executor import execute_tool

        malicious_code = """
import os
os.system('rm -rf /')
"""

        with patch('data_scientist_chatbot.app.tools.execute_python_in_sandbox') as mock_sandbox:
            mock_sandbox.return_value = {
                'success': False,
                'stdout': '',
                'stderr': 'SecurityError: File system access not allowed',
                'plots': []
            }

            result = execute_tool(
                "python_code_interpreter",
                {"code": malicious_code},
                "test_session_security"
            )

            assert "Error" in result or "SecurityError" in result


    def test_sandbox_prevents_network_access(self):
        from data_scientist_chatbot.app.tools.executor import execute_tool

        network_code = """
import urllib.request
urllib.request.urlopen('http://malicious-site.com')
"""

        with patch('data_scientist_chatbot.app.tools.executor.execute_python_in_sandbox') as mock_sandbox:
            mock_sandbox.return_value = {
                'success': False,
                'stdout': '',
                'stderr': 'NetworkError: Network access not allowed in sandbox',
                'plots': []
            }

            result = execute_tool(
                "python_code_interpreter",
                {"code": network_code},
                "test_session_network"
            )

            assert "Error" in result or "Network" in result


    def test_sandbox_memory_limits(self):
        from data_scientist_chatbot.app.tools.executor import execute_tool

        memory_intensive_code = """
huge_list = [0] * (10**9)
"""

        with patch('data_scientist_chatbot.app.tools.execute_python_in_sandbox') as mock_sandbox:
            mock_sandbox.return_value = {
                'success': False,
                'stdout': '',
                'stderr': 'MemoryError: Memory limit exceeded',
                'plots': []
            }

            result = execute_tool(
                "python_code_interpreter",
                {"code": memory_intensive_code},
                "test_session_memory"
            )

            assert "Error" in result or "Memory" in result


    def test_sandbox_timeout_enforcement(self):
        from data_scientist_chatbot.app.tools.executor import execute_tool

        infinite_loop_code = """
while True:
    pass
"""

        with patch('data_scientist_chatbot.app.tools.execute_python_in_sandbox') as mock_sandbox:
            mock_sandbox.return_value = {
                'success': False,
                'stdout': '',
                'stderr': 'TimeoutError: Execution exceeded time limit',
                'plots': []
            }

            result = execute_tool(
                "python_code_interpreter",
                {"code": infinite_loop_code},
                "test_session_timeout"
            )

            assert "Error" in result or "Timeout" in result


    def test_sandbox_safe_code_execution(self):
        from data_scientist_chatbot.app.tools.executor import execute_tool

        safe_code = """
import pandas as pd
result = 2 + 2
print(result)
"""

        with patch('data_scientist_chatbot.app.tools.execute_python_in_sandbox') as mock_sandbox:
            mock_sandbox.return_value = {
                'success': True,
                'stdout': '4\n',
                'stderr': '',
                'plots': []
            }

            result = execute_tool(
                "python_code_interpreter",
                {"code": safe_code},
                "test_session_safe"
            )

            assert "4" in result
            assert "Error" not in result


    def test_sandbox_stateful_execution(self):
        from data_scientist_chatbot.app.tools.executor import execute_tool

        with patch('data_scientist_chatbot.app.tools.executor.execute_python_in_sandbox') as mock_sandbox:
            mock_sandbox.side_effect = [
                {'success': True, 'stdout': '', 'stderr': '', 'plots': []},
                {'success': True, 'stdout': '10\n', 'stderr': '', 'plots': []}
            ]

            session_id = "test_stateful_session"

            execute_tool("python_code_interpreter", {"code": "x = 10"}, session_id)
            result2 = execute_tool("python_code_interpreter", {"code": "print(x)"}, session_id)

            assert mock_sandbox.call_count == 2


    def test_sandbox_plot_generation(self):
        from data_scientist_chatbot.app.tools.executor import execute_tool

        plot_code = """
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig('test_plot.png')
"""

        with patch('data_scientist_chatbot.app.tools.execute_python_in_sandbox') as mock_sandbox:
            mock_sandbox.return_value = {
                'success': True,
                'stdout': '',
                'stderr': '',
                'plots': ['test_plot.png']
            }

            result = execute_tool(
                "python_code_interpreter",
                {"code": plot_code},
                "test_session_plot"
            )

            assert "test_plot.png" in result or "visualization" in result.lower()


    def test_sandbox_dataframe_persistence(self):
        from data_scientist_chatbot.app.tools.executor import execute_tool

        with patch('data_scientist_chatbot.app.tools.executor.execute_python_in_sandbox') as mock_sandbox:
            mock_sandbox.return_value = {
                'success': True,
                'stdout': '(5, 2)\n',
                'stderr': '',
                'plots': []
            }

            code = "print(df.shape)"
            result = execute_tool(
                "python_code_interpreter",
                {"code": code},
                "test_session_df"
            )

            assert "(5, 2)" in result or "5" in result


    def test_sandbox_error_handling(self):
        from data_scientist_chatbot.app.tools.executor import execute_tool

        error_code = """
undefined_variable + 1
"""

        with patch('data_scientist_chatbot.app.tools.execute_python_in_sandbox') as mock_sandbox:
            mock_sandbox.return_value = {
                'success': False,
                'stdout': '',
                'stderr': "NameError: name 'undefined_variable' is not defined",
                'plots': []
            }

            result = execute_tool(
                "python_code_interpreter",
                {"code": error_code},
                "test_session_error"
            )

            assert "Error" in result or "NameError" in result


    def test_sandbox_package_installation_prevention(self):
        from data_scientist_chatbot.app.tools.executor import execute_tool

        install_code = """
import subprocess
subprocess.run(['pip', 'install', 'malicious-package'])
"""

        with patch('data_scientist_chatbot.app.tools.execute_python_in_sandbox') as mock_sandbox:
            mock_sandbox.return_value = {
                'success': False,
                'stdout': '',
                'stderr': 'SecurityError: Package installation not allowed',
                'plots': []
            }

            result = execute_tool(
                "python_code_interpreter",
                {"code": install_code},
                "test_session_install"
            )

            assert "Error" in result or "SecurityError" in result
