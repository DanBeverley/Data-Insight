import pytest
from typing import Dict, Any


@pytest.mark.security
class TestInputSanitization:
    SQL_INJECTION_PAYLOADS = ["'; DROP TABLE users; --", "1' OR '1'='1", "admin'--", "' UNION SELECT NULL--"]

    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<svg onload=alert('XSS')>",
    ]

    COMMAND_INJECTION_PAYLOADS = ["; ls -la", "| cat /etc/passwd", "`whoami`", "$(cat /etc/passwd)"]

    def test_sql_injection_protection(self, sample_session_id: str):
        from data_scientist_chatbot.app.utils.text_processing import sanitize_input

        for payload in self.SQL_INJECTION_PAYLOADS:
            sanitized = sanitize_input(payload)
            assert "DROP" not in sanitized.upper()
            assert "UNION" not in sanitized.upper()
            assert "--" not in sanitized

    def test_xss_protection(self, sample_session_id: str):
        from data_scientist_chatbot.app.utils.text_processing import sanitize_input

        for payload in self.XSS_PAYLOADS:
            sanitized = sanitize_input(payload)
            assert "<script" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "onerror" not in sanitized.lower()

    def test_command_injection_protection(self, sample_session_id: str):
        from data_scientist_chatbot.app.utils.text_processing import sanitize_input

        for payload in self.COMMAND_INJECTION_PAYLOADS:
            sanitized = sanitize_input(payload)
            assert "`" not in sanitized
            assert "$(" not in sanitized
            assert "|" not in sanitized or "pipe" in sanitized.lower()

    def test_path_traversal_protection(self, sample_session_id: str):
        from data_scientist_chatbot.app.utils.text_processing import sanitize_file_path

        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
        ]

        for path in dangerous_paths:
            with pytest.raises(ValueError):
                sanitize_file_path(path)

    def test_session_id_validation(self):
        from src.api_utils.session_management import validate_session_id

        valid_ids = ["550e8400-e29b-41d4-a716-446655440000", "test_session_123", "session-2024-01-01"]

        invalid_ids = ["../../../etc/passwd", "<script>alert(1)</script>", "'; DROP TABLE sessions;--", ""]

        for session_id in valid_ids:
            assert validate_session_id(session_id) is True

        for session_id in invalid_ids:
            assert validate_session_id(session_id) is False

    def test_file_upload_validation(self):
        from src.api_utils.upload_handler import validate_file_upload
        import io

        safe_content = io.BytesIO(b"col1,col2\n1,2\n3,4")
        assert validate_file_upload(safe_content, "test.csv") is True

        malicious_content = io.BytesIO(b"<?php system($_GET['cmd']); ?>")
        assert validate_file_upload(malicious_content, "test.php") is False

    def test_agent_query_length_limit(self, sample_session_id: str):
        from src.api_utils.agent_response import stream_agent_response

        extremely_long_query = "A" * 100000

        with pytest.raises(ValueError, match="query too long|exceeds maximum"):
            list(stream_agent_response(extremely_long_query, sample_session_id, False))

    def test_no_code_execution_in_user_input(self, sample_session_id: str):
        from data_scientist_chatbot.app.utils.text_processing import sanitize_input

        code_payloads = ["exec('import os; os.system(\"rm -rf /\")')", "eval('1+1')", "__import__('os').system('ls')"]

        for payload in code_payloads:
            sanitized = sanitize_input(payload)
            assert "exec" not in sanitized.lower()
            assert "eval" not in sanitized.lower()
            assert "__import__" not in sanitized
