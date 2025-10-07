import pytest
from unittest.mock import patch, MagicMock
import requests


@pytest.mark.chaos
class TestOllamaFailures:

    def test_ollama_connection_timeout(self, sample_session_id: str):
        from src.api_utils.agent_response import stream_agent_response

        with patch('ollama.chat') as mock_chat:
            mock_chat.side_effect = requests.exceptions.Timeout("Connection timeout")

            result = []
            try:
                for chunk in stream_agent_response("test query", sample_session_id, False):
                    result.append(chunk)
            except Exception as e:
                assert "timeout" in str(e).lower() or "unavailable" in str(e).lower()

    def test_ollama_connection_refused(self, sample_session_id: str):
        from src.api_utils.agent_response import stream_agent_response

        with patch('ollama.chat') as mock_chat:
            mock_chat.side_effect = requests.exceptions.ConnectionError("Connection refused")

            result = []
            try:
                for chunk in stream_agent_response("test query", sample_session_id, False):
                    result.append(chunk)
            except Exception as e:
                assert "connection" in str(e).lower() or "unavailable" in str(e).lower()

    def test_ollama_model_not_loaded(self, sample_session_id: str):
        from src.api_utils.agent_response import stream_agent_response

        with patch('ollama.chat') as mock_chat:
            mock_chat.side_effect = Exception("model 'nonexistent:latest' not found")

            result = []
            try:
                for chunk in stream_agent_response("test query", sample_session_id, False):
                    result.append(chunk)
            except Exception as e:
                assert "model" in str(e).lower() or "not found" in str(e).lower()

    def test_ollama_incomplete_response(self, sample_session_id: str):
        from src.api_utils.agent_response import stream_agent_response

        with patch('ollama.chat') as mock_chat:
            mock_response = MagicMock()
            mock_response.get.return_value = {"message": {"content": ""}}
            mock_chat.return_value = mock_response

            result = []
            for chunk in stream_agent_response("test query", sample_session_id, False):
                result.append(chunk)

            assert len(result) > 0

    def test_ollama_malformed_json_response(self, sample_session_id: str):
        from src.api_utils.agent_response import stream_agent_response

        with patch('ollama.chat') as mock_chat:
            mock_chat.return_value = "not json"

            result = []
            try:
                for chunk in stream_agent_response("test query", sample_session_id, False):
                    result.append(chunk)
            except Exception as e:
                assert "json" in str(e).lower() or "parse" in str(e).lower()

    def test_ollama_slow_response_handling(self, sample_session_id: str):
        from src.api_utils.agent_response import stream_agent_response
        import time

        with patch('ollama.chat') as mock_chat:
            def slow_response(*args, **kwargs):
                time.sleep(5)
                return {"message": {"content": "slow response"}}

            mock_chat.side_effect = slow_response

            start = time.time()
            result = []
            for chunk in stream_agent_response("test query", sample_session_id, False):
                result.append(chunk)
            duration = time.time() - start

            assert duration < 10

    def test_graceful_degradation_to_fallback(self, sample_session_id: str):
        from src.api_utils.agent_response import stream_agent_response

        with patch('ollama.chat') as mock_chat:
            mock_chat.side_effect = Exception("Ollama unavailable")

            result = []
            for chunk in stream_agent_response("test query", sample_session_id, False):
                result.append(chunk)

            assert len(result) > 0
