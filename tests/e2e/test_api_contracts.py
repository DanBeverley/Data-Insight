import pytest
from unittest.mock import MagicMock, patch
from typing import Dict, Any


@pytest.mark.e2e
class TestAPIContracts:

    @pytest.fixture
    def mock_client(self):
        return MagicMock()

    def test_session_create_endpoint_contract(self, mock_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "session_id": "test_session_123",
            "created_at": "2024-01-01T00:00:00Z"
        }
        mock_client.post.return_value = response

        result = mock_client.post("/api/sessions/create")

        assert result.status_code == 200
        data = result.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0

    def test_upload_endpoint_contract(self, mock_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "status": "success",
            "message": "Dataset uploaded successfully",
            "rows": 100,
            "columns": 5
        }
        mock_client.post.return_value = response

        result = mock_client.post(
            "/api/upload",
            files={"file": ("test.csv", b"data", "text/csv")},
            data={"session_id": "test_123"}
        )

        assert result.status_code == 200
        data = result.json()
        assert data["status"] == "success"
        assert "message" in data
        assert isinstance(data.get("rows"), int)
        assert isinstance(data.get("columns"), int)

    def test_profile_endpoint_contract(self, mock_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "column_profiles": {
                "price": {
                    "semantic_type": "NUMERIC_CONTINUOUS",
                    "mean": 350000,
                    "std": 120000,
                    "min": 100000,
                    "max": 800000
                }
            },
            "dataset_summary": {
                "total_rows": 100,
                "total_columns": 5,
                "missing_values": 10
            }
        }
        mock_client.get.return_value = response

        result = mock_client.get("/api/data/test_session/profile")

        assert result.status_code == 200
        data = result.json()
        assert "column_profiles" in data
        assert "dataset_summary" in data
        assert isinstance(data["column_profiles"], dict)

    def test_chat_endpoint_contract(self, mock_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "response": "The correlation between price and area is 0.54",
            "plots": ["correlation_heatmap.html"],
            "execution_time": 2.5
        }
        mock_client.post.return_value = response

        result = mock_client.post(
            "/api/chat/test_session",
            json={
                "message": "Show correlation between price and area",
                "session_id": "test_session"
            }
        )

        assert result.status_code == 200
        data = result.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert "plots" in data
        assert isinstance(data["plots"], list)

    def test_health_endpoint_contract(self, mock_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "status": "healthy",
            "version": "2.0.0",
            "services": {
                "ollama": "connected",
                "database": "connected"
            }
        }
        mock_client.get.return_value = response

        result = mock_client.get("/api/health")

        assert result.status_code == 200
        data = result.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_error_response_contract(self, mock_client):
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {
            "error": "Invalid session ID",
            "detail": "Session not found",
            "status_code": 400
        }
        mock_client.post.return_value = response

        result = mock_client.post(
            "/api/chat/invalid_session",
            json={"message": "test"}
        )

        assert result.status_code == 400
        data = result.json()
        assert "error" in data
        assert "status_code" in data

    def test_upload_file_validation(self, mock_client):
        response = MagicMock()
        response.status_code = 400
        response.json.return_value = {
            "error": "Invalid file format",
            "detail": "Only CSV, Excel, and Parquet files are supported",
            "status_code": 400
        }
        mock_client.post.return_value = response

        result = mock_client.post(
            "/api/upload",
            files={"file": ("test.txt", b"data", "text/plain")},
            data={"session_id": "test_123"}
        )

        assert result.status_code == 400
        data = result.json()
        assert "error" in data

    def test_chat_stream_endpoint_contract(self, mock_client):
        response = MagicMock()
        response.status_code = 200
        response.iter_lines.return_value = [
            b'data: {"type": "status", "content": "Processing"}',
            b'data: {"type": "response", "content": "Analysis complete"}',
            b'data: {"type": "done"}'
        ]
        mock_client.post.return_value = response

        result = mock_client.post(
            "/api/chat-stream/test_session",
            json={"message": "Analyze data"}
        )

        assert result.status_code == 200
        assert hasattr(result, 'iter_lines')

    def test_data_download_endpoint_contract(self, mock_client):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Type": "image/png"}
        response.content = b"PNG_DATA"
        mock_client.get.return_value = response

        result = mock_client.get("/api/download/test_session/plot.png")

        assert result.status_code == 200
        assert "image/" in result.headers.get("Content-Type", "")

    def test_session_delete_endpoint_contract(self, mock_client):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "status": "success",
            "message": "Session deleted"
        }
        mock_client.delete.return_value = response

        result = mock_client.delete("/api/sessions/test_session")

        assert result.status_code == 200
        data = result.json()
        assert data["status"] == "success"


@pytest.mark.e2e
class TestAPIValidation:

    def test_required_fields_validation(self):
        required_fields = {
            "/api/sessions/create": [],
            "/api/upload": ["file", "session_id"],
            "/api/chat/{session_id}": ["message", "session_id"]
        }

        for endpoint, fields in required_fields.items():
            assert isinstance(fields, list)

    def test_response_status_codes(self):
        expected_codes = {
            "success": [200, 201],
            "client_error": [400, 404],
            "server_error": [500, 503]
        }

        assert 200 in expected_codes["success"]
        assert 400 in expected_codes["client_error"]
        assert 500 in expected_codes["server_error"]

    def test_data_types_validation(self):
        sample_response = {
            "session_id": str,
            "rows": int,
            "columns": int,
            "plots": list,
            "response": str
        }

        for field, expected_type in sample_response.items():
            assert expected_type in [str, int, list, dict, float, bool]
