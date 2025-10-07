import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any
import pandas as pd
import io


@pytest.fixture(scope="module")
def api_client():
    from src.api import app
    return TestClient(app)


@pytest.mark.e2e
class TestAPIContracts:

    def test_session_create_endpoint_contract(self, api_client):
        response = api_client.post("/api/sessions/new")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)
        assert len(data["session_id"]) > 0

    def test_upload_endpoint_contract(self, api_client, housing_dataset):
        session_response = api_client.post("/api/sessions/new")
        session_id = session_response.json()["session_id"]

        csv_buffer = io.StringIO()
        housing_dataset.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        response = api_client.post(
            "/api/upload",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            data={"session_id": session_id}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "message" in data
        assert isinstance(data.get("rows"), int)
        assert isinstance(data.get("columns"), int)

    def test_profile_endpoint_contract(self, api_client, housing_dataset):
        session_response = api_client.post("/api/sessions/new")
        session_id = session_response.json()["session_id"]

        csv_buffer = io.StringIO()
        housing_dataset.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        api_client.post(
            "/api/upload",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            data={"session_id": session_id}
        )

        response = api_client.get(f"/api/data/{session_id}/profile")

        assert response.status_code == 200
        data = response.json()
        assert "column_profiles" in data or "profiles" in data or "dataset_summary" in data
        if "column_profiles" in data:
            assert isinstance(data["column_profiles"], dict)

    def test_chat_endpoint_contract(self, api_client, housing_dataset):
        session_response = api_client.post("/api/sessions/new")
        session_id = session_response.json()["session_id"]

        csv_buffer = io.StringIO()
        housing_dataset.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        api_client.post(
            "/api/upload",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            data={"session_id": session_id}
        )

        response = api_client.get(
            "/api/agent/chat-stream",
            params={
                "message": "What are the columns in the dataset?",
                "session_id": session_id,
                "web_search_enabled": "false"
            },
            timeout=60.0
        )

        assert response.status_code == 200
        assert response.text is not None

    def test_health_endpoint_contract(self, api_client):
        response = api_client.get("/")

        assert response.status_code == 200

    def test_error_response_contract(self, api_client):
        response = api_client.get(
            "/api/agent/chat-stream",
            params={
                "message": "test",
                "session_id": "invalid_session_does_not_exist_xyz",
                "web_search_enabled": "false"
            }
        )

        assert response.status_code in [400, 404, 500]

    def test_session_list_endpoint(self, api_client):
        response = api_client.get("/api/sessions")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))

    def test_session_delete_endpoint_contract(self, api_client):
        session_response = api_client.post("/api/sessions/new")
        session_id = session_response.json()["session_id"]

        response = api_client.delete(f"/api/sessions/{session_id}")

        assert response.status_code in [200, 204]


@pytest.mark.e2e
class TestAPIValidation:

    def test_upload_requires_session_id(self, api_client):
        csv_buffer = io.StringIO()
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode()

        response = api_client.post(
            "/api/upload",
            files={"file": ("test.csv", csv_bytes, "text/csv")}
        )

        assert response.status_code in [400, 422]

    def test_session_create_response_structure(self, api_client):
        response = api_client.post("/api/sessions/new")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)
