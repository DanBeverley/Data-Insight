import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from src.knowledge_graph.service import (
    KnowledgeGraphService,
    SessionDataStorage,
    GraphDatabaseInterface,
    NodeType,
    RelationshipType,
)
from src.knowledge_graph.schema import Relationship


@pytest.mark.integration
class TestKnowledgeGraphOperations:
    @pytest.fixture
    def mock_database(self):
        db = Mock(spec=GraphDatabaseInterface)
        db.connect.return_value = True
        db.create_node.return_value = "node_123"
        db.create_relationship.return_value = True
        db.find_node.return_value = {"id": "node_123", "name": "Test"}
        db.find_nodes.return_value = [{"id": "node_1"}, {"id": "node_2"}]
        db.execute_query.return_value = [{"result": "data"}]
        db.get_node_relationships.return_value = []
        return db

    @pytest.fixture
    def kg_service(self, mock_database):
        service = KnowledgeGraphService(database=mock_database)
        return service

    def test_session_data_storage_initialization(self):
        storage = SessionDataStorage()
        assert storage.sessions == {}
        assert storage.correlations == []
        assert storage.datasets == []

    def test_add_session_to_storage(self):
        storage = SessionDataStorage()
        data = {"dataset_info": {"name": "test"}, "correlations": [{"feature1": "feature2", "value": 0.8}]}

        storage.add_session("session_1", data)

        assert "session_1" in storage.sessions
        assert len(storage.datasets) == 1
        assert len(storage.correlations) == 1

    def test_get_all_data_from_storage(self):
        storage = SessionDataStorage()
        storage.add_session("session_1", {"dataset_info": {"name": "test"}})

        all_data = storage.get_all_data()

        assert "sessions" in all_data
        assert "datasets" in all_data
        assert all_data["sessions"] == 1

    def test_kg_service_initialization(self, kg_service, mock_database):
        assert kg_service.database == mock_database
        assert kg_service.connected is False

    def test_initialize_connects_to_database(self, kg_service, mock_database):
        result = kg_service.initialize()

        assert result is True
        assert kg_service.connected is True
        mock_database.connect.assert_called_once()

    def test_shutdown_disconnects_database(self, kg_service, mock_database):
        kg_service.connected = True
        kg_service.shutdown()

        assert kg_service.connected is False
        mock_database.disconnect.assert_called_once()

    def test_create_dataset_node(self, kg_service, mock_database):
        kg_service.connected = True
        dataset_chars = {"name": "Test Dataset", "shape": (100, 10), "domain": "finance"}

        node_id = kg_service.create_dataset_node(dataset_chars)

        assert node_id == "node_123"
        mock_database.create_node.assert_called_once()

    def test_create_model_node(self, kg_service, mock_database):
        kg_service.connected = True
        model_info = {"name": "Test Model", "algorithm": "RandomForest"}
        performance = {"accuracy": 0.95}

        node_id = kg_service.create_model_node(model_info, performance)

        assert node_id == "node_123"
        mock_database.create_node.assert_called_once()

    def test_create_project_node(self, kg_service, mock_database):
        kg_service.connected = True
        project_def = {"name": "Test Project", "objective": "classification", "domain": "healthcare"}

        node_id = kg_service.create_project_node(project_def)

        assert node_id == "node_123"
        mock_database.create_node.assert_called_once()

    def test_create_execution_node(self, kg_service, mock_database):
        kg_service.connected = True
        execution_info = {"execution_id": "exec_123", "start_time": datetime.now(), "status": "completed"}

        node_id = kg_service.create_execution_node(execution_info)

        assert node_id is not None
        mock_database.create_node.assert_called_once()

    def test_link_execution_to_project(self, kg_service, mock_database):
        kg_service.connected = True

        result = kg_service.link_execution_to_project("exec_123", "proj_123")

        assert result is True
        mock_database.create_relationship.assert_called_once()

    def test_link_model_to_project(self, kg_service, mock_database):
        kg_service.connected = True

        result = kg_service.link_model_to_project("model_123", "proj_123")

        assert result is True
        mock_database.create_relationship.assert_called_once()

    def test_query_similar_projects(self, kg_service, mock_database):
        kg_service.connected = True
        project_chars = {"domain": "finance", "constraints": {"time_limit": 300}}

        results = kg_service.query_similar_projects(project_chars, limit=5)

        assert isinstance(results, list)

    def test_get_feature_importance_ranking(self, kg_service, mock_database):
        kg_service.connected = True

        results = kg_service.get_feature_importance_ranking("revenue_prediction", limit=10)

        assert isinstance(results, list)

    def test_record_pipeline_execution(self, kg_service, mock_database):
        kg_service.connected = True
        execution_data = {
            "dataset_characteristics": {"name": "test", "shape": (100, 10), "domain": "test"},
            "project_definition": {"name": "test project", "objective": "test"},
            "model_info": {"name": "test model"},
            "performance_metrics": {"accuracy": 0.9},
        }

        node_ids = kg_service.record_pipeline_execution(execution_data)

        assert isinstance(node_ids, dict)
        assert "dataset" in node_ids or "project" in node_ids or "execution" in node_ids

    def test_add_execution_to_storage(self):
        storage = SessionDataStorage()
        execution_data = {"code": "print(1)", "success": True}

        storage.add_execution(execution_data)

        assert len(storage.executions) == 1

    def test_storage_limits_executions_to_500(self):
        storage = SessionDataStorage()

        for i in range(600):
            storage.add_execution({"id": i})

        assert len(storage.executions) == 500

    def test_get_recent_executions(self):
        storage = SessionDataStorage()

        for i in range(20):
            storage.add_execution({"id": i})

        all_data = storage.get_all_data()

        assert len(all_data["executions"]) == 10
