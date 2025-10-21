import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

try:
    from src.database.service import OptimizedDatabaseService
    from src.database.connection import DatabaseManager, DatabaseConfig
    from src.learning.persistent_storage import PipelineExecution, DatasetCharacteristics, ProjectConfig

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not available")
@pytest.mark.unit
class TestOptimizedDatabaseService:

    @pytest.fixture
    def mock_db_manager(self):
        manager = Mock(spec=DatabaseManager)
        manager.config = Mock(spec=DatabaseConfig)
        manager.config.is_postgresql.return_value = False
        return manager

    @pytest.fixture
    def service(self, mock_db_manager):
        return OptimizedDatabaseService(database_manager=mock_db_manager)

    def test_service_initialization(self, service):
        assert service.db is not None
        assert hasattr(service, "logger")
        assert hasattr(service, "sql_queries")

    def test_load_sql_queries_when_file_exists(self, mock_db_manager):
        service = OptimizedDatabaseService(database_manager=mock_db_manager)
        assert isinstance(service.sql_queries, str)

    def test_load_sql_queries_when_file_missing(self, service):
        assert service.sql_queries == "" or isinstance(service.sql_queries, str)

    def test_store_execution_sqlite_fallback(self, service, mock_db_manager):
        mock_db_manager.config.is_postgresql.return_value = False

        execution = Mock(spec=PipelineExecution)
        execution.execution_id = "test_exec"

        result = service.store_execution_optimized(execution)
        assert isinstance(result, bool)

    def test_get_system_summary_sqlite(self, service, mock_db_manager):
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value = MagicMock(
            __enter__=MagicMock(return_value=mock_session), __exit__=MagicMock()
        )
        mock_db_manager.config.is_postgresql.return_value = False

        summary = service.get_system_summary()

        assert isinstance(summary, dict)
        assert "total_executions" in summary
        assert "database_type" in summary
        assert summary["database_type"] == "SQLite"

    @patch("src.database.service.OptimizedDatabaseService._store_execution_sqlite")
    def test_store_execution_calls_sqlite_when_not_postgres(self, mock_sqlite, service, mock_db_manager):
        mock_db_manager.config.is_postgresql.return_value = False
        mock_sqlite.return_value = True

        execution = Mock(spec=PipelineExecution)
        result = service.store_execution_optimized(execution)

        mock_sqlite.assert_called_once()
        assert result is True

    def test_find_similar_executions_empty_when_not_postgres(self, service, mock_db_manager):
        mock_db_manager.config.is_postgresql.return_value = False

        dataset_chars = Mock(spec=DatasetCharacteristics)
        dataset_chars.n_samples = 1000
        dataset_chars.n_features = 10
        dataset_chars.domain = "test"
        dataset_chars.task_complexity_score = 0.5

        results = service.find_similar_executions(dataset_chars)
        assert results == []

    def test_get_domain_analytics_empty_when_not_postgres(self, service, mock_db_manager):
        mock_db_manager.config.is_postgresql.return_value = False

        results = service.get_domain_analytics()
        assert results == []

    def test_get_strategy_effectiveness_empty_when_not_postgres(self, service, mock_db_manager):
        mock_db_manager.config.is_postgresql.return_value = False

        results = service.get_strategy_effectiveness()
        assert results == []

    def test_cleanup_old_data_returns_zero_for_sqlite(self, service, mock_db_manager):
        mock_db_manager.config.is_postgresql.return_value = False
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value = MagicMock(
            __enter__=MagicMock(return_value=mock_session), __exit__=MagicMock()
        )

        result = service.cleanup_old_data(retention_days=30)
        assert result == 0

    def test_update_execution_feedback_sqlite(self, service, mock_db_manager):
        mock_session = MagicMock()
        mock_db_manager.get_session.return_value = MagicMock(
            __enter__=MagicMock(return_value=mock_session), __exit__=MagicMock()
        )
        mock_db_manager.config.is_postgresql.return_value = False

        result = service.update_execution_feedback("exec_123", 4.5, 0.9)
        assert isinstance(result, bool)

    def test_cleanup_handles_exception(self, service, mock_db_manager):
        mock_db_manager.get_session.side_effect = Exception("Database error")

        result = service.cleanup_old_data(365)
        assert result == 0

    def test_update_feedback_handles_exception(self, service, mock_db_manager):
        mock_db_manager.get_session.side_effect = Exception("Database error")

        result = service.update_execution_feedback("exec_123", 4.5, 0.9)
        assert result is False

    def test_get_system_summary_handles_exception(self, service, mock_db_manager):
        mock_db_manager.get_session.side_effect = Exception("Database error")

        summary = service.get_system_summary()
        assert "error" in summary

    def test_find_similar_executions_handles_exception(self, service, mock_db_manager):
        mock_db_manager.config.is_postgresql.return_value = True
        mock_db_manager.get_session.side_effect = Exception("Database error")

        dataset_chars = Mock(spec=DatasetCharacteristics)
        dataset_chars.n_samples = 1000
        dataset_chars.n_features = 10
        dataset_chars.domain = "test"
        dataset_chars.task_complexity_score = 0.5

        results = service.find_similar_executions(dataset_chars)
        assert results == []

    def test_get_database_service_function(self):
        from src.database.service import get_database_service

        service = get_database_service()
        assert isinstance(service, OptimizedDatabaseService)
