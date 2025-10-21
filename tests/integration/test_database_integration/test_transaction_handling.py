import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

try:
    from src.database.connection import DatabaseManager, DatabaseConfig
    from src.database.models import DatasetCharacteristics, ProjectConfig, PipelineExecution, Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not available")
@pytest.mark.integration
class TestDatabaseTransactionHandling:

    @pytest.fixture
    def db_engine(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine

    @pytest.fixture
    def db_session(self, db_engine):
        SessionLocal = sessionmaker(bind=db_engine)
        session = SessionLocal()
        yield session
        session.close()

    def test_transaction_commit_on_success(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="tx_success", n_samples=100, n_features=5, target_type="classification", domain="test"
        )

        db_session.add(dataset)
        db_session.commit()

        retrieved = db_session.query(DatasetCharacteristics).filter_by(dataset_hash="tx_success").first()
        assert retrieved is not None
        assert retrieved.n_samples == 100

    def test_transaction_rollback_on_error(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="tx_rollback", n_samples=100, n_features=5, target_type="classification", domain="test"
        )

        db_session.add(dataset)

        try:
            db_session.flush()
            raise Exception("Simulated error")
        except Exception:
            db_session.rollback()

        retrieved = db_session.query(DatasetCharacteristics).filter_by(dataset_hash="tx_rollback").first()
        assert retrieved is None

    def test_multiple_operations_in_transaction(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="multi_tx_ds", n_samples=100, n_features=5, target_type="classification", domain="test"
        )
        config = ProjectConfig(
            config_hash="multi_tx_cfg", objective="accuracy", domain="test", strategy_applied="baseline"
        )

        db_session.add(dataset)
        db_session.add(config)
        db_session.commit()

        retrieved_ds = db_session.query(DatasetCharacteristics).filter_by(dataset_hash="multi_tx_ds").first()
        retrieved_cfg = db_session.query(ProjectConfig).filter_by(config_hash="multi_tx_cfg").first()

        assert retrieved_ds is not None
        assert retrieved_cfg is not None

    def test_partial_rollback_scenario(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="partial_rb_ds", n_samples=100, n_features=5, target_type="classification", domain="test"
        )

        db_session.add(dataset)
        db_session.commit()

        config = ProjectConfig(
            config_hash="partial_rb_cfg", objective="accuracy", domain="test", strategy_applied="baseline"
        )

        db_session.add(config)

        try:
            raise Exception("Error after second insert")
        except Exception:
            db_session.rollback()

        retrieved_ds = db_session.query(DatasetCharacteristics).filter_by(dataset_hash="partial_rb_ds").first()
        retrieved_cfg = db_session.query(ProjectConfig).filter_by(config_hash="partial_rb_cfg").first()

        assert retrieved_ds is not None
        assert retrieved_cfg is None

    def test_foreign_key_constraint_handling(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="fk_ds", n_samples=100, n_features=5, target_type="classification", domain="test"
        )
        config = ProjectConfig(config_hash="fk_cfg", objective="accuracy", domain="test", strategy_applied="baseline")
        execution = PipelineExecution(
            execution_id="fk_test",
            session_id="session_fk",
            dataset_hash="fk_ds",
            config_hash="fk_cfg",
            timestamp=datetime.utcnow(),
        )

        db_session.add(dataset)
        db_session.add(config)
        db_session.add(execution)
        db_session.commit()

        retrieved = db_session.query(PipelineExecution).filter_by(execution_id="fk_test").first()
        assert retrieved is not None
        assert retrieved.dataset_hash == "fk_ds"

    def test_nested_transaction_simulation(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="nested_tx", n_samples=100, n_features=5, target_type="classification", domain="test"
        )

        db_session.begin_nested()
        db_session.add(dataset)
        db_session.commit()

        retrieved = db_session.query(DatasetCharacteristics).filter_by(dataset_hash="nested_tx").first()
        assert retrieved is not None

    def test_update_in_transaction(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="update_tx", n_samples=100, n_features=5, target_type="classification", domain="test"
        )

        db_session.add(dataset)
        db_session.commit()

        dataset.n_samples = 200
        db_session.commit()

        retrieved = db_session.query(DatasetCharacteristics).filter_by(dataset_hash="update_tx").first()
        assert retrieved.n_samples == 200

    def test_delete_in_transaction(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="delete_tx", n_samples=100, n_features=5, target_type="classification", domain="test"
        )

        db_session.add(dataset)
        db_session.commit()

        db_session.delete(dataset)
        db_session.commit()

        retrieved = db_session.query(DatasetCharacteristics).filter_by(dataset_hash="delete_tx").first()
        assert retrieved is None
