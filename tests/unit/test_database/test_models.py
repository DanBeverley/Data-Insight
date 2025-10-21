import pytest
from datetime import datetime
import json

try:
    from src.database.models import (
        SQLALCHEMY_AVAILABLE,
        DatasetCharacteristics,
        ProjectConfig,
        PipelineExecution,
        LearningPattern,
        ExecutionFeedback,
        SystemMetrics,
        JSONType,
    )
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
except ImportError:
    SQLALCHEMY_AVAILABLE = False


@pytest.mark.skipif(not SQLALCHEMY_AVAILABLE, reason="SQLAlchemy not available")
@pytest.mark.unit
class TestSQLAlchemyModels:

    @pytest.fixture
    def db_session(self):
        engine = create_engine("sqlite:///:memory:")
        from src.database.models import Base

        Base.metadata.create_all(engine)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        yield session
        session.close()

    def test_dataset_characteristics_creation(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="test_hash_123",
            n_samples=1000,
            n_features=10,
            n_categorical=3,
            n_numerical=7,
            target_type="classification",
            domain="housing",
        )
        db_session.add(dataset)
        db_session.commit()

        retrieved = db_session.query(DatasetCharacteristics).filter_by(dataset_hash="test_hash_123").first()
        assert retrieved is not None
        assert retrieved.n_samples == 1000
        assert retrieved.n_features == 10
        assert retrieved.domain == "housing"

    def test_dataset_characteristics_defaults(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="test_hash_456", n_samples=500, n_features=5, target_type="regression", domain="finance"
        )
        db_session.add(dataset)
        db_session.commit()

        retrieved = db_session.query(DatasetCharacteristics).first()
        assert retrieved.n_categorical == 0
        assert retrieved.n_numerical == 0
        assert retrieved.missing_ratio == 0.0
        assert retrieved.created_at is not None

    def test_project_config_with_jsontype(self, db_session):
        config = ProjectConfig(
            config_hash="config_abc",
            objective="classification",
            domain="healthcare",
            constraints={"time_limit": 300, "memory_limit": "4GB"},
            strategy_applied="auto_ml",
            compliance_requirements=["HIPAA", "GDPR"],
        )
        db_session.add(config)
        db_session.commit()

        retrieved = db_session.query(ProjectConfig).filter_by(config_hash="config_abc").first()
        assert retrieved is not None
        assert isinstance(retrieved.constraints, dict)
        assert retrieved.constraints["time_limit"] == 300
        assert isinstance(retrieved.compliance_requirements, list)
        assert "HIPAA" in retrieved.compliance_requirements

    def test_pipeline_execution_relationships(self, db_session):
        dataset = DatasetCharacteristics(
            dataset_hash="ds_123", n_samples=100, n_features=5, target_type="classification", domain="test"
        )
        config = ProjectConfig(config_hash="cfg_123", objective="accuracy", domain="test", strategy_applied="baseline")
        execution = PipelineExecution(
            execution_id="exec_123",
            session_id="session_123",
            dataset_hash="ds_123",
            config_hash="cfg_123",
            timestamp=datetime.utcnow(),
            success_rating=0.85,
        )

        db_session.add(dataset)
        db_session.add(config)
        db_session.add(execution)
        db_session.commit()

        retrieved = db_session.query(PipelineExecution).filter_by(execution_id="exec_123").first()
        assert retrieved.dataset.dataset_hash == "ds_123"
        assert retrieved.config.config_hash == "cfg_123"

    def test_pipeline_execution_json_fields(self, db_session):
        execution = PipelineExecution(
            execution_id="exec_json",
            session_id="session_json",
            dataset_hash="ds_test",
            config_hash="cfg_test",
            pipeline_stages=["data_prep", "feature_eng", "model_train"],
            final_performance={"accuracy": 0.92, "f1_score": 0.89},
            execution_metadata={"experiment_name": "test_exp", "version": "1.0"},
            timestamp=datetime.utcnow(),
        )

        db_session.add(execution)
        db_session.commit()

        retrieved = db_session.query(PipelineExecution).filter_by(execution_id="exec_json").first()
        assert isinstance(retrieved.pipeline_stages, list)
        assert "feature_eng" in retrieved.pipeline_stages
        assert isinstance(retrieved.final_performance, dict)
        assert retrieved.final_performance["accuracy"] == 0.92
        assert retrieved.execution_metadata["version"] == "1.0"

    def test_learning_pattern_creation(self, db_session):
        pattern = LearningPattern(
            pattern_id="pattern_001",
            pattern_type="feature_engineering",
            dataset_context={"domain": "finance", "size": "medium"},
            config_elements={"method": "polynomial_features"},
            success_indicators={"improvement": 0.15},
            confidence_score=0.78,
            usage_count=5,
        )
        db_session.add(pattern)
        db_session.commit()

        retrieved = db_session.query(LearningPattern).filter_by(pattern_id="pattern_001").first()
        assert retrieved.pattern_type == "feature_engineering"
        assert retrieved.confidence_score == 0.78
        assert retrieved.usage_count == 5

    def test_execution_feedback_relationship(self, db_session):
        execution = PipelineExecution(
            execution_id="exec_feedback",
            session_id="session_feedback",
            dataset_hash="ds_fb",
            config_hash="cfg_fb",
            timestamp=datetime.utcnow(),
        )
        feedback = ExecutionFeedback(
            feedback_id="fb_001",
            execution_id="exec_feedback",
            user_rating=4.5,
            issues_encountered=["slow_training"],
            performance_expectation_met=True,
            would_recommend=True,
        )

        db_session.add(execution)
        db_session.add(feedback)
        db_session.commit()

        retrieved_feedback = db_session.query(ExecutionFeedback).filter_by(feedback_id="fb_001").first()
        assert retrieved_feedback.execution.execution_id == "exec_feedback"
        assert retrieved_feedback.user_rating == 4.5
        assert retrieved_feedback.would_recommend is True

    def test_system_metrics_creation(self, db_session):
        metric = SystemMetrics(
            metric_id="metric_001",
            metric_type="latency",
            metric_value=125.5,
            metric_metadata={"percentile": "p95", "endpoint": "/api/predict"},
            timestamp=datetime.utcnow(),
        )
        db_session.add(metric)
        db_session.commit()

        retrieved = db_session.query(SystemMetrics).filter_by(metric_id="metric_001").first()
        assert retrieved.metric_type == "latency"
        assert retrieved.metric_value == 125.5
        assert retrieved.metric_metadata["percentile"] == "p95"

    def test_jsontype_handles_none(self):
        json_type = JSONType()
        result = json_type.process_bind_param(None, None)
        assert result is None

        result = json_type.process_result_value(None, None)
        assert result is None

    def test_jsontype_serialization(self):
        json_type = JSONType()
        test_data = {"key": "value", "number": 42}
        serialized = json_type.process_bind_param(test_data, None)
        assert isinstance(serialized, str)
        assert "key" in serialized

        deserialized = json_type.process_result_value(serialized, None)
        assert isinstance(deserialized, dict)
        assert deserialized["key"] == "value"

    def test_multiple_feedback_per_execution(self, db_session):
        execution = PipelineExecution(
            execution_id="exec_multi_fb",
            session_id="session_multi",
            dataset_hash="ds_multi",
            config_hash="cfg_multi",
            timestamp=datetime.utcnow(),
        )
        feedback1 = ExecutionFeedback(feedback_id="fb_multi_1", execution_id="exec_multi_fb", user_rating=3.0)
        feedback2 = ExecutionFeedback(feedback_id="fb_multi_2", execution_id="exec_multi_fb", user_rating=4.5)

        db_session.add(execution)
        db_session.add(feedback1)
        db_session.add(feedback2)
        db_session.commit()

        retrieved_execution = db_session.query(PipelineExecution).filter_by(execution_id="exec_multi_fb").first()
        assert len(retrieved_execution.feedback_records) == 2
        ratings = [fb.user_rating for fb in retrieved_execution.feedback_records]
        assert 3.0 in ratings
        assert 4.5 in ratings
