"""SQLAlchemy models for production database schema"""

from datetime import datetime
from typing import Optional, Dict, Any
import json

try:
    from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, Index
    from sqlalchemy.orm import declarative_base, relationship
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.types import TypeDecorator, VARCHAR

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class JSONType(TypeDecorator):
        """Platform-independent JSON type"""

        impl = VARCHAR

        def load_dialect_impl(self, dialect):
            if dialect.name == "postgresql":
                return dialect.type_descriptor(JSONB())
            else:
                return dialect.type_descriptor(VARCHAR())

        def process_bind_param(self, value, dialect):
            if value is None:
                return value
            return json.dumps(value)

        def process_result_value(self, value, dialect):
            if value is None:
                return value
            if isinstance(value, dict):
                return value
            return json.loads(value)

    class DatasetCharacteristics(Base):
        """Dataset characteristics for meta-learning"""

        __tablename__ = "dataset_characteristics"

        dataset_hash = Column(String(64), primary_key=True)
        n_samples = Column(Integer, nullable=False)
        n_features = Column(Integer, nullable=False)
        n_categorical = Column(Integer, default=0)
        n_numerical = Column(Integer, default=0)
        n_text = Column(Integer, default=0)
        n_datetime = Column(Integer, default=0)
        missing_ratio = Column(Float, default=0.0)
        target_type = Column(String(50), nullable=False)
        target_cardinality = Column(Integer, default=0)
        class_imbalance_ratio = Column(Float, default=0.0)
        correlation_strength = Column(Float, default=0.0)
        skewness_avg = Column(Float, default=0.0)
        kurtosis_avg = Column(Float, default=0.0)
        domain = Column(String(50), nullable=False)
        task_complexity_score = Column(Float, default=0.0)
        feature_diversity_score = Column(Float, default=0.0)
        data_quality_score = Column(Float, default=0.0)
        created_at = Column(DateTime, default=datetime.utcnow)

        # Relationships
        executions = relationship("PipelineExecution", back_populates="dataset")

        # Indexes for efficient querying
        __table_args__ = (
            Index("idx_dataset_domain", "domain"),
            Index("idx_dataset_complexity", "task_complexity_score"),
            Index("idx_dataset_quality", "data_quality_score"),
            Index("idx_dataset_type", "target_type"),
        )

    class ProjectConfig(Base):
        """Project configuration patterns"""

        __tablename__ = "project_configs"

        config_hash = Column(String(64), primary_key=True)
        objective = Column(String(50), nullable=False)
        domain = Column(String(50), nullable=False)
        constraints = Column(JSONType, default=dict)
        strategy_applied = Column(String(100), nullable=False)
        feature_engineering_enabled = Column(Boolean, default=True)
        feature_selection_enabled = Column(Boolean, default=True)
        security_level = Column(String(20), default="standard")
        privacy_level = Column(String(20), default="medium")
        compliance_requirements = Column(JSONType, default=list)
        created_at = Column(DateTime, default=datetime.utcnow)

        # Relationships
        executions = relationship("PipelineExecution", back_populates="config")

        # Indexes
        __table_args__ = (
            Index("idx_config_objective", "objective"),
            Index("idx_config_domain", "domain"),
            Index("idx_config_strategy", "strategy_applied"),
        )

    class PipelineExecution(Base):
        """Complete pipeline execution records"""

        __tablename__ = "pipeline_executions"

        execution_id = Column(String(100), primary_key=True)
        session_id = Column(String(100), nullable=False)
        dataset_hash = Column(String(64), ForeignKey("dataset_characteristics.dataset_hash"), nullable=False)
        config_hash = Column(String(64), ForeignKey("project_configs.config_hash"), nullable=False)
        pipeline_stages = Column(JSONType, default=list)
        execution_time = Column(Float, default=0.0)
        final_performance = Column(JSONType, default=dict)
        trust_score = Column(Float, default=0.0)
        validation_success = Column(Boolean, default=False)
        budget_compliance_rate = Column(Float, default=0.0)
        trade_off_efficiency = Column(Float, default=0.0)
        user_satisfaction = Column(Float, nullable=True)
        success_rating = Column(Float, default=0.0)
        error_count = Column(Integer, default=0)
        recovery_attempts = Column(Integer, default=0)
        timestamp = Column(DateTime, nullable=False)
        execution_metadata = Column(JSONType, default=dict)
        created_at = Column(DateTime, default=datetime.utcnow)

        # Relationships
        dataset = relationship("DatasetCharacteristics", back_populates="executions")
        config = relationship("ProjectConfig", back_populates="executions")
        feedback_records = relationship("ExecutionFeedback", back_populates="execution")

        # Indexes for performance
        __table_args__ = (
            Index("idx_execution_success", "success_rating"),
            Index("idx_execution_timestamp", "timestamp"),
            Index("idx_execution_session", "session_id"),
            Index("idx_execution_validation", "validation_success"),
            Index("idx_execution_performance", "success_rating", "trust_score"),
        )

    class LearningPattern(Base):
        """Discovered learning patterns and optimizations"""

        __tablename__ = "learning_patterns"

        pattern_id = Column(String(100), primary_key=True)
        pattern_type = Column(String(50), nullable=False)
        dataset_context = Column(JSONType, default=dict)
        config_elements = Column(JSONType, default=dict)
        success_indicators = Column(JSONType, default=dict)
        confidence_score = Column(Float, default=0.0)
        usage_count = Column(Integer, default=0)
        last_validated = Column(DateTime, nullable=True)
        improvement_evidence = Column(JSONType, default=list)
        created_at = Column(DateTime, default=datetime.utcnow)

        # Indexes
        __table_args__ = (
            Index("idx_pattern_confidence", "confidence_score"),
            Index("idx_pattern_type", "pattern_type"),
            Index("idx_pattern_usage", "usage_count"),
        )

    class ExecutionFeedback(Base):
        """User feedback on pipeline executions"""

        __tablename__ = "execution_feedback"

        feedback_id = Column(String(100), primary_key=True)
        execution_id = Column(String(100), ForeignKey("pipeline_executions.execution_id"), nullable=False)
        user_rating = Column(Float, nullable=True)
        issues_encountered = Column(JSONType, default=list)
        suggestions = Column(Text, nullable=True)
        performance_expectation_met = Column(Boolean, nullable=True)
        would_recommend = Column(Boolean, nullable=True)
        feedback_timestamp = Column(DateTime, default=datetime.utcnow)

        # Relationships
        execution = relationship("PipelineExecution", back_populates="feedback_records")

        # Indexes
        __table_args__ = (
            Index("idx_feedback_rating", "user_rating"),
            Index("idx_feedback_timestamp", "feedback_timestamp"),
            Index("idx_feedback_execution", "execution_id"),
        )

    class SystemMetrics(Base):
        """System performance and health metrics"""

        __tablename__ = "system_metrics"

        metric_id = Column(String(100), primary_key=True)
        metric_type = Column(String(50), nullable=False)
        metric_value = Column(Float, nullable=False)
        metric_metadata = Column(JSONType, default=dict)
        timestamp = Column(DateTime, default=datetime.utcnow)

        # Indexes
        __table_args__ = (
            Index("idx_metrics_type", "metric_type"),
            Index("idx_metrics_timestamp", "timestamp"),
            Index("idx_metrics_type_time", "metric_type", "timestamp"),
        )

    class PerformanceMetric(Base):
        """Application performance metrics"""

        __tablename__ = "performance_metrics"

        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(100), nullable=True)
        metric_name = Column(String(100), nullable=False)
        metric_value = Column(Float, nullable=False)
        timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
        context = Column(JSONType, default=dict)
        created_at = Column(DateTime, default=datetime.utcnow)

        __table_args__ = (
            Index("idx_perf_session", "session_id"),
            Index("idx_perf_timestamp", "timestamp"),
            Index("idx_perf_metric", "metric_name"),
        )

    class ModelRegistry(Base):
        """Registry for trained models stored in object storage"""

        __tablename__ = "model_registry"

        model_id = Column(String(100), primary_key=True)
        session_id = Column(String(100), nullable=False)
        user_id = Column(String(100), nullable=True)
        dataset_hash = Column(String(64), nullable=False)
        model_type = Column(String(100), nullable=False)
        hyperparameters = Column(JSONType, default=dict)
        blob_path = Column(String(500), nullable=False)
        blob_url = Column(String(1000), nullable=False)
        file_checksum = Column(String(64), nullable=False)
        file_size_bytes = Column(Integer, nullable=False)
        training_metrics = Column(JSONType, default=dict)
        model_version = Column(Integer, default=1)
        framework = Column(String(50), nullable=True)
        dependencies = Column(JSONType, default=list)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        accessed_at = Column(DateTime, nullable=True)
        access_count = Column(Integer, default=0)

        # Indexes
        __table_args__ = (
            Index("idx_model_session", "session_id"),
            Index("idx_model_user", "user_id"),
            Index("idx_model_dataset", "dataset_hash"),
            Index("idx_model_type", "model_type"),
            Index("idx_model_active", "is_active"),
            Index("idx_model_lookup", "session_id", "dataset_hash", "model_type"),
        )

    class User(Base):
        """User authentication model"""

        __tablename__ = "users"

        id = Column(String(100), primary_key=True)
        email = Column(String(255), unique=True, nullable=False)
        hashed_password = Column(String(255), nullable=True)
        full_name = Column(String(255), nullable=True)
        avatar_url = Column(String(500), nullable=True)
        google_oauth_id = Column(String(255), nullable=True, unique=True)
        allow_email_notifications = Column(Boolean, default=False)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        last_login = Column(DateTime, nullable=True)

        sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")

        __table_args__ = (
            Index("idx_user_email", "email"),
            Index("idx_user_active", "is_active"),
            Index("idx_user_google", "google_oauth_id"),
        )

    class UserSession(Base):
        """User chat sessions"""

        __tablename__ = "user_sessions"

        id = Column(String(100), primary_key=True)
        user_id = Column(String(100), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
        title = Column(String(255), default="New Chat")
        dataset_path = Column(String(500), nullable=True)
        dataset_name = Column(String(255), nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        last_accessed = Column(DateTime, default=datetime.utcnow)

        user = relationship("User", back_populates="sessions")

        __table_args__ = (
            Index("idx_session_user", "user_id"),
            Index("idx_session_accessed", "last_accessed"),
        )

    class Report(Base):
        """Analysis reports generated from dataset exploration"""

        __tablename__ = "reports"

        id = Column(String(100), primary_key=True)
        session_id = Column(String(100), nullable=False)
        dataset_name = Column(String(255), nullable=False)
        status = Column(String(50), nullable=False)
        report_data = Column(JSONType, default=dict)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

        artifacts = relationship("ReportArtifact", back_populates="report", cascade="all, delete-orphan")

        __table_args__ = (
            Index("idx_reports_session", "session_id"),
            Index("idx_reports_status", "status"),
            Index("idx_reports_created", "created_at"),
        )

    class ReportArtifact(Base):
        """Artifacts associated with analysis reports"""

        __tablename__ = "report_artifacts"

        id = Column(String(100), primary_key=True)
        report_id = Column(String(100), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)
        artifact_type = Column(String(50), nullable=False)
        file_path = Column(Text, nullable=False)
        file_size_bytes = Column(Integer, nullable=True)
        artifact_metadata = Column(JSONType, default=dict)
        created_at = Column(DateTime, default=datetime.utcnow)

        report = relationship("Report", back_populates="artifacts")

        __table_args__ = (
            Index("idx_artifacts_report", "report_id"),
            Index("idx_artifacts_type", "artifact_type"),
        )

    class CancelledTask(Base):
        """Track cancelled agent tasks"""

        __tablename__ = "cancelled_tasks"

        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(100), nullable=False)
        cancelled_at = Column(DateTime, default=datetime.utcnow, nullable=False)
        reason = Column(String(255), nullable=True)

        __table_args__ = (
            Index("idx_cancelled_session", "session_id"),
            Index("idx_cancelled_timestamp", "cancelled_at"),
        )

else:
    # Fallback classes when SQLAlchemy is not available
    class Base:
        pass

    class DatasetCharacteristics:
        pass

    class ProjectConfig:
        pass

    class PipelineExecution:
        pass

    class LearningPattern:
        pass

    class ExecutionFeedback:
        pass

    class SystemMetrics:
        pass

    class ModelRegistry:
        pass

    class User:
        pass

    class Report:
        pass

    class ReportArtifact:
        pass

    class CancelledTask:
        pass
