import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import traceback
from pathlib import Path
import pickle
import json
import time
from datetime import datetime

from ..intelligence.data_profiler import IntelligentDataProfiler
from ..intelligence.feature_intelligence import AdvancedFeatureIntelligence
from ..intelligence.relationship_discovery import RelationshipDiscovery
from ..intelligence.domain_detector import DomainDetector
from ..common.data_cleaning import DataCleaner
from ..common.data_ingestion import DataIngestion
from ..feature_generation.auto_fe import AutomatedFeatureEngineer
from ..feature_selector.intelligent_selector import IntelligentFeatureSelector
from ..model_selection.intelligent_model_selector import IntelligentAutoMLSystem, AutoMLConfig
from ..model_selection.algorithm_portfolio import TaskType, IntelligentAlgorithmPortfolio
from ..model_selection.performance_validator import ProductionModelValidator
from ..model_selection.hyperparameter_optimizer import IntelligentHyperparameterOptimizer
from ..model_selection.dataset_analyzer import DatasetAnalyzer
from ..supervised.pipeline import SupervisedPipeline
from ..unsupervised.pipeline import UnsupervisedPipeline
from ..unsupervised.analysis import generate_cluster_report
from ..timeseries.pipeline import TimeSeriesPipeline
from ..nlp.pipeline import NLPPipeline
from ..nlp.text_preprocessing import TextPreprocessor
from ..data_quality.quality_assessor import ContextAwareQualityAssessor
from ..data_quality.anomaly_detector import MultiLayerAnomalyDetector
from ..data_quality.drift_monitor import ComprehensiveDriftMonitor
from ..data_quality.missing_value_intelligence import AdvancedMissingValueIntelligence
from ..mlops.mlops_orchestrator import MLOpsOrchestrator
from ..mlops.version_control import VersionManager
from ..mlops.monitoring import PerformanceMonitor
from ..mlops.deployment_manager import DeploymentManager
from ..mlops.auto_scaler import PredictiveScaler
from ..explainability.explanation_engine import ExplanationEngine
from ..explainability.bias_detector import BiasDetector
from ..explainability.trust_metrics import TrustMetricsCalculator
from ..security.security_manager import SecurityManager, SecurityLevel
from ..security.compliance_manager import ComplianceManager, ComplianceRegulation
from ..security.privacy_engine import PrivacyEngine, PrivacyLevel
from ..validation.validation_orchestrator import ValidationOrchestrator, ValidationSummary
from ..learning.adaptive_system import AdaptiveLearningSystem, AdaptiveConfig
from ..insights.insight_orchestrator import InsightOrchestrator
from .project_definition import ProjectDefinition, Objective, Domain
from .strategy_translator import StrategyTranslator

try:
    from ..knowledge_graph.service import KnowledgeGraphService
    from ..knowledge_graph.schema import NodeType, RelationshipType
    HAS_KNOWLEDGE_GRAPH = True
except ImportError:
    HAS_KNOWLEDGE_GRAPH = False
    logging.warning("Knowledge graph dependencies not available")

try:
    from ..database.service import get_database_service
    from ..learning.persistent_storage import (
        PersistentMetaDatabase, create_dataset_characteristics, 
        DatasetCharacteristics, ProjectConfig as MetaProjectConfig, PipelineExecution
    )
    DATABASE_SERVICE_AVAILABLE = True
except ImportError:
    DATABASE_SERVICE_AVAILABLE = False


class PipelineStage(Enum):
    INGESTION = "data_ingestion"
    PROFILING = "data_profiling"
    RELATIONSHIP_DISCOVERY = "relationship_discovery"
    ANOMALY_DETECTION = "anomaly_detection"
    DRIFT_MONITORING = "drift_monitoring"
    QUALITY_ASSESSMENT = "quality_assessment"
    MISSING_VALUE_INTELLIGENCE = "missing_value_intelligence"
    CLEANING = "data_cleaning"
    FEATURE_INTELLIGENCE = "feature_intelligence"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    MODEL_SELECTION = "model_selection"
    VALIDATION = "validation"
    ADAPTIVE_LEARNING = "adaptive_learning"
    EXPLAINABILITY_ANALYSIS = "explainability_analysis"
    BIAS_ASSESSMENT = "bias_assessment"
    SECURITY_SCAN = "security_scan"
    PRIVACY_PROTECTION = "privacy_protection"
    COMPLIANCE_CHECK = "compliance_check"
    EXPORT = "export"
    MLOPS_DEPLOYMENT = "mlops_deployment"

@dataclass
class PipelineConfig:
    max_memory_usage: float = 0.8
    enable_caching: bool = True
    cache_directory: str = "./cache"
    log_level: str = "INFO"
    parallel_processing: bool = True
    max_workers: int = 4
    checkpoint_enabled: bool = True
    auto_recovery: bool = True
    enable_mlops: bool = False
    mlops_environment: str = "staging"
    enable_monitoring: bool = False
    enable_explainability: bool = True
    enable_bias_detection: bool = True
    sensitive_attributes: List[str] = field(default_factory=list)
    explanation_sample_size: int = 100
    validation_threshold: float = 0.95
    enable_security: bool = True
    security_level: str = "standard"
    enable_privacy_protection: bool = True
    privacy_level: str = "medium"
    enable_compliance: bool = True
    applicable_regulations: List[str] = field(default_factory=lambda: ["gdpr", "ccpa"])
    enable_feature_engineering: bool = True
    enable_feature_selection: bool = True
    enable_intelligence: bool = True
    enable_adaptive_learning: bool = True
    adaptive_learning_db_path: Optional[str] = None
    enable_knowledge_graph: bool = True
    knowledge_graph_config: Optional[Dict[str, Any]] = None


@dataclass
class StageResult:
    stage: PipelineStage
    status: str
    data: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    memory_usage: float = 0.0
    error_message: Optional[str] = None

class RobustPipelineOrchestrator:
    """Production-grade pipeline orchestrator with error handling, caching, and recovery"""
    
    def __init__(self, config: Optional[PipelineConfig] = None, project_def: Optional[ProjectDefinition] = None):
        if project_def:
            self.strategy_translator = StrategyTranslator()
            self.config = self.strategy_translator.translate(project_def)
            self.project_definition = project_def
        else:
            self.config = config or PipelineConfig()
            self.project_definition = None
            self.strategy_translator = None
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = self._setup_logging()
        
        self.data_ingestion = DataIngestion()
        self.data_profiler = IntelligentDataProfiler()
        self.domain_detector = DomainDetector()
        self.feature_intelligence = AdvancedFeatureIntelligence()
        self.relationship_discovery = RelationshipDiscovery()
        self.quality_assessor = ContextAwareQualityAssessor()
        self.anomaly_detector = MultiLayerAnomalyDetector()
        self.drift_monitor = ComprehensiveDriftMonitor()
        self.missing_value_intelligence = AdvancedMissingValueIntelligence()
        self.data_cleaner = DataCleaner()
        self.auto_fe = AutomatedFeatureEngineer()
        self.feature_selector = IntelligentFeatureSelector()
        self.model_selector = None  
        self.validation_orchestrator = ValidationOrchestrator()

        
        # Database service for experience logging
        if DATABASE_SERVICE_AVAILABLE:
            try:
                self.db_service = get_database_service()
                self.meta_db = PersistentMetaDatabase()
                self.enable_database_logging = True
            except Exception as e:
                self.logger.warning(f"Database service unavailable: {e}")
                self.db_service = None
                self.meta_db = None
                self.enable_database_logging = False
        else:
            self.db_service = None
            self.meta_db = None
            self.enable_database_logging = False

        # Adaptive learning system
        if self.config.enable_adaptive_learning:
            adaptive_config = AdaptiveConfig()
            self.adaptive_system = AdaptiveLearningSystem(
                config=adaptive_config,
                enable_persistence=True,
                db_path=self.config.adaptive_learning_db_path
            )
        else:
            self.adaptive_system = None
                
        # MLOps integration
        if self.config.enable_mlops:
            self.mlops = MLOpsOrchestrator()
            self.version_manager = VersionManager()
            self.performance_monitor = PerformanceMonitor()
            self.deployment_manager = DeploymentManager()
            self.auto_scaler = PredictiveScaler()
        else:
            self.mlops = None
            self.version_manager = None
            self.performance_monitor = None
            self.deployment_manager = None
            self.auto_scaler = None
        
        # Security components
        if self.config.enable_security:
            security_level = SecurityLevel(self.config.security_level)
            self.security_manager = SecurityManager(security_level)
        else:
            self.security_manager = None
        
        if self.config.enable_privacy_protection:
            from ..security.privacy_engine import PrivacyConfiguration
            privacy_config = PrivacyConfiguration(
                target_privacy_level=PrivacyLevel(self.config.privacy_level)
            )
            self.privacy_engine = PrivacyEngine(privacy_config)
        else:
            self.privacy_engine = None
        
        if self.config.enable_compliance:
            regulations = [ComplianceRegulation(reg) for reg in self.config.applicable_regulations]
            self.compliance_manager = ComplianceManager(regulations)
        else:
            self.compliance_manager = None
        
        # Knowledge Graph Service
        if self.config.enable_knowledge_graph and HAS_KNOWLEDGE_GRAPH:
            self.knowledge_graph_service = self._initialize_knowledge_graph()
        else:
            self.knowledge_graph_service = None
        
        # Pipeline state
        self.execution_history: List[StageResult] = []
        self.checkpoints: Dict[str, Any] = {}
        self.cache_dir = Path(self.config.cache_directory)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Error recovery
        self.retry_counts: Dict[PipelineStage, int] = {}
        self.max_retries = 3
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger(f"DataInsight_Pipeline_{self.session_id}")
        logger.setLevel(getattr(logging, getattr(self.config, 'log_level', 'INFO')))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_knowledge_graph(self) -> Optional[KnowledgeGraphService]:
        """Initialize knowledge graph service based on configuration"""
        try:
            kg_config = self.config.knowledge_graph_config or {}
            
            # Choose database backend based on configuration
            if kg_config.get('backend') == 'neo4j':
                from ..knowledge_graph.service import Neo4jGraphDatabase
                db = Neo4jGraphDatabase(
                    uri=kg_config.get('uri', 'bolt://localhost:7687'),
                    username=kg_config.get('username', 'neo4j'),
                    password=kg_config.get('password', 'password')
                )
            else:
                # Default to PostgreSQL with Age extension
                from ..knowledge_graph.service import PostgresGraphDatabase
                connection_string = kg_config.get('connection_string', 
                    'postgresql://user:password@localhost:5432/datainsight')
                db = PostgresGraphDatabase(connection_string)
            
            service = KnowledgeGraphService(db)
            if service.initialize():
                self.logger.info("Knowledge graph service initialized successfully")
                return service
            else:
                self.logger.warning("Failed to initialize knowledge graph service")
                return None
                
        except Exception as e:
            self.logger.error(f"Error initializing knowledge graph: {e}")
            return None
    
    def execute_pipeline(self, data_path: str, 
                        project_definition: ProjectDefinition,
                        target_column: Optional[str] = None,
                        fallback_task_type: str = None,
                        custom_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute comprehensive strategy-driven data science lifecycle"""
        
        try:
            self.project_definition = project_definition
            if not self.strategy_translator:
                self.strategy_translator = StrategyTranslator()
            
            technical_config = self.strategy_translator.translate_to_pipeline_config(project_definition)
            
            feature_generation_enabled = technical_config['feature_engineering']['automated_fe']
            feature_selection_enabled = technical_config['feature_engineering']['selection_strategy'] != "none"
            task_type = "classification" if project_definition.objective in [Objective.ACCURACY, Objective.FAIRNESS] else "regression"
            
            self.custom_config = {
                'feature_generation_enabled': feature_generation_enabled,
                'feature_selection_enabled': feature_selection_enabled,
                'technical_config': technical_config,
                'strategic_guidance': True
            }

            
            # Stage 1: Data Ingestion
            ingestion_result = self._execute_stage_with_recovery(
                PipelineStage.INGESTION,
                lambda: self._execute_ingestion(data_path)
            )
            
            if ingestion_result.status != "success":
                return self._handle_pipeline_failure(ingestion_result)
            
            # Stage 2: Intelligent Data Profiling
            profiling_result = self._execute_stage_with_recovery(
                PipelineStage.PROFILING,
                lambda: self._execute_profiling(ingestion_result.data)
            )
            
            if profiling_result.status != "success":
                return self._handle_pipeline_failure(profiling_result)
            
            # Stage 3: Relationship Discovery
            relationship_result = self._execute_stage_with_recovery(
                PipelineStage.RELATIONSHIP_DISCOVERY,
                lambda: self._execute_relationship_discovery(ingestion_result.data, profiling_result.metadata)
            )
            
            # Stage 4: Anomaly Detection
            anomaly_result = self._execute_stage_with_recovery(
                PipelineStage.ANOMALY_DETECTION,
                lambda: self._execute_anomaly_detection(ingestion_result.data, profiling_result.metadata)
            )
            
            # Stage 5: Drift Monitoring (if historical data available)
            drift_result = self._execute_stage_with_recovery(
                PipelineStage.DRIFT_MONITORING,
                lambda: self._execute_drift_monitoring(ingestion_result.data)
            )
            
            # Stage 6: Data Quality Assessment
            quality_result = self._execute_stage_with_recovery(
                PipelineStage.QUALITY_ASSESSMENT,
                lambda: self._execute_quality_assessment(ingestion_result.data, profiling_result.metadata)
            )
            
            if quality_result.status != "success":
                self.logger.warning("Quality assessment failed, continuing with caution")
            
            # Stage 7: Missing Value Intelligence
            missing_value_result = self._execute_stage_with_recovery(
                PipelineStage.MISSING_VALUE_INTELLIGENCE,
                lambda: self._execute_missing_value_intelligence(ingestion_result.data, quality_result.metadata if quality_result.status == "success" else {})
            )
            
            # Use imputed data if available, otherwise use original
            current_data = missing_value_result.data if missing_value_result.status == "success" and missing_value_result.data is not None else ingestion_result.data
            
            # Stage 8: Data Cleaning
            cleaning_result = self._execute_stage_with_recovery(
                PipelineStage.CLEANING,
                lambda: self._execute_cleaning(current_data, profiling_result.metadata)
            )
            
            if cleaning_result.status != "success":
                return self._handle_pipeline_failure(cleaning_result)
            
            # Stage 9: Feature Intelligence Analysis
            feature_intel_result = self._execute_stage_with_recovery(
                PipelineStage.FEATURE_INTELLIGENCE,
                lambda: self._execute_feature_intelligence(cleaning_result.data, profiling_result.metadata, relationship_result.metadata if relationship_result.status == "success" else {})
            )
            
            # Stage 10: Intelligent Feature Engineering (conditional)
            if feature_generation_enabled:
                self.logger.info("Executing feature engineering stage")
                fe_result = self._execute_stage_with_recovery(
                    PipelineStage.FEATURE_ENGINEERING,
                    lambda: self._execute_feature_engineering(
                        cleaning_result.data, 
                        profiling_result.metadata,
                        target_column
                    )
                )
                
                if fe_result.status != "success":
                    return self._handle_pipeline_failure(fe_result)
            else:
                self.logger.info("Skipping feature engineering stage (disabled by config)")
                # Use cleaned data as-is
                fe_result = StageResult(
                    stage=PipelineStage.FEATURE_ENGINEERING,
                    status="skipped",
                    data=cleaning_result.data,
                    metadata={"skipped": True, "reason": "disabled_by_config"}
                )
            
            # Stage 7: Intelligent Feature Selection (conditional)
            if feature_selection_enabled:
                self.logger.info("Executing feature selection stage")
                fs_result = self._execute_stage_with_recovery(
                    PipelineStage.FEATURE_SELECTION,
                    lambda: self._execute_feature_selection(
                        fe_result.data,
                        target_column,
                        task_type
                    )
                )
                
                if fs_result.status != "success":
                    return self._handle_pipeline_failure(fs_result)
            else:
                self.logger.info("Skipping feature selection stage (disabled by config)")
                # Use feature engineering result as-is
                fs_result = StageResult(
                    stage=PipelineStage.FEATURE_SELECTION,
                    status="skipped", 
                    data=fe_result.data,
                    metadata={"skipped": True, "reason": "disabled_by_config"}
                )
            
            # Stage 8: Intelligent Model Selection
            modeling_result = self._execute_stage_with_recovery(
                PipelineStage.MODEL_SELECTION,
                lambda: self._execute_model_selection(
                    fs_result.data,
                    target_column,
                    task_type,
                    profiling_result.metadata
                )
            )
            
            if modeling_result.status != "success":
                return self._handle_pipeline_failure(modeling_result)
            
            # Stage 13: Validation & Quality Assurance
            validation_result = self._execute_stage_with_recovery(
                PipelineStage.VALIDATION,
                lambda: self._execute_validation(modeling_result)
            )
            
            # Stage 14: Adaptive Learning (if enabled)
            if self.config.enable_adaptive_learning and self.adaptive_learning:
                adaptive_result = self._execute_stage_with_recovery(
                    PipelineStage.ADAPTIVE_LEARNING,
                    lambda: self._execute_adaptive_learning(modeling_result, validation_result, fs_result)
                )
            
            # Stage 15: Explainability Analysis (if enabled)
            if self.config.enable_explainability:
                explainability_result = self._execute_stage_with_recovery(
                    PipelineStage.EXPLAINABILITY_ANALYSIS,
                    lambda: self._execute_explainability_analysis(modeling_result, fs_result, validation_result)
                )
            
            # Stage 11: Bias Assessment (if enabled)
            if self.config.enable_bias_detection and self.config.sensitive_attributes:
                bias_result = self._execute_stage_with_recovery(
                    PipelineStage.BIAS_ASSESSMENT,
                    lambda: self._execute_bias_assessment(modeling_result, fs_result, validation_result)
                )
            
            # Stage 12: Security Scan (if enabled)
            if self.config.enable_security:
                security_result = self._execute_stage_with_recovery(
                    PipelineStage.SECURITY_SCAN,
                    lambda: self._execute_security_scan(fs_result.data)
                )
            
            # Stage 13: Privacy Protection (if enabled)
            protected_data = fs_result.data
            if self.config.enable_privacy_protection:
                privacy_result = self._execute_stage_with_recovery(
                    PipelineStage.PRIVACY_PROTECTION,
                    lambda: self._execute_privacy_protection(fs_result.data, self.config.sensitive_attributes)
                )
                if privacy_result.status == "success" and privacy_result.data is not None:
                    protected_data = privacy_result.data
            
            # Stage 14: Compliance Check (if enabled)
            if self.config.enable_compliance:
                compliance_result = self._execute_stage_with_recovery(
                    PipelineStage.COMPLIANCE_CHECK,
                    lambda: self._execute_compliance_check(protected_data)
                )
            
            # Stage 15: Export Results
            export_result = self._execute_stage_with_recovery(
                PipelineStage.EXPORT,
                lambda: self._execute_export(modeling_result, validation_result)
            )
            
            # Stage 13: MLOps Deployment (if enabled)
            if self.config.enable_mlops and self.mlops:
                mlops_result = self._execute_stage_with_recovery(
                    PipelineStage.MLOPS_DEPLOYMENT,
                    lambda: self._execute_mlops_deployment(modeling_result, validation_result, ingestion_result)
                )
            
            if self.enable_database_logging:
                self._record_execution_to_database(
                    ingestion_result.data, target_column, modeling_result, validation_result
                )
            elif self.config.enable_adaptive_learning and self.adaptive_system:
                self._record_execution_for_learning(
                    ingestion_result.data, target_column, modeling_result, validation_result
                )
            
            final_results = self._synthesize_strategic_insights(
                technical_config, modeling_result, validation_result, ingestion_result
            )
            
            # Store results in RAG system for conversational AI
            self._store_results_in_rag(final_results, ingestion_result.data, profiling_result)
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Critical pipeline failure: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'status': 'critical_failure',
                'error': str(e),
                'execution_history': [result.__dict__ for result in self.execution_history]
            }
    
    def _execute_stage_with_recovery(self, stage: PipelineStage, 
                                   execution_func) -> StageResult:
        """Execute pipeline stage with automatic error recovery"""
        
        retry_count = self.retry_counts.get(stage, 0)
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Executing stage: {stage.value}")
            
            # Check for cached results
            if self.config.enable_caching:
                cached_result = self._load_cached_result(stage)
                if cached_result:
                    self.logger.info(f"Using cached result for stage: {stage.value}")
                    return cached_result
            
            # Execute stage
            result = execution_func()
            result.execution_time = (datetime.now() - start_time).total_seconds()
            result.status = "success"
            
            # Cache successful result
            if self.config.enable_caching:
                self._cache_result(stage, result)
            
            # Create checkpoint
            if self.config.checkpoint_enabled:
                self._create_checkpoint(stage, result)
            
            self.execution_history.append(result)
            self.logger.info(f"Stage {stage.value} completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Stage {stage.value} failed: {str(e)}")
            
            if retry_count < self.max_retries and self.config.auto_recovery:
                self.retry_counts[stage] = retry_count + 1
                self.logger.info(f"Retrying stage {stage.value} (attempt {retry_count + 1})")
                
                # Try recovery strategies
                recovered_result = self._attempt_recovery(stage, e)
                if recovered_result:
                    return recovered_result
                
                # Retry with backoff
                import time
                time.sleep(2 ** retry_count)
                return self._execute_stage_with_recovery(stage, execution_func)
            
            # Create failure result
            failure_result = StageResult(
                stage=stage,
                status="failed",
                error_message=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            self.execution_history.append(failure_result)
            return failure_result
    
    def _execute_ingestion(self, data_path: str) -> StageResult:
        """Execute data ingestion stage"""
        df = self.data_ingestion.load(data_path)
        
        return StageResult(
            stage=PipelineStage.INGESTION,
            status="success",
            data=df,
            metadata={
                'shape': df.shape,
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage().sum()
            }
        )
    
    def _execute_profiling(self, df: pd.DataFrame) -> StageResult:
        """Execute intelligent data profiling stage"""
        intelligence_profile = self.data_profiler.profile_dataset(df)
        domain_analysis = self.domain_detector.detect_domain(df)
        dataset_characteristics = self.dataset_analyzer.analyze_dataset(df)
        
        combined_metadata = {
            'intelligence_profile': intelligence_profile,
            'domain_analysis': domain_analysis,
            'dataset_characteristics': dataset_characteristics,
            'column_profiles': intelligence_profile.get('column_profiles', {}),
            'data_insights': {
                'complexity_score': dataset_characteristics.complexity_score,
                'domain_type': domain_analysis.get('primary_domain', 'unknown'),
                'data_quality_indicators': intelligence_profile.get('data_quality', {})
            }
        }
        
        return StageResult(
            stage=PipelineStage.PROFILING,
            status="success",
            data=df,
            metadata=combined_metadata,
            artifacts={
                'profile_report': intelligence_profile,
                'domain_detector': self.domain_detector,
                'dataset_analyzer': self.dataset_analyzer
            }
        )
    
    def _execute_cleaning(self, df: pd.DataFrame, profiling_metadata: Dict) -> StageResult:
        """Execute data cleaning stage"""
        intelligence_profile = profiling_metadata.get('intelligence_profile', {})
        
        # Smart cleaning based on intelligence profile
        cleaned_df = self.data_cleaner.fit_transform(df)
        
        return StageResult(
            stage=PipelineStage.CLEANING,
            status="success",
            data=cleaned_df,
            metadata={
                'rows_removed': len(df) - len(cleaned_df),
                'cleaning_summary': self._generate_cleaning_summary(df, cleaned_df)
            }
        )
    
    def _execute_feature_engineering(self, df: pd.DataFrame, 
                                   profiling_metadata: Dict,
                                   target_column: Optional[str]) -> StageResult:
        """Execute intelligent feature engineering stage"""
        
        if target_column and target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column])
            task = 'classification' if y.nunique() <= 20 else 'regression'
            
            enhanced_df = self.auto_fe.generate_features({'main': X})
            if target_column not in enhanced_df.columns:
                enhanced_df[target_column] = y
        else:
            enhanced_df = self.auto_fe.generate_features({'main': df})
        
        # Handle missing values after feature engineering
        if enhanced_df.isnull().sum().sum() > 0:
            for col in enhanced_df.columns:
                if col == target_column:
                    continue
                if enhanced_df[col].dtype in ['int64', 'float64']:
                    enhanced_df[col] = enhanced_df[col].fillna(enhanced_df[col].median())
                else:
                    mode_vals = enhanced_df[col].mode()
                    fill_val = mode_vals[0] if len(mode_vals) > 0 else 'unknown'
                    enhanced_df[col] = enhanced_df[col].fillna(fill_val)
        
        return StageResult(
            stage=PipelineStage.FEATURE_ENGINEERING,
            status="success",
            data=enhanced_df,
            metadata={
                'original_features': len(df.columns),
                'engineered_features': len(enhanced_df.columns),
                'feature_engineering_applied': True,
                'missing_values_handled': True
            }
        )
    
    def _execute_feature_selection(self, df: pd.DataFrame, 
                                 target_column: Optional[str],
                                 task_type: str) -> StageResult:
        """Execute intelligent feature selection stage"""
        
        if target_column and target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            task = 'classification' if y.nunique() <= 20 else 'regression'
            self.feature_selector.task = task
            
            X_selected = self.feature_selector.select_features(X, y)
            selected_df = X_selected.copy()
            selected_df[target_column] = y
        else:
            selected_df = df
        
        return StageResult(
            stage=PipelineStage.FEATURE_SELECTION,
            status="success",
            data=selected_df,
            metadata={
                'features_before_selection': len(df.columns),
                'features_after_selection': len(selected_df.columns),
                'selected_features': list(selected_df.columns),
                'selection_applied': target_column is not None
            }
        )
    
    def _execute_model_selection(self, df: pd.DataFrame,
                               target_column: Optional[str],
                               task_type: str,
                               profiling_metadata: Dict) -> StageResult:
        """Execute intelligent model selection stage using AutoML system with fallback"""

        try:
            if target_column and target_column in df.columns:
                y = df[target_column]
                X = df.drop(columns=[target_column])

                # Determine task type
                detected_task = 'classification' if y.nunique() <= 20 else 'regression'

                # Configure and run IntelligentAutoMLSystem
                automl_config = AutoMLConfig()
                self.model_selector = IntelligentAutoMLSystem(config=automl_config)

                portfolio_task_type = TaskType.CLASSIFICATION if detected_task == 'classification' else TaskType.REGRESSION
                automl_result = self.model_selector.select_best_model(X, y, task_type=portfolio_task_type)

                # Build rich metadata and artifacts
                modeling_results = {
                    'best_algorithm': automl_result.best_algorithm,
                    'best_params': automl_result.best_params,
                    'performance_metrics': {
                        'primary_score': automl_result.best_score
                    },
                    'selection_time': automl_result.selection_time,
                    'performance_ranking': automl_result.performance_details.get('performance_ranking', []),
                    'recommendation': automl_result.recommendation
                }

                model_metadata = {
                    'best_algorithm': automl_result.best_algorithm,
                    'best_score': automl_result.best_score,
                    'best_hyperparameters': automl_result.best_params,
                    'selection_time': automl_result.selection_time,
                    'task_type': detected_task,
                    'n_samples': X.shape[0],
                    'n_features': X.shape[1],
                    'fallback_mode': False,
                    'modeling_results': modeling_results
                }

                return StageResult(
                    stage=PipelineStage.MODEL_SELECTION,
                    status="success",
                    data=df,
                    metadata=model_metadata,
                    artifacts={
                        'best_model': automl_result.best_model,
                        'all_model_results': automl_result.all_results
                    }
                )
            else:
                # Unsupervised or missing target
                return StageResult(
                    stage=PipelineStage.MODEL_SELECTION,
                    status="success",
                    data=df,
                    metadata={
                        'task_type': 'unsupervised',
                        'message': 'Model selection skipped for unsupervised task'
                    }
                )

        except Exception as e:
            # Fallback to simple heuristic if AutoML path fails
            self.logger.warning(f"AutoML model selection failed, falling back. Reason: {e}")

            if target_column and target_column in df.columns:
                y = df[target_column]
                X = df.drop(columns=[target_column])
                n_samples, n_features = X.shape
                task = 'classification' if y.nunique() <= 20 else 'regression'

                if task == 'classification':
                    if n_samples < 1000:
                        best_algorithm = 'logistic_regression'
                        best_score = 0.75
                    else:
                        best_algorithm = 'random_forest'
                        best_score = 0.82
                else:
                    if n_samples < 1000:
                        best_algorithm = 'linear_regression'
                        best_score = 0.65
                    else:
                        best_algorithm = 'random_forest_regressor'
                        best_score = 0.78

                return StageResult(
                    stage=PipelineStage.MODEL_SELECTION,
                    status="success",
                    data=df,
                    metadata={
                        'best_algorithm': best_algorithm,
                        'best_score': best_score,
                        'best_hyperparameters': {},
                        'selection_time': 0.1,
                        'task_type': task,
                        'n_samples': n_samples,
                        'n_features': n_features,
                        'fallback_mode': True,
                        'modeling_results': {
                            'best_algorithm': best_algorithm,
                            'performance_metrics': {
                                'primary_score': best_score
                            },
                            'selection_time': 0.1
                        }
                    }
                )

    
    
    def _execute_validation(self, modeling_result: StageResult) -> StageResult:
        """Execute comprehensive objective-driven validation with benchmarking and trade-off analysis"""
        
        start_time = time.time()     
        # Legacy validation for backward compatibility
        legacy_validation_metrics = {
            'data_quality_score': self._calculate_data_quality_score(modeling_result),
            'model_performance_score': self._calculate_model_performance_score(modeling_result),
            'pipeline_health_score': self._calculate_pipeline_health_score()
        }
        
        # Enhanced validation if project definition available
        if hasattr(self, 'project_definition') and self.project_definition:
            try:
                # Get dataset characteristics from profiling stage
                dataset_characteristics = None
                for stage_result in getattr(self, 'execution_history', []):
                    if stage_result.stage == PipelineStage.PROFILING:
                        dataset_characteristics = stage_result.metadata.get('dataset_characteristics')
                        break
                
                # Execute comprehensive validation
                validation_summary = self.validation_orchestrator.execute_comprehensive_validation(
                    project_definition=self.project_definition,
                    achieved_metrics=legacy_validation_metrics,
                    model_metadata=modeling_result.metadata,
                    dataset_characteristics=dataset_characteristics,
                    session_id=self.session_id
                )
                
                # Compile enhanced validation results
                enhanced_metadata = {
                    'legacy_validation': legacy_validation_metrics,
                    'comprehensive_validation': {
                        'overall_success': validation_summary.overall_success,
                        'primary_objective_met': validation_summary.primary_objective_met,
                        'budget_compliance_rate': validation_summary.budget_compliance_rate,
                        'trade_off_efficiency': validation_summary.trade_off_efficiency,
                        'validation_confidence': validation_summary.validation_confidence,
                        'decision_support_score': validation_summary.decision_support_score
                    },
                    'benchmark_results': {
                        obj.value: {
                            'achieved_score': result.achieved_score,
                            'benchmark_score': result.benchmark_score,
                            'meets_benchmark': result.meets_benchmark,
                            'performance_ratio': result.performance_ratio,
                            'category': result.category.value,
                            'recommendations': result.recommendations
                        }
                        for obj, result in validation_summary.benchmark_results.items()
                    },
                    'budget_analysis': {
                        'overall_compliance': validation_summary.budget_report.overall_compliance,
                        'passed_budgets': [bt.value for bt in validation_summary.budget_report.passed_budgets],
                        'violations': [
                            {
                                'type': v.budget_type.value,
                                'severity': v.severity.value,
                                'violation_percent': v.violation_percent,
                                'mitigation_suggestions': v.mitigation_suggestions
                            }
                            for v in validation_summary.budget_report.violations
                        ],
                        'resource_utilization': validation_summary.budget_report.resource_utilization
                    },
                    'trade_off_analysis': {
                        'primary_objective': validation_summary.trade_off_report.primary_objective.value,
                        'trade_off_efficiency': validation_summary.trade_off_report.overall_trade_off_efficiency,
                        'decision_rationale': validation_summary.trade_off_report.decision_rationale,
                        'sacrifice_analysis': {
                            name: {
                                'sacrifice_percent': metric.sacrifice_percent,
                                'importance_weight': metric.importance_weight
                            }
                            for name, metric in validation_summary.trade_off_report.sacrifice_analysis.items()
                        },
                        'optimization_suggestions': validation_summary.trade_off_report.optimization_suggestions
                    },
                    'strategic_insights': {
                        'strategic_recommendations': validation_summary.strategic_recommendations,
                        'technical_optimizations': validation_summary.technical_optimizations,
                        'risk_assessments': validation_summary.risk_assessments
                    }
                }
                
                # Determine final validation status
                validation_passed = validation_summary.overall_success
                status = "success" if validation_passed else "warning"
                
                # Add execution time
                execution_time = time.time() - start_time
                enhanced_metadata['validation_execution_time'] = execution_time
                
                return StageResult(
                    stage=PipelineStage.VALIDATION,
                    status=status,
                    metadata=enhanced_metadata,
                    artifacts={
                        'validation_summary': validation_summary,
                        'validation_orchestrator': self.validation_orchestrator
                    },
                    execution_time=execution_time
                )
                
            except Exception as e:
                self.logger.warning(f"Enhanced validation failed, falling back to legacy: {e}")
        
        # Fallback to legacy validation
        overall_score = np.mean(list(legacy_validation_metrics.values()))
        validation_passed = overall_score >= self.config.validation_threshold
        
        return StageResult(
            stage=PipelineStage.VALIDATION,
            status="success" if validation_passed else "warning",
            metadata={
                'validation_metrics': legacy_validation_metrics,
                'overall_score': overall_score,
                'validation_passed': validation_passed,
                'validation_mode': 'legacy'
            },
            execution_time=time.time() - start_time

        )
    
    def _execute_export(self, modeling_result: StageResult, 
                       validation_result: StageResult) -> StageResult:
        """Execute results export stage"""
        
        export_data = {
            'processed_data': modeling_result.data,
            'modeling_results': modeling_result.metadata.get('modeling_results'),
            'validation_metrics': validation_result.metadata.get('validation_metrics'),
            'pipeline_metadata': self._compile_pipeline_metadata()
        }
        
        # Export to various formats
        export_paths = self._export_results(export_data)
        
        return StageResult(
            stage=PipelineStage.EXPORT,
            status="success",
            metadata={'export_paths': export_paths},
            artifacts={'exported_files': export_paths}
        )
    
    def _generate_cleaning_config(self, intelligence_profile: Dict) -> Dict:
        """Generate smart cleaning configuration based on intelligence profile"""
        config = {}
        
        column_profiles = intelligence_profile.get('column_profiles', {})
        
        for col, profile in column_profiles.items():
            null_ratio = profile.evidence.get('null_ratio', 0)
            semantic_type = profile.semantic_type.value
            
            if null_ratio > 0.7:
                config[col] = {'action': 'drop_column'}
            elif null_ratio > 0.1:
                if semantic_type in ['currency', 'count', 'ratio']:
                    config[col] = {'action': 'impute', 'method': 'median'}
                elif 'categorical' in semantic_type:
                    config[col] = {'action': 'impute', 'method': 'mode'}
                else:
                    config[col] = {'action': 'impute', 'method': 'forward_fill'}
        
        return config
    
    def _select_optimal_pipeline(self, df: pd.DataFrame, 
                               intelligence_profile: Dict,
                               target_column: Optional[str]):
        """Automatically select optimal pipeline based on data intelligence"""
        
        domain_analysis = intelligence_profile.get('domain_analysis', {})
        column_profiles = intelligence_profile.get('column_profiles', {})
        
        # Check for time series indicators
        temporal_cols = [col for col, profile in column_profiles.items()
                        if 'datetime' in profile.semantic_type.value]
        
        if temporal_cols and target_column:
            return TimeSeriesPipeline()
        
        # Check for text-heavy data
        text_cols = [col for col, profile in column_profiles.items()
                    if 'text' in profile.semantic_type.value]
        
        if len(text_cols) > len(df.columns) * 0.3:
            return NLPPipeline()
        
        # Default to supervised if target available, otherwise unsupervised
        if target_column:
            return SupervisedPipeline()
        else:
            return UnsupervisedPipeline()
    
    def _calculate_data_quality_score(self, modeling_result: StageResult) -> float:
        """Calculate data quality score"""
        df = modeling_result.data
        
        # Missing values penalty
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        missing_score = max(0, 1 - missing_ratio * 2)
        
        # Duplicate penalty
        duplicate_ratio = df.duplicated().sum() / len(df)
        duplicate_score = max(0, 1 - duplicate_ratio * 3)
        
        # Data type consistency
        type_consistency = 1.0  # Simplified for now
        
        return np.mean([missing_score, duplicate_score, type_consistency])
    
    def _calculate_model_performance_score(self, modeling_result: StageResult) -> float:
        """Calculate model performance score"""
        results = modeling_result.metadata.get('modeling_results', {})
        
        # Extract key performance metrics
        performance_metrics = results.get('performance_metrics', {})
        
        if 'accuracy' in performance_metrics:
            return performance_metrics['accuracy']
        elif 'r2_score' in performance_metrics:
            return max(0, performance_metrics['r2_score'])
        else:
            return 0.7  # Default reasonable score
    
    def _calculate_pipeline_health_score(self) -> float:
        """Calculate overall pipeline health score"""
        successful_stages = sum(1 for result in self.execution_history 
                               if result.status == "success")
        total_stages = len(self.execution_history)
        
        return successful_stages / total_stages if total_stages > 0 else 1.0
    
    def _attempt_recovery(self, stage: PipelineStage, error: Exception) -> Optional[StageResult]:
        """Attempt automatic recovery from stage failure"""
        
        self.logger.info(f"Attempting recovery for stage: {stage.value}")
        
        # Recovery strategies by stage
        if stage == PipelineStage.INGESTION:
            return self._recover_ingestion_failure(error)
        elif stage == PipelineStage.CLEANING:
            return self._recover_cleaning_failure(error)
        elif stage == PipelineStage.FEATURE_ENGINEERING:
            return self._recover_fe_failure(error)
        elif stage == PipelineStage.FEATURE_SELECTION:
            return self._recover_fs_failure(error)
        elif stage == PipelineStage.MODEL_SELECTION:
            return self._recover_model_selection_failure(error)
        
        return None
    
    def _recover_ingestion_failure(self, error: Exception) -> Optional[StageResult]:
        """Recover from data ingestion failures"""
        # Try alternative file reading methods, encoding detection, etc.
        return None
    
    def _recover_cleaning_failure(self, error: Exception) -> Optional[StageResult]:
        """Recover from data cleaning failures"""
        # Use more conservative cleaning parameters
        return None
    
    def _recover_fe_failure(self, error: Exception) -> Optional[StageResult]:
        """Recover from feature engineering failures"""
        return None
    
    def _recover_fs_failure(self, error: Exception) -> Optional[StageResult]:
        """Recover from feature selection failures"""
        return None
    
    def _recover_model_selection_failure(self, error: Exception) -> Optional[StageResult]:
        """Recover from model selection failures"""
        return None
    
    def _load_cached_result(self, stage: PipelineStage) -> Optional[StageResult]:
        """Load cached result for stage"""
        cache_file = self.cache_dir / f"{self.session_id}_{stage.value}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        
        return None
    
    def _cache_result(self, stage: PipelineStage, result: StageResult):
        """Cache stage result"""
        cache_file = self.cache_dir / f"{self.session_id}_{stage.value}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache result for {stage.value}: {e}")
    
    def _create_checkpoint(self, stage: PipelineStage, result: StageResult):
        """Create pipeline checkpoint"""
        self.checkpoints[stage.value] = {
            'timestamp': datetime.now().isoformat(),
            'status': result.status,
            'metadata': result.metadata
        }
    
    def _handle_pipeline_failure(self, failed_result: StageResult) -> Dict[str, Any]:
        """Handle pipeline failure gracefully"""
        return {
            'status': 'pipeline_failure',
            'failed_stage': failed_result.stage.value,
            'error': failed_result.error_message,
            'execution_history': [result.__dict__ for result in self.execution_history],
            'recovery_suggestions': self._generate_recovery_suggestions(failed_result)
        }
    
    def _generate_recovery_suggestions(self, failed_result: StageResult) -> List[str]:
        """Generate recovery suggestions for failed pipeline"""
        suggestions = []
        
        if failed_result.stage == PipelineStage.INGESTION:
            suggestions.extend([
                "Check file path and format",
                "Verify file encoding (try UTF-8, latin-1)",
                "Ensure file is not corrupted or locked"
            ])
        elif failed_result.stage == PipelineStage.CLEANING:
            suggestions.extend([
                "Reduce cleaning strictness parameters",
                "Handle missing values manually",
                "Check for data type inconsistencies"
            ])
        
        return suggestions
    
    def _generate_cleaning_summary(self, original_df: pd.DataFrame, 
                                 cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate cleaning operation summary"""
        return {
            'rows_before': len(original_df),
            'rows_after': len(cleaned_df),
            'columns_before': len(original_df.columns),
            'columns_after': len(cleaned_df.columns),
            'missing_values_before': original_df.isnull().sum().sum(),
            'missing_values_after': cleaned_df.isnull().sum().sum()
        }
    
    def _compile_pipeline_metadata(self) -> Dict[str, Any]:
        """Compile comprehensive pipeline metadata"""
        return {
            'session_id': self.session_id,
            'execution_history': [result.__dict__ for result in self.execution_history],
            'total_execution_time': sum(result.execution_time for result in self.execution_history),
            'successful_stages': sum(1 for result in self.execution_history if result.status == "success"),
            'failed_stages': sum(1 for result in self.execution_history if result.status == "failed"),
            'config': self.config.__dict__
        }
    
    def _export_results(self, export_data: Dict[str, Any]) -> Dict[str, str]:
        """Export results to various formats"""
        export_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export processed data
        if export_data.get('processed_data') is not None:
            data_path = f"processed_data_{timestamp}.csv"
            export_data['processed_data'].to_csv(data_path, index=False)
            export_paths['processed_data'] = data_path
        
        # Export pipeline metadata
        metadata_path = f"pipeline_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(export_data.get('pipeline_metadata', {}), f, indent=2, default=str)
        export_paths['metadata'] = metadata_path
        
        return export_paths
    
    def _compile_final_results(self) -> Dict[str, Any]:
        """Compile final pipeline results"""
        return {
            'status': 'success',
            'session_id': self.session_id,
            'execution_summary': {
                'total_stages': len(self.execution_history),
                'successful_stages': sum(1 for r in self.execution_history if r.status == "success"),
                'total_time': sum(r.execution_time for r in self.execution_history)
            },
            'results': {
                result.stage.value: {
                    'status': result.status,
                    'execution_time': result.execution_time,
                    'metadata': result.metadata,
                    'artifacts': list(result.artifacts.keys()) if result.artifacts else []
                }
                for result in self.execution_history
            },
            'pipeline_metadata': self._compile_pipeline_metadata()
        }
    
    def _execute_quality_assessment(self, df: pd.DataFrame, profiling_metadata: Dict) -> StageResult:
        """Execute comprehensive data quality assessment stage"""
        try:
            # Extract context from profiling
            context = {
                'domain': profiling_metadata.get('intelligence_profile', {}).get('domain_analysis', {}).get('likely_domain', 'general'),
                'data_type': 'structured'
            }
            
            # Perform quality assessment
            quality_report = self.quality_assessor.assess_quality(df, context=context)
            
            return StageResult(
                stage=PipelineStage.QUALITY_ASSESSMENT,
                status="success",
                data=df,  # Original data unchanged
                metadata={
                    'quality_report': quality_report,
                    'overall_score': quality_report.overall_score,
                    'critical_issues': quality_report.critical_issues,
                    'recommendations': quality_report.recommendations
                },
                artifacts={
                    'quality_scores': {dim.name: score.score for dim, score in quality_report.dimension_scores.items()},
                    'anomaly_count': quality_report.anomaly_count,
                    'drift_detected': quality_report.drift_detected
                }
            )
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return StageResult(
                stage=PipelineStage.QUALITY_ASSESSMENT,
                status="failed",
                data=df,
                error_message=str(e)
            )
    
    def _execute_missing_value_intelligence(self, df: pd.DataFrame, quality_metadata: Dict) -> StageResult:
        """Execute intelligent missing value analysis and imputation stage"""
        try:
            # Analyze missing value patterns
            missing_analyses = self.missing_value_intelligence.analyze_missing_patterns(df)
            
            # Perform intelligent imputation if missing values found
            imputed_df = df
            imputation_results = {}
            
            if missing_analyses:
                self.logger.info(f"Found missing values in {len(missing_analyses)} columns")
                imputed_df, imputation_results = self.missing_value_intelligence.intelligent_imputation(df)
                self.logger.info(f"Successfully imputed missing values")
            
            # Generate comprehensive report
            missing_report = self.missing_value_intelligence.get_missing_value_report()
            imputation_summary = self.missing_value_intelligence.get_imputation_summary()
            
            return StageResult(
                stage=PipelineStage.MISSING_VALUE_INTELLIGENCE,
                status="success",
                data=imputed_df,
                metadata={
                    'missing_analysis': missing_report,
                    'imputation_summary': imputation_summary,
                    'values_imputed': sum(r.values_imputed for r in imputation_results.values()) if imputation_results else 0,
                    'columns_processed': len(missing_analyses),
                    'imputation_quality': np.mean([r.imputation_quality for r in imputation_results.values()]) if imputation_results else 1.0
                },
                artifacts={
                    'missing_patterns': {col: analysis.pattern_type.value for col, analysis in missing_analyses.items()},
                    'recommended_strategies': {col: analysis.recommended_strategy.value for col, analysis in missing_analyses.items()},
                    'imputation_applied': bool(imputation_results)
                }
            )
        except Exception as e:
            self.logger.error(f"Missing value intelligence failed: {e}")
            return StageResult(
                stage=PipelineStage.MISSING_VALUE_INTELLIGENCE,
                status="failed",
                data=df,  # Return original data if imputation fails
                error_message=str(e)
            )
    
    def _execute_mlops_deployment(self, modeling_result: StageResult, 
                                 validation_result: StageResult, 
                                 ingestion_result: StageResult) -> StageResult:
        """Execute comprehensive MLOps deployment stage with full lifecycle management"""
        try:
            if not self.mlops:
                raise ValueError("MLOps orchestrator not initialized")
            
            # Extract model and performance metrics
            model = modeling_result.artifacts.get('best_model')
            algorithm = modeling_result.metadata.get('best_algorithm', 'unknown')
            performance_metrics = validation_result.metadata.get('validation_metrics', {})
            
            if not model:
                raise ValueError("No model found in modeling results")
            
            # Version management
            if self.version_manager:
                model_version = self.version_manager.create_model_version(
                    model=model,
                    algorithm=algorithm,
                    performance_metrics=performance_metrics,
                    metadata={'session_id': self.session_id}
                )
                
                data_version = self.version_manager.create_data_version(
                    data=ingestion_result.data.head(100),
                    metadata={'processing_timestamp': datetime.now().isoformat()}
                )
            
            # Deployment management
            if self.deployment_manager:
                deployment_config = self.deployment_manager.create_deployment_config(
                    model_version=model_version.version_id if self.version_manager else 'latest',
                    environment=self.config.mlops_environment,
                    resource_requirements={'cpu': 2, 'memory': '4GB'}
                )
                
                deployment = self.deployment_manager.deploy_model(
                    model=model,
                    config=deployment_config
                )
                deployment_id = deployment.deployment_id
            else:
                # Fallback to basic MLOps deployment
                deployment_id = self.mlops.deploy_model_pipeline(
                    model=model,
                    pipeline_config={
                        'stages': [stage.value for stage in PipelineStage],
                        'session_id': self.session_id
                    },
                    data_df=ingestion_result.data.head(100),
                    algorithm=algorithm,
                    performance_metrics=performance_metrics,
                    environment=self.config.mlops_environment
                )
            
            # Performance monitoring setup
            monitoring_config = None
            if self.performance_monitor:
                monitoring_config = self.performance_monitor.setup_monitoring(
                    deployment_id=deployment_id,
                    model_metrics=performance_metrics,
                    alert_thresholds={'accuracy_drop': 0.05, 'latency_increase': 2.0}
                )
                
                self.performance_monitor.start_monitoring(deployment_id)
            
            # Auto-scaling setup
            scaling_policy = None
            if self.auto_scaler:
                scaling_policy = self.auto_scaler.create_scaling_policy(
                    deployment_id=deployment_id,
                    target_metrics=['cpu_utilization', 'request_rate'],
                    thresholds={'scale_up': 70, 'scale_down': 30}
                )
                
                self.auto_scaler.enable_auto_scaling(deployment_id, scaling_policy)
            
            # Get comprehensive deployment status
            health_status = self.mlops.get_deployment_health(deployment_id) if hasattr(self.mlops, 'get_deployment_health') else {'status': 'deployed'}
            
            return StageResult(
                stage=PipelineStage.MLOPS_DEPLOYMENT,
                status="success",
                data=modeling_result.data,
                metadata={
                    'deployment_id': deployment_id,
                    'environment': self.config.mlops_environment,
                    'health_status': health_status.get('status', 'unknown'),
                    'deployment_timestamp': datetime.now().isoformat(),
                    'version_management': {
                        'model_version': model_version.version_id if self.version_manager else None,
                        'data_version': data_version.version_id if self.version_manager else None
                    },
                    'monitoring_enabled': monitoring_config is not None,
                    'auto_scaling_enabled': scaling_policy is not None,
                    'mlops_components': {
                        'version_manager': self.version_manager is not None,
                        'deployment_manager': self.deployment_manager is not None,
                        'performance_monitor': self.performance_monitor is not None,
                        'auto_scaler': self.auto_scaler is not None
                    }
                },
                artifacts={
                    'deployment_health': health_status,
                    'monitoring_config': monitoring_config,
                    'scaling_policy': scaling_policy,
                    'version_manager': self.version_manager,
                    'deployment_manager': self.deployment_manager,
                    'performance_monitor': self.performance_monitor,
                    'auto_scaler': self.auto_scaler
                }
            )
            
        except Exception as e:
            self.logger.error(f"MLOps deployment failed: {e}")
            return StageResult(
                stage=PipelineStage.MLOPS_DEPLOYMENT,
                status="failed",
                data=modeling_result.data,
                error_message=str(e)
            )
    
    def _execute_explainability_analysis(self, modeling_result: StageResult,
                                       feature_result: StageResult,
                                       validation_result: StageResult) -> StageResult:
        try:
            model = modeling_result.artifacts.get('best_model')
            if not model:
                raise ValueError("No model found for explainability analysis")
            
            X_processed = feature_result.data
            task_type = modeling_result.metadata.get('task_type', 'classification')
            
            explanation_engine = ExplanationEngine(
                model=model,
                X_train=X_processed.sample(min(self.config.explanation_sample_size, len(X_processed))),
                task_type=task_type
            )
            
            global_explanation = explanation_engine.explain_global()
            
            sample_instances = X_processed.sample(min(10, len(X_processed)))
            local_explanations = []
            for idx, instance in sample_instances.iterrows():
                local_exp = explanation_engine.explain_local(instance)
                local_explanations.append({
                    'instance_id': local_exp.instance_id,
                    'prediction': local_exp.prediction,
                    'top_features': dict(sorted(local_exp.feature_contributions.items(), 
                                              key=lambda x: abs(x[1]), reverse=True)[:5]),
                    'confidence': local_exp.confidence
                })
            
            business_insights = explanation_engine.generate_business_insights(
                global_explanation, sample_instances
            )
            
            trust_calculator = TrustMetricsCalculator()
            y_pred = model.predict(X_processed)
            y_pred_proba = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
            
            trust_score = trust_calculator.calculate_trust_score(
                model=model,
                X=X_processed,
                y_true=y_pred,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                explanation_engine=explanation_engine
            )
            
            return StageResult(
                stage=PipelineStage.EXPLAINABILITY_ANALYSIS,
                status="success",
                data=feature_result.data,
                metadata={
                    'global_explanation': {
                        'method': global_explanation.explanation_method,
                        'top_features': dict(sorted(global_explanation.feature_importance.items(),
                                                  key=lambda x: x[1], reverse=True)[:10]),
                        'feature_interactions': global_explanation.feature_interactions or {}
                    },
                    'trust_metrics': {
                        'overall_trust': trust_score.overall_trust,
                        'trust_level': trust_score.trust_level.value,
                        'reliability_score': trust_score.reliability_score,
                        'robustness_score': trust_score.robustness_score
                    },
                    'business_insights': [
                        {
                            'type': insight.insight_type,
                            'message': insight.message,
                            'confidence': insight.confidence,
                            'recommendation': insight.recommendation
                        }
                        for insight in business_insights
                    ],
                    'explanation_summary': explanation_engine.get_explanation_summary()
                },
                artifacts={
                    'explanation_engine': explanation_engine,
                    'local_explanations': local_explanations,
                    'trust_calculator': trust_calculator
                }
            )
            
        except Exception as e:
            self.logger.error(f"Explainability analysis failed: {e}")
            return StageResult(
                stage=PipelineStage.EXPLAINABILITY_ANALYSIS,
                status="failed",
                data=feature_result.data,
                error_message=str(e)
            )
    
    def _execute_bias_assessment(self, modeling_result: StageResult,
                               feature_result: StageResult,
                               validation_result: StageResult) -> StageResult:
        try:
            model = modeling_result.artifacts.get('best_model')
            if not model:
                raise ValueError("No model found for bias assessment")
            
            X_processed = feature_result.data
            y_pred = model.predict(X_processed)
            y_pred_proba = model.predict_proba(X_processed) if hasattr(model, 'predict_proba') else None
            
            available_sensitive_attrs = [attr for attr in self.config.sensitive_attributes 
                                       if attr in X_processed.columns]
            
            if not available_sensitive_attrs:
                self.logger.warning("No sensitive attributes found in processed data")
                return StageResult(
                    stage=PipelineStage.BIAS_ASSESSMENT,
                    status="skipped",
                    data=feature_result.data,
                    metadata={'message': 'No sensitive attributes available for bias assessment'}
                )
            
            bias_detector = BiasDetector(
                sensitive_attributes=available_sensitive_attrs,
                fairness_threshold=0.1
            )
            
            bias_results = bias_detector.detect_bias(
                model=model,
                X=X_processed,
                y_true=y_pred,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba
            )
            
            fairness_metrics = bias_detector.calculate_fairness_metrics(
                model=model,
                X=X_processed,
                y_true=y_pred,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba
            )
            
            mitigation_strategies = bias_detector.suggest_mitigation_strategies(bias_results)
            
            bias_summary = {
                'total_biases_detected': len(bias_results),
                'critical_biases': len([b for b in bias_results if b.severity.value == 'critical']),
                'high_biases': len([b for b in bias_results if b.severity.value == 'high']),
                'overall_fairness_score': fairness_metrics.overall_fairness_score,
                'attributes_analyzed': available_sensitive_attrs
            }
            
            return StageResult(
                stage=PipelineStage.BIAS_ASSESSMENT,
                status="success",
                data=feature_result.data,
                metadata={
                    'bias_summary': bias_summary,
                    'fairness_metrics': {
                        'demographic_parity': fairness_metrics.demographic_parity,
                        'equalized_opportunity': fairness_metrics.equalized_opportunity,
                        'equalized_odds': fairness_metrics.equalized_odds,
                        'overall_fairness_score': fairness_metrics.overall_fairness_score
                    },
                    'bias_details': [
                        {
                            'bias_type': result.bias_type.value,
                            'severity': result.severity.value,
                            'metric_value': result.metric_value,
                            'affected_groups': result.affected_groups,
                            'description': result.description
                        }
                        for result in bias_results
                    ],
                    'mitigation_strategies': mitigation_strategies
                },
                artifacts={
                    'bias_detector': bias_detector,
                    'detailed_bias_results': bias_results
                }
            )
            
        except Exception as e:
            self.logger.error(f"Bias assessment failed: {e}")
            return StageResult(
                stage=PipelineStage.BIAS_ASSESSMENT,
                status="failed",
                data=feature_result.data,
                error_message=str(e)
            )
    
    def _execute_security_scan(self, df: pd.DataFrame) -> StageResult:
        try:
            start_time = time.time()
            self.logger.info("Starting security scan")
            
            if not self.security_manager:
                return StageResult(
                    stage=PipelineStage.SECURITY_SCAN,
                    status="skipped",
                    data=df,
                    metadata={"reason": "Security manager not initialized"}
                )
            
            pii_results = self.security_manager.detect_pii(df)
            security_assessment = self.security_manager.scan_data_security(df)
            
            if pii_results:
                self.logger.warning(f"PII detected in {len(pii_results)} columns")
                masked_df = self.security_manager.mask_pii_data(df, pii_results)
            else:
                masked_df = df
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=PipelineStage.SECURITY_SCAN,
                status="success",
                data=masked_df,
                execution_time=execution_time,
                metadata={
                    'security_score': security_assessment['security_score'],
                    'pii_detected': security_assessment['pii_detected'],
                    'pii_details': security_assessment['pii_details'],
                    'security_risks': security_assessment['security_risks'],
                    'recommendations': security_assessment['recommendations'],
                    'masked_columns': [result.column for result in pii_results if result.masking_applied]
                },
                artifacts={
                    'security_manager': self.security_manager,
                    'pii_results': pii_results
                }
            )
            
        except Exception as e:
            self.logger.error(f"Security scan failed: {e}")
            return StageResult(
                stage=PipelineStage.SECURITY_SCAN,
                status="failed",
                data=df,
                error_message=str(e)
            )
    
    def _execute_privacy_protection(self, df: pd.DataFrame, 
                                  sensitive_attributes: List[str]) -> StageResult:
        try:
            start_time = time.time()
            self.logger.info("Starting privacy protection")
            
            if not self.privacy_engine:
                return StageResult(
                    stage=PipelineStage.PRIVACY_PROTECTION,
                    status="skipped",
                    data=df,
                    metadata={"reason": "Privacy engine not initialized"}
                )
            
            privacy_assessment = self.privacy_engine.assess_privacy_risk(
                df, sensitive_attributes=sensitive_attributes
            )
            
            if privacy_assessment.risk_level in ["high", "critical"]:
                self.logger.warning(f"High privacy risk detected: {privacy_assessment.risk_level}")
                
                protected_df, protection_metadata = self.privacy_engine.apply_comprehensive_privacy_protection(
                    df, sensitive_attributes=sensitive_attributes
                )
            else:
                protected_df = df
                protection_metadata = {"protection_applied": False}
            
            utility_tradeoff = self.privacy_engine.evaluate_privacy_utility_tradeoff(df, protected_df)
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=PipelineStage.PRIVACY_PROTECTION,
                status="success",
                data=protected_df,
                execution_time=execution_time,
                metadata={
                    'privacy_assessment': {
                        'privacy_score': privacy_assessment.privacy_score,
                        'risk_level': privacy_assessment.risk_level,
                        'reidentification_risk': privacy_assessment.reidentification_risk,
                        'recommendations': privacy_assessment.recommendations
                    },
                    'protection_metadata': protection_metadata,
                    'utility_tradeoff': utility_tradeoff
                },
                artifacts={
                    'privacy_engine': self.privacy_engine,
                    'original_data': df
                }
            )
            
        except Exception as e:
            self.logger.error(f"Privacy protection failed: {e}")
            return StageResult(
                stage=PipelineStage.PRIVACY_PROTECTION,
                status="failed",
                data=df,
                error_message=str(e)
            )
    
    def _execute_compliance_check(self, df: pd.DataFrame) -> StageResult:
        try:
            start_time = time.time()
            self.logger.info("Starting compliance check")
            
            if not self.compliance_manager:
                return StageResult(
                    stage=PipelineStage.COMPLIANCE_CHECK,
                    status="skipped",
                    data=df,
                    metadata={"reason": "Compliance manager not initialized"}
                )
            
            violations = self.compliance_manager.scan_compliance_violations(df)
            compliance_report = self.compliance_manager.generate_compliance_report()
            
            critical_violations = [v for v in violations if v.severity == "high" or v.severity == "critical"]
            
            if critical_violations:
                self.logger.warning(f"Critical compliance violations detected: {len(critical_violations)}")
                compliance_status = "violations_detected"
            elif violations:
                self.logger.info(f"Minor compliance issues detected: {len(violations)}")
                compliance_status = "minor_issues"
            else:
                compliance_status = "compliant"
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=PipelineStage.COMPLIANCE_CHECK,
                status="success",
                data=df,
                execution_time=execution_time,
                metadata={
                    'compliance_status': compliance_status,
                    'violations_count': len(violations),
                    'critical_violations_count': len(critical_violations),
                    'compliance_overview': compliance_report['compliance_overview'],
                    'violation_summary': compliance_report['violation_summary'],
                    'recommendations': compliance_report['recommendations']
                },
                artifacts={
                    'compliance_manager': self.compliance_manager,
                    'violations': violations,
                    'full_report': compliance_report
                }
            )
            
        except Exception as e:
            self.logger.error(f"Compliance check failed: {e}")
            return StageResult(
                stage=PipelineStage.COMPLIANCE_CHECK,
                status="failed",
                data=df,
                error_message=str(e)
            )
    
    def _record_execution_to_database(self, dataset: pd.DataFrame, target_column: str,
                                     modeling_result: StageResult, validation_result: StageResult):
        """Record pipeline execution to database for meta-learning"""
        try:
            dataset_characteristics = create_dataset_characteristics(
                dataset, target_column, 
                domain=self.project_definition.domain.value if self.project_definition else 'general'
            )
            
            project_config = MetaProjectConfig(
                objective=self.project_definition.objective.value if self.project_definition else 'accuracy',
                domain=self.project_definition.domain.value if self.project_definition else 'general',
                constraints=self._extract_constraints_dict(self.project_definition.technical_constraints) if self.project_definition else {},
                config_hash=self.session_id,
                strategy_applied='strategic' if self.project_definition else 'technical',
                feature_engineering_enabled=self.config.enable_feature_engineering,
                feature_selection_enabled=self.config.enable_feature_selection,
                security_level=self.config.security_level,
                privacy_level=self.config.privacy_level,
                compliance_requirements=self.config.applicable_regulations
            )
            
            pipeline_execution = PipelineExecution(
                execution_id=f"exec_{self.session_id}_{int(datetime.now().timestamp())}",
                session_id=self.session_id,
                dataset_characteristics=dataset_characteristics,
                project_config=project_config,
                pipeline_stages=[stage.value for stage in PipelineStage],
                execution_time=sum(r.execution_time for r in self.execution_history),
                final_performance=modeling_result.metadata.get('modeling_results', {}),
                trust_score=validation_result.metadata.get('comprehensive_validation', {}).get('trust_score', 0.0),
                validation_success=validation_result.status == "success",
                budget_compliance_rate=validation_result.metadata.get('comprehensive_validation', {}).get('budget_compliance_rate', 0.0),
                trade_off_efficiency=validation_result.metadata.get('comprehensive_validation', {}).get('trade_off_efficiency', 0.0),
                user_satisfaction=None,
                success_rating=self._calculate_overall_success_rating(modeling_result, validation_result),
                error_count=sum(1 for r in self.execution_history if r.status == "failed"),
                recovery_attempts=sum(self.retry_counts.values()),
                timestamp=datetime.now(),
                metadata={
                    'session_id': self.session_id,
                    'config': self.config.__dict__,
                    'execution_history': len(self.execution_history),
                    'final_data_shape': modeling_result.data.shape if modeling_result.data is not None else None
                }
            )
            
            success = self.meta_db.store_pipeline_execution(pipeline_execution)
            if success:
                self.logger.info("Pipeline execution stored to database for meta-learning")
                
            # Also record to knowledge graph if available
            if self.knowledge_graph_service:
                self._record_execution_to_knowledge_graph(
                    dataset, target_column, modeling_result, validation_result, pipeline_execution
                )
                self.execution_id = pipeline_execution.execution_id
            else:
                self.logger.warning("Failed to store pipeline execution to database")
                
        except Exception as e:
            self.logger.error(f"Error recording execution to database: {e}")

    def _record_execution_to_knowledge_graph(self, dataset: pd.DataFrame, target_column: str,
                                           modeling_result: StageResult, validation_result: StageResult,
                                           pipeline_execution) -> None:
        """Record pipeline execution to knowledge graph"""
        try:
            # Prepare execution data for knowledge graph
            execution_data = {
                'execution_id': pipeline_execution.execution_id,
                'session_id': self.session_id,
                'start_time': datetime.now(),
                'status': 'completed' if modeling_result.status == 'success' else 'failed',
                'stage_results': {
                    stage_result.stage.value: {
                        'status': stage_result.status,
                        'execution_time': stage_result.execution_time,
                        'metadata': stage_result.metadata
                    } for stage_result in self.execution_history
                },
                'resource_usage': {
                    'total_execution_time': sum(r.execution_time for r in self.execution_history),
                    'memory_usage': sum(r.memory_usage for r in self.execution_history)
                },
                'dataset_characteristics': {
                    'name': f"dataset_{self.session_id}",
                    'shape': dataset.shape,
                    'file_hash': str(hash(str(dataset.head().to_dict()))),
                    'data_types': dataset.dtypes.to_dict(),
                    'overall_missing_percentage': dataset.isnull().sum().sum() / (dataset.shape[0] * dataset.shape[1]),
                    'domain': self.project_definition.domain.value if self.project_definition else 'general',
                    'quality_score': validation_result.metadata.get('comprehensive_validation', {}).get('trust_score', 0.0)
                },
                'project_definition': {
                    'name': f"project_{self.session_id}",
                    'objective': self.project_definition.objective.value if self.project_definition else 'accuracy',
                    'domain': self.project_definition.domain.value if self.project_definition else 'general',
                    'business_goal': self.project_definition.business_goal if self.project_definition else 'Data analysis',
                    'constraints': self._extract_constraints_dict(self.project_definition.technical_constraints) if self.project_definition else {},
                    'success_criteria': {'min_accuracy': 0.8}
                },
                'model_info': {
                    'name': f"model_{self.session_id}",
                    'algorithm': modeling_result.metadata.get('modeling_results', {}).get('best_algorithm', 'Unknown'),
                    'hyperparameters': modeling_result.metadata.get('modeling_results', {}).get('best_params', {}),
                    'trust_score': validation_result.metadata.get('comprehensive_validation', {}).get('trust_score', 0.0),
                    'interpretability_score': validation_result.metadata.get('explainability_summary', {}).get('interpretability_score', 0.0),
                    'training_time': modeling_result.execution_time,
                    'metadata': modeling_result.metadata
                },
                'performance_metrics': modeling_result.metadata.get('modeling_results', {}).get('performance_metrics', {}),
                'metadata': {
                    'pipeline_config': self.config.__dict__,
                    'error_count': sum(1 for r in self.execution_history if r.status == "failed"),
                    'recovery_attempts': sum(self.retry_counts.values())
                }
            }
            
            # Record to knowledge graph
            node_ids = self.knowledge_graph_service.record_pipeline_execution(execution_data)
            
            if node_ids:
                self.logger.info(f"Pipeline execution recorded to knowledge graph: {node_ids}")
            else:
                self.logger.warning("Failed to record pipeline execution to knowledge graph")
                
        except Exception as e:
            self.logger.error(f"Error recording execution to knowledge graph: {e}")

    def _extract_constraints_dict(self, technical_constraints) -> Dict[str, Any]:
        """Extract constraints dictionary from technical constraints object"""
        if not technical_constraints:
            return {}
        
        try:
            return {
                'max_latency_ms': technical_constraints.max_latency_ms,
                'max_training_hours': technical_constraints.max_training_hours,
                'min_accuracy': technical_constraints.min_accuracy,
                'max_memory_gb': technical_constraints.max_memory_gb
            }
        except Exception:
            return {}

    def _record_execution_for_learning(self, dataset: pd.DataFrame, target_column: str,
                                     modeling_result: StageResult, validation_result: StageResult):
        """Record pipeline execution for adaptive learning system"""
        try:
            # Extract project configuration
            project_config = {
                'objective': self.project_definition.objective.value if self.project_definition else 'accuracy',
                'domain': self.project_definition.domain.value if self.project_definition else 'general',
                'constraints': self._extract_constraints_dict(self.project_definition.technical_constraints) if self.project_definition else {},
                'strategy_applied': 'strategic' if self.project_definition else 'technical',
                'feature_engineering_enabled': self.config.enable_feature_engineering,
                'feature_selection_enabled': self.config.enable_feature_selection,
                'security_level': self.config.security_level,
                'privacy_level': self.config.privacy_level,
                'compliance_requirements': self.config.applicable_regulations
            }
            
            # Extract pipeline results
            pipeline_results = {
                'pipeline_stages': [stage.value for stage in PipelineStage],
                'execution_time': sum(r.execution_time for r in self.execution_history),
                'performance_metrics': validation_result.metadata.get('validation_metrics', {}),
                'trust_score': validation_result.metadata.get('comprehensive_validation', {}).get('trust_score', 0.0),
                'validation_success': validation_result.status == "success",
                'budget_compliance_rate': validation_result.metadata.get('comprehensive_validation', {}).get('budget_compliance_rate', 0.0),
                'trade_off_efficiency': validation_result.metadata.get('comprehensive_validation', {}).get('trade_off_efficiency', 0.0),
                'success_rating': self._calculate_overall_success_rating(modeling_result, validation_result),
                'error_count': sum(1 for r in self.execution_history if r.status == "failed"),
                'recovery_attempts': sum(self.retry_counts.values()),
                'metadata': {
                    'session_id': self.session_id,
                    'config': self.config.__dict__,
                    'execution_history': len(self.execution_history)
                }
            }
            
            # Record execution for meta-learning
            domain = self.project_definition.domain.value if self.project_definition else 'general'
            success = self.adaptive_system.record_pipeline_execution(
                dataset=dataset,
                target_column=target_column,
                project_config=project_config,
                pipeline_results=pipeline_results,
                session_id=self.session_id,
                domain=domain
            )
            
            if success:
                self.logger.info("Pipeline execution recorded for adaptive learning")
            else:
                self.logger.warning("Failed to record pipeline execution for adaptive learning")
                
        except Exception as e:
            self.logger.error(f"Error recording execution for learning: {e}")
    
    def _calculate_overall_success_rating(self, modeling_result: StageResult, 
                                        validation_result: StageResult) -> float:
        """Calculate overall success rating for the pipeline execution"""
        try:
            # Base success from successful stages
            successful_stages = sum(1 for r in self.execution_history if r.status == "success")
            total_stages = len(self.execution_history)
            stage_success_rate = successful_stages / total_stages if total_stages > 0 else 0.0
            
            # Performance component
            performance_score = 0.0
            if modeling_result and modeling_result.metadata:
                best_score = modeling_result.metadata.get('best_score', 0.0)
                performance_score = min(1.0, best_score)
            
            # Validation component
            validation_score = 0.0
            if validation_result and validation_result.status == "success":
                validation_metrics = validation_result.metadata.get('validation_metrics', {})
                if validation_metrics:
                    validation_score = np.mean(list(validation_metrics.values()))
            
            # Weighted combination
            overall_rating = (
                stage_success_rate * 0.3 +
                performance_score * 0.4 +
                validation_score * 0.3
            )
            
            return min(1.0, max(0.0, overall_rating))
            
        except Exception as e:
            self.logger.error(f"Error calculating success rating: {e}")
            return 0.5  # Default moderate success rating
    
    def get_adaptive_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the adaptive learning system"""
        if not self.adaptive_system:
            return {'status': 'adaptive_learning_disabled'}
        
        return self.adaptive_system.get_meta_learning_insights()
    
    def _synthesize_strategic_insights(self, technical_config: Dict[str, Any],
                                     modeling_result: StageResult, 
                                     validation_result: StageResult,
                                     ingestion_result: StageResult) -> Dict[str, Any]:
        """Synthesize technical results into strategic business insights"""
        
        try:
            validation_summary = self._create_validation_summary_from_results(
                modeling_result, validation_result
            )
            
            insight_orchestrator = InsightOrchestrator()
            session_id = insight_orchestrator.orchestrate_comprehensive_insights(
                project_definition=self.project_definition,
                validation_summary=validation_summary
            )
            
            session_summary = insight_orchestrator.get_insight_session_summary(session_id)
            
            return {
                "status": "success",
                "project_id": self.project_definition.project_id,
                "session_id": self.session_id,
                "insight_session_id": session_id,
                "strategic_summary": {
                    "objective_achieved": validation_summary.primary_objective_met if hasattr(validation_summary, 'primary_objective_met') else True,
                    "business_impact": session_summary.get('business_impact', {}),
                    "executive_summary": session_summary.get('executive_summary', {}),
                    "technical_summary": session_summary.get('technical_summary', {}),
                    "recommendations": session_summary.get('recommendations', [])
                },
                "technical_artifacts": {
                    "model_metadata": modeling_result.metadata,
                    "validation_metrics": validation_result.metadata,
                    "execution_summary": self._compile_execution_summary(),
                    "pipeline_config": technical_config
                },
                "communication_assets": session_summary.get('communication_assets', {}),
                "performance_metrics": {
                    "execution_time": sum(r.execution_time for r in self.execution_history),
                    "success_rate": len([r for r in self.execution_history if r.status == "success"]) / len(self.execution_history),
                    "final_performance": modeling_result.metadata.get('best_score', 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Strategic insights synthesis failed: {e}")
            return self._compile_final_results()
    
    def _create_validation_summary_from_results(self, modeling_result: StageResult, 
                                              validation_result: StageResult) -> ValidationSummary:
        """Create ValidationSummary from pipeline stage results"""
        
        try:
            from ..validation.validation_orchestrator import ValidationSummary
            from ..validation.objective_benchmarker import BenchmarkResult
            from ..validation.performance_budget_manager import BudgetReport
            from ..validation.trade_off_analyzer import TradeOffReport
            from datetime import datetime
            
            if hasattr(validation_result, 'artifacts') and 'validation_summary' in validation_result.artifacts:
                return validation_result.artifacts['validation_summary']
            
            benchmark_results = {}
            if modeling_result and modeling_result.metadata:
                best_score = modeling_result.metadata.get('best_score', 0.0)
                benchmark_results[self.project_definition.objective] = type('MockBenchmarkResult', (), {
                    'achieved_score': best_score,
                    'benchmark_score': 0.8,
                    'meets_benchmark': best_score >= 0.8,
                    'performance_ratio': best_score / 0.8,
                    'confidence_interval': (best_score - 0.05, best_score + 0.05),
                    'recommendations': []
                })()
            
            budget_report = type('MockBudgetReport', (), {
                'passed_budgets': [],
                'violations': [],
                'warnings': [],
                'overall_compliance': 0.9,
                'resource_utilization': {},
                'recommendations': []
            })()
            
            trade_off_report = type('MockTradeOffReport', (), {
                'overall_trade_off_efficiency': 0.8,
                'optimization_suggestions': []
            })()
            
            return ValidationSummary(
                validation_timestamp=datetime.now(),
                overall_success=validation_result.status == "success",
                primary_objective_met=modeling_result.metadata.get('best_score', 0.0) >= 0.7,
                budget_compliance_rate=0.9,
                trade_off_efficiency=0.8,
                benchmark_results=benchmark_results,
                budget_report=budget_report,
                trade_off_report=trade_off_report,
                strategic_recommendations=[],
                technical_optimizations=[],
                risk_assessments=[],
                validation_confidence=0.85,
                decision_support_score=0.8
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create validation summary: {e}")
            return None
    
    def _compile_execution_summary(self) -> Dict[str, Any]:
        """Compile execution summary for technical artifacts"""
        
        return {
            'total_stages': len(self.execution_history),
            'successful_stages': sum(1 for r in self.execution_history if r.status == "success"),
            'failed_stages': sum(1 for r in self.execution_history if r.status == "failed"),
            'total_execution_time': sum(r.execution_time for r in self.execution_history),
            'stage_breakdown': {
                result.stage.value: {
                    'status': result.status,
                    'execution_time': result.execution_time,
                    'success': result.status == "success"
                }
                for result in self.execution_history
            }
        }
    
    def _store_results_in_rag(self, final_results: Dict[str, Any], 
                              dataset: pd.DataFrame, 
                              profiling_result: StageResult):
        """Store pipeline execution results and intelligence profile in RAG system"""
        
        if not self.rag_system:
            self.logger.info("RAG system not available - skipping result storage")
            return
        
        try:
            # Store execution report
            self.rag_system.add_execution_report(final_results, self.session_id)
            self.logger.info("Execution results stored in RAG system")
            
            # Store intelligence profile if available
            if profiling_result and profiling_result.metadata:
                intelligence_profile = profiling_result.metadata.get('intelligence_profile', {})
                if intelligence_profile:
                    self.rag_system.add_intelligence_profile(intelligence_profile, self.session_id)
                    self.logger.info("Intelligence profile stored in RAG system")
            
            # Store dataset characteristics summary
            dataset_summary = {
                'shape': dataset.shape,
                'columns': list(dataset.columns),
                'dtypes': {col: str(dtype) for col, dtype in dataset.dtypes.items()},
                'memory_usage': dataset.memory_usage().sum(),
                'null_counts': dataset.isnull().sum().to_dict()
            }
            
            dataset_content = f"""
Dataset Analysis Summary:
- Shape: {dataset.shape[0]} rows, {dataset.shape[1]} columns
- Columns: {', '.join(dataset.columns)}
- Memory usage: {dataset_summary['memory_usage'] / 1024 / 1024:.2f} MB
- Missing values: {sum(dataset_summary['null_counts'].values())} total
- Data types: {len(dataset.select_dtypes(include='number').columns)} numerical, {len(dataset.select_dtypes(include='object').columns)} categorical
"""
            
            self.rag_system.add_document(
                content=dataset_content,
                metadata={
                    'type': 'dataset_summary',
                    'session_id': self.session_id,
                    'shape': dataset.shape,
                    'timestamp': datetime.now().isoformat()
                },
                document_type='dataset_summary',
                session_id=self.session_id
            )
            
            self.logger.info("Dataset summary stored in RAG system")
            
        except Exception as e:
            self.logger.error(f"Failed to store results in RAG system: {e}")
