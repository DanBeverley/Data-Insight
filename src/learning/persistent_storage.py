import sqlite3
import json
import pickle
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from enum import Enum

import pandas as pd
import numpy as np

try:
    from ..database import DatabaseManager, get_database_manager, DatasetCharacteristics as DBDatasetCharacteristics
    from ..database import ProjectConfig as DBProjectConfig, PipelineExecution as DBPipelineExecution
    from ..database import LearningPattern as DBLearningPattern, ExecutionFeedback as DBExecutionFeedback
    from ..database.migrations import initialize_database_schema
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class StorageEngine(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    AUTO = "auto"


@dataclass
class DatasetCharacteristics:
    """Comprehensive dataset characterization for meta-learning"""
    dataset_hash: str
    n_samples: int
    n_features: int
    n_categorical: int
    n_numerical: int
    n_text: int
    n_datetime: int
    missing_ratio: float
    target_type: str  # classification, regression, clustering
    target_cardinality: int
    class_imbalance_ratio: float
    correlation_strength: float
    skewness_avg: float
    kurtosis_avg: float
    domain: str
    task_complexity_score: float
    feature_diversity_score: float
    data_quality_score: float


@dataclass
class ProjectConfig:
    """Project configuration fingerprint for matching similar projects"""
    objective: str
    domain: str
    constraints: Dict[str, Any]
    config_hash: str
    strategy_applied: str
    feature_engineering_enabled: bool
    feature_selection_enabled: bool
    security_level: str
    privacy_level: str
    compliance_requirements: List[str]


@dataclass
class PipelineExecution:
    """Complete pipeline execution record for meta-learning"""
    execution_id: str
    session_id: str
    dataset_characteristics: DatasetCharacteristics
    project_config: ProjectConfig
    pipeline_stages: List[str]
    execution_time: float
    final_performance: Dict[str, float]
    trust_score: float
    validation_success: bool
    budget_compliance_rate: float
    trade_off_efficiency: float
    user_satisfaction: Optional[float]
    success_rating: float
    error_count: int
    recovery_attempts: int
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class LearningPattern:
    """Discovered learning patterns for recommendation"""
    pattern_id: str
    pattern_type: str  # 'successful_config', 'failure_mode', 'optimization_path'
    dataset_context: Dict[str, Any]
    config_elements: Dict[str, Any]
    success_indicators: Dict[str, float]
    confidence_score: float
    usage_count: int
    last_validated: datetime
    improvement_evidence: List[Dict[str, Any]]


class PersistentMetaDatabase:
    """Production-grade persistent storage with optimized PostgreSQL operations"""
    
    def __init__(self, db_path: Optional[str] = None, 
                 storage_engine: StorageEngine = StorageEngine.AUTO):
        self.storage_engine = storage_engine
        self.database_manager: Optional[DatabaseManager] = None
        
        if SQLALCHEMY_AVAILABLE:
            self._initialize_sqlalchemy_database()
            # Use optimized database service for high-performance operations
            try:
                from ..database.service import get_database_service
                self.db_service = get_database_service()
            except ImportError:
                self.db_service = None
        else:
            self._initialize_sqlite_fallback(db_path or "data_insight_meta.db")
            self.db_service = None
    
    def _initialize_sqlalchemy_database(self):
        """Initialize with production SQLAlchemy infrastructure"""
        try:
            self.database_manager = get_database_manager()
            success = initialize_database_schema(self.database_manager)
            if not success:
                raise RuntimeError("Failed to initialize database schema")
        except Exception as e:
            print(f"SQLAlchemy initialization failed: {e}")
            print("Falling back to SQLite")
            self._initialize_sqlite_fallback("data_insight_meta.db")
    
    def _initialize_sqlite_fallback(self, db_path: str):
        """Fallback to direct SQLite initialization"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.database_manager = None
        self._initialize_sqlite_schema()
    
    def _initialize_sqlite_schema(self):
        """Initialize SQLite database schema (fallback)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            schemas = [
                '''CREATE TABLE IF NOT EXISTS dataset_characteristics (
                    dataset_hash TEXT PRIMARY KEY, n_samples INTEGER, n_features INTEGER,
                    n_categorical INTEGER, n_numerical INTEGER, n_text INTEGER, n_datetime INTEGER,
                    missing_ratio REAL, target_type TEXT, target_cardinality INTEGER,
                    class_imbalance_ratio REAL, correlation_strength REAL, skewness_avg REAL,
                    kurtosis_avg REAL, domain TEXT, task_complexity_score REAL,
                    feature_diversity_score REAL, data_quality_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
                
                '''CREATE TABLE IF NOT EXISTS project_configs (
                    config_hash TEXT PRIMARY KEY, objective TEXT, domain TEXT, constraints TEXT,
                    strategy_applied TEXT, feature_engineering_enabled BOOLEAN,
                    feature_selection_enabled BOOLEAN, security_level TEXT, privacy_level TEXT,
                    compliance_requirements TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
                
                '''CREATE TABLE IF NOT EXISTS pipeline_executions (
                    execution_id TEXT PRIMARY KEY, session_id TEXT, dataset_hash TEXT, config_hash TEXT,
                    pipeline_stages TEXT, execution_time REAL, final_performance TEXT, trust_score REAL,
                    validation_success BOOLEAN, budget_compliance_rate REAL, trade_off_efficiency REAL,
                    user_satisfaction REAL, success_rating REAL, error_count INTEGER, recovery_attempts INTEGER,
                    timestamp TIMESTAMP, metadata TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dataset_hash) REFERENCES dataset_characteristics(dataset_hash),
                    FOREIGN KEY (config_hash) REFERENCES project_configs(config_hash))''',
                
                '''CREATE TABLE IF NOT EXISTS learning_patterns (
                    pattern_id TEXT PRIMARY KEY, pattern_type TEXT, dataset_context TEXT, config_elements TEXT,
                    success_indicators TEXT, confidence_score REAL, usage_count INTEGER DEFAULT 0,
                    last_validated TIMESTAMP, improvement_evidence TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
                
                '''CREATE TABLE IF NOT EXISTS execution_feedback (
                    feedback_id TEXT PRIMARY KEY, execution_id TEXT, user_rating REAL, issues_encountered TEXT,
                    suggestions TEXT, performance_expectation_met BOOLEAN, would_recommend BOOLEAN,
                    feedback_timestamp TIMESTAMP,
                    FOREIGN KEY (execution_id) REFERENCES pipeline_executions(execution_id))'''
            ]
            
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_dataset_domain ON dataset_characteristics(domain)',
                'CREATE INDEX IF NOT EXISTS idx_dataset_complexity ON dataset_characteristics(task_complexity_score)',
                'CREATE INDEX IF NOT EXISTS idx_execution_success ON pipeline_executions(success_rating)',
                'CREATE INDEX IF NOT EXISTS idx_execution_timestamp ON pipeline_executions(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_pattern_confidence ON learning_patterns(confidence_score)'
            ]
            
            for schema in schemas:
                cursor.execute(schema)
            for index in indexes:
                cursor.execute(index)
            
            conn.commit()
    
    @contextmanager 
    def _get_connection(self):
        """Context manager for database connections"""
        if self.database_manager:
            with self.database_manager.get_session() as session:
                yield session
        else:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            try:
                yield conn
            finally:
                conn.close()
    
    def store_pipeline_execution(self, execution: PipelineExecution) -> bool:
        """Store complete pipeline execution using optimized operations"""
        try:
            if self.db_service:
                return self.db_service.store_execution_optimized(execution)
            elif self.database_manager:
                return self._store_execution_sqlalchemy(execution)
            else:
                return self._store_execution_sqlite(execution)
        except Exception as e:
            print(f"Error storing pipeline execution: {e}")
            return False
    
    def _store_execution_sqlalchemy(self, execution: PipelineExecution) -> bool:
        """Store execution using SQLAlchemy ORM"""
        with self._get_connection() as session:
            dataset_chars = DBDatasetCharacteristics(
                dataset_hash=execution.dataset_characteristics.dataset_hash,
                n_samples=execution.dataset_characteristics.n_samples,
                n_features=execution.dataset_characteristics.n_features,
                n_categorical=execution.dataset_characteristics.n_categorical,
                n_numerical=execution.dataset_characteristics.n_numerical,
                n_text=execution.dataset_characteristics.n_text,
                n_datetime=execution.dataset_characteristics.n_datetime,
                missing_ratio=execution.dataset_characteristics.missing_ratio,
                target_type=execution.dataset_characteristics.target_type,
                target_cardinality=execution.dataset_characteristics.target_cardinality,
                class_imbalance_ratio=execution.dataset_characteristics.class_imbalance_ratio,
                correlation_strength=execution.dataset_characteristics.correlation_strength,
                skewness_avg=execution.dataset_characteristics.skewness_avg,
                kurtosis_avg=execution.dataset_characteristics.kurtosis_avg,
                domain=execution.dataset_characteristics.domain,
                task_complexity_score=execution.dataset_characteristics.task_complexity_score,
                feature_diversity_score=execution.dataset_characteristics.feature_diversity_score,
                data_quality_score=execution.dataset_characteristics.data_quality_score
            )
            session.merge(dataset_chars)
            
            project_config = DBProjectConfig(
                config_hash=execution.project_config.config_hash,
                objective=execution.project_config.objective,
                domain=execution.project_config.domain,
                constraints=execution.project_config.constraints,
                strategy_applied=execution.project_config.strategy_applied,
                feature_engineering_enabled=execution.project_config.feature_engineering_enabled,
                feature_selection_enabled=execution.project_config.feature_selection_enabled,
                security_level=execution.project_config.security_level,
                privacy_level=execution.project_config.privacy_level,
                compliance_requirements=execution.project_config.compliance_requirements
            )
            session.merge(project_config)
            
            pipeline_exec = DBPipelineExecution(
                execution_id=execution.execution_id,
                session_id=execution.session_id,
                dataset_hash=execution.dataset_characteristics.dataset_hash,
                config_hash=execution.project_config.config_hash,
                pipeline_stages=execution.pipeline_stages,
                execution_time=execution.execution_time,
                final_performance=execution.final_performance,
                trust_score=execution.trust_score,
                validation_success=execution.validation_success,
                budget_compliance_rate=execution.budget_compliance_rate,
                trade_off_efficiency=execution.trade_off_efficiency,
                user_satisfaction=execution.user_satisfaction,
                success_rating=execution.success_rating,
                error_count=execution.error_count,
                recovery_attempts=execution.recovery_attempts,
                timestamp=execution.timestamp,
                metadata=execution.metadata
            )
            session.add(pipeline_exec)
            return True
    
    def _store_execution_sqlite(self, execution: PipelineExecution) -> bool:
        """Store execution using direct SQLite"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            dataset_data = asdict(execution.dataset_characteristics)
            cursor.execute('''
                INSERT OR REPLACE INTO dataset_characteristics VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            ''', tuple(dataset_data.values()))
            
            config_data = asdict(execution.project_config)
            config_data['constraints'] = json.dumps(config_data['constraints'])
            config_data['compliance_requirements'] = json.dumps(config_data['compliance_requirements'])
            cursor.execute('''
                INSERT OR REPLACE INTO project_configs VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            ''', tuple(config_data.values()))
            
            execution_data = (
                execution.execution_id, execution.session_id,
                execution.dataset_characteristics.dataset_hash, execution.project_config.config_hash,
                json.dumps(execution.pipeline_stages), execution.execution_time,
                json.dumps(execution.final_performance), execution.trust_score,
                execution.validation_success, execution.budget_compliance_rate,
                execution.trade_off_efficiency, execution.user_satisfaction,
                execution.success_rating, execution.error_count, execution.recovery_attempts,
                execution.timestamp.isoformat(), json.dumps(execution.metadata)
            )
            
            cursor.execute('''
                INSERT INTO pipeline_executions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
            ''', execution_data)
            
            conn.commit()
            return True
    
    def find_similar_successful_executions(self, 
                                         dataset_chars: DatasetCharacteristics,
                                         project_config: ProjectConfig,
                                         similarity_threshold: float = 0.8,
                                         min_success_rating: float = 0.7,
                                         min_user_satisfaction: float = 0.6,
                                         limit: int = 10) -> List[PipelineExecution]:
        """Find similar successful executions using optimized similarity search"""
        
        if self.db_service:
            try:
                similar_results = self.db_service.find_similar_executions(
                    dataset_chars, min_success_rating, limit
                )
                # Convert to PipelineExecution objects (simplified)
                return [self._create_mock_execution(r) for r in similar_results 
                       if r['similarity_score'] >= similarity_threshold]
            except Exception as e:
                print(f"Optimized similarity search failed: {e}")
        
        return self._find_similar_fallback(dataset_chars, project_config, 
                                         similarity_threshold, min_success_rating, min_user_satisfaction, limit)
    
    def _find_similar_fallback(self, dataset_chars: DatasetCharacteristics,
                              project_config: ProjectConfig,
                              similarity_threshold: float,
                              min_success_rating: float,
                              min_user_satisfaction: float,
                              limit: int) -> List[PipelineExecution]:
        """Find similar successful pipeline executions for recommendation"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Calculate similarity based on dataset characteristics and config
                query = '''
                    SELECT pe.*, dc.*, pc.*
                    FROM pipeline_executions pe
                    JOIN dataset_characteristics dc ON pe.dataset_hash = dc.dataset_hash
                    JOIN project_configs pc ON pe.config_hash = pc.config_hash
                    WHERE pe.success_rating >= ?
                    AND pe.validation_success = 1
                    AND (pe.user_satisfaction IS NULL OR pe.user_satisfaction >= ?)
                    ORDER BY 
                        CASE WHEN pe.user_satisfaction IS NOT NULL THEN pe.user_satisfaction ELSE pe.success_rating END DESC,
                        pe.success_rating DESC, 
                        pe.trust_score DESC
                    LIMIT ?
                '''
                
                cursor.execute(query, (min_success_rating, min_user_satisfaction, limit * 3))  # Get more for filtering
                rows = cursor.fetchall()
                
                similar_executions = []
                for row in rows:
                    similarity = self._calculate_execution_similarity(dataset_chars, project_config, row)
                    if similarity >= similarity_threshold:
                        execution = self._row_to_pipeline_execution(row)
                        user_satisfaction = execution.user_satisfaction or 0.0
                        
                        # Weight similarity by user satisfaction (if available)
                        if user_satisfaction > 0:
                            weighted_similarity = similarity * (0.7 + 0.3 * user_satisfaction)
                        else:
                            weighted_similarity = similarity * 0.8  # Slight penalty for no feedback
                            
                        execution.metadata['similarity_score'] = weighted_similarity
                        execution.metadata['user_validated'] = user_satisfaction > 0
                        similar_executions.append(execution)
                
                # Sort by similarity and return top results
                similar_executions.sort(key=lambda x: x.metadata.get('similarity_score', 0), reverse=True)
                return similar_executions[:limit]
                
        except Exception as e:
            print(f"Error finding similar executions: {e}")
            return []
    
    def _calculate_execution_similarity(self, target_dataset: DatasetCharacteristics,
                                      target_config: ProjectConfig, row: tuple) -> float:
        """Calculate similarity score between target and stored execution"""
        similarity_scores = []
        
        # Dataset similarity (weight: 0.6)
        dataset_features = [
            (target_dataset.n_samples, row[1], 0.1),  # n_samples
            (target_dataset.n_features, row[2], 0.15),  # n_features  
            (target_dataset.missing_ratio, row[7], 0.1),  # missing_ratio
            (target_dataset.task_complexity_score, row[15], 0.15),  # task_complexity_score
            (target_dataset.data_quality_score, row[17], 0.1)  # data_quality_score
        ]
        
        dataset_similarity = 0.0
        for target_val, stored_val, weight in dataset_features:
            if stored_val is not None and target_val is not None:
                if stored_val == 0:
                    feature_sim = 1.0 if target_val == 0 else 0.0
                else:
                    feature_sim = 1 - abs(target_val - stored_val) / max(target_val, stored_val)
                dataset_similarity += feature_sim * weight
        
        similarity_scores.append(dataset_similarity * 0.6)
        
        # Domain similarity (weight: 0.2)
        domain_similarity = 1.0 if target_dataset.domain == row[14] else 0.5
        similarity_scores.append(domain_similarity * 0.2)
        
        # Objective similarity (weight: 0.2)
        objective_similarity = 1.0 if target_config.objective == row[19] else 0.3
        similarity_scores.append(objective_similarity * 0.2)
        
        return sum(similarity_scores)
    
    def _row_to_pipeline_execution(self, row: tuple) -> PipelineExecution:
        """Convert database row to PipelineExecution object"""
        # This is a simplified version - in production, would properly map all fields
        return PipelineExecution(
            execution_id=row[0],
            session_id=row[1],
            dataset_characteristics=DatasetCharacteristics(
                dataset_hash=row[2],
                n_samples=row[19] or 0,
                n_features=row[20] or 0,
                n_categorical=row[21] or 0,
                n_numerical=row[22] or 0,
                n_text=row[23] or 0,
                n_datetime=row[24] or 0,
                missing_ratio=row[25] or 0.0,
                target_type=row[26] or "unknown",
                target_cardinality=row[27] or 0,
                class_imbalance_ratio=row[28] or 0.0,
                correlation_strength=row[29] or 0.0,
                skewness_avg=row[30] or 0.0,
                kurtosis_avg=row[31] or 0.0,
                domain=row[32] or "general",
                task_complexity_score=row[33] or 0.0,
                feature_diversity_score=row[34] or 0.0,
                data_quality_score=row[35] or 0.0
            ),
            project_config=ProjectConfig(
                objective=row[37] or "accuracy",
                domain=row[38] or "general",
                constraints=json.loads(row[39]) if row[39] else {},
                config_hash=row[3],
                strategy_applied=row[40] or "default",
                feature_engineering_enabled=bool(row[41]),
                feature_selection_enabled=bool(row[42]),
                security_level=row[43] or "standard",
                privacy_level=row[44] or "medium",
                compliance_requirements=json.loads(row[45]) if row[45] else []
            ),
            pipeline_stages=json.loads(row[4]) if row[4] else [],
            execution_time=row[5] or 0.0,
            final_performance=json.loads(row[6]) if row[6] else {},
            trust_score=row[7] or 0.0,
            validation_success=bool(row[8]),
            budget_compliance_rate=row[9] or 0.0,
            trade_off_efficiency=row[10] or 0.0,
            user_satisfaction=row[11],
            success_rating=row[12] or 0.0,
            error_count=row[13] or 0,
            recovery_attempts=row[14] or 0,
            timestamp=datetime.fromisoformat(row[15]),
            metadata=json.loads(row[16]) if row[16] else {}
        )
    
    def store_learning_pattern(self, pattern: LearningPattern) -> bool:
        """Store discovered learning pattern"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                pattern_data = (
                    pattern.pattern_id,
                    pattern.pattern_type,
                    json.dumps(pattern.dataset_context),
                    json.dumps(pattern.config_elements),
                    json.dumps(pattern.success_indicators),
                    pattern.confidence_score,
                    pattern.usage_count,
                    pattern.last_validated.isoformat(),
                    json.dumps(pattern.improvement_evidence)
                )
                
                cursor.execute('''
                    INSERT OR REPLACE INTO learning_patterns VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                    )
                ''', pattern_data)
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error storing learning pattern: {e}")
            return False
    
    def get_recommended_strategies(self, dataset_chars: DatasetCharacteristics,
                                 objective: str, domain: str) -> Dict[str, Any]:
        """Get strategy recommendations based on historical success patterns"""
        similar_executions = self.find_similar_successful_executions(
            dataset_chars,
            ProjectConfig(
                objective=objective,
                domain=domain,
                constraints={},
                config_hash="temp",
                strategy_applied="",
                feature_engineering_enabled=True,
                feature_selection_enabled=True,
                security_level="standard",
                privacy_level="medium",
                compliance_requirements=[]
            )
        )
        
        if not similar_executions:
            return self._get_default_recommendations(dataset_chars, objective, domain)
        
        # Analyze successful patterns
        fe_success_rate = sum(1 for e in similar_executions if e.project_config.feature_engineering_enabled) / len(similar_executions)
        fs_success_rate = sum(1 for e in similar_executions if e.project_config.feature_selection_enabled) / len(similar_executions)
        
        avg_performance = np.mean([e.success_rating for e in similar_executions])
        avg_execution_time = np.mean([e.execution_time for e in similar_executions])
        
        strategies = []
        for execution in similar_executions[:3]:  # Top 3 similar successful executions
            strategies.append({
                'strategy_name': f"Similar_Success_{execution.execution_id[:8]}",
                'config': {
                    'feature_engineering_enabled': execution.project_config.feature_engineering_enabled,
                    'feature_selection_enabled': execution.project_config.feature_selection_enabled,
                    'security_level': execution.project_config.security_level,
                    'privacy_level': execution.project_config.privacy_level
                },
                'expected_performance': execution.success_rating,
                'expected_execution_time': execution.execution_time,
                'confidence': execution.metadata.get('similarity_score', 0.0),
                'evidence': {
                    'similar_executions': 1,
                    'success_rating': execution.success_rating,
                    'trust_score': execution.trust_score
                }
            })
        
        return {
            'recommended_strategies': strategies,
            'meta_insights': {
                'total_similar_cases': len(similar_executions),
                'feature_engineering_success_rate': fe_success_rate,
                'feature_selection_success_rate': fs_success_rate,
                'average_performance': avg_performance,
                'average_execution_time': avg_execution_time
            },
            'confidence_score': np.mean([s['confidence'] for s in strategies])
        }
    
    def _get_default_recommendations(self, dataset_chars: DatasetCharacteristics,
                                   objective: str, domain: str) -> Dict[str, Any]:
        """Provide default recommendations when no similar cases exist"""
        return {
            'recommended_strategies': [{
                'strategy_name': 'Default_Conservative',
                'config': {
                    'feature_engineering_enabled': dataset_chars.n_features < 100,
                    'feature_selection_enabled': dataset_chars.n_features > 50,
                    'security_level': 'standard',
                    'privacy_level': 'medium'
                },
                'expected_performance': 0.75,
                'expected_execution_time': dataset_chars.n_samples * 0.001,
                'confidence': 0.5,
                'evidence': {'type': 'heuristic', 'basis': 'conservative_defaults'}
            }],
            'meta_insights': {
                'total_similar_cases': 0,
                'recommendation_basis': 'heuristic_defaults'
            },
            'confidence_score': 0.5
        }
    
    def record_user_feedback(self, execution_id: str, feedback: Dict[str, Any]) -> bool:
        """Record user feedback for execution quality"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                feedback_id = f"feedback_{execution_id}_{int(time.time())}"
                feedback_data = (
                    feedback_id,
                    execution_id,
                    feedback.get('user_rating'),
                    json.dumps(feedback.get('issues_encountered', [])),
                    feedback.get('suggestions'),
                    feedback.get('performance_expectation_met'),
                    feedback.get('would_recommend'),
                    datetime.now().isoformat()
                )
                
                cursor.execute('''
                    INSERT INTO execution_feedback VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                ''', feedback_data)
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error recording feedback: {e}")
            return False

    def update_execution_with_feedback(self, execution_id: str, user_satisfaction: float, success_rating: float) -> bool:
        """Update pipeline execution record with user feedback"""
        try:
            if self.db_service:
                return self.db_service.update_execution_feedback(execution_id, user_satisfaction, success_rating)
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE pipeline_executions 
                    SET user_satisfaction = ?, success_rating = ?
                    WHERE execution_id = ?
                ''', (user_satisfaction, success_rating, execution_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error updating execution with feedback: {e}")
            return False
    
    def get_system_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning system summary"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Basic statistics
                cursor.execute("SELECT COUNT(*) FROM pipeline_executions")
                total_executions = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM pipeline_executions WHERE success_rating >= 0.8")
                high_success_executions = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(success_rating) FROM pipeline_executions WHERE success_rating > 0")
                avg_success_rating = cursor.fetchone()[0] or 0.0
                
                cursor.execute("SELECT COUNT(DISTINCT domain) FROM dataset_characteristics")
                domains_covered = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM learning_patterns")
                patterns_discovered = cursor.fetchone()[0]
                
                # Recent trend
                cursor.execute('''
                    SELECT AVG(success_rating) 
                    FROM pipeline_executions 
                    WHERE timestamp > datetime('now', '-30 days')
                    AND success_rating > 0
                ''')
                recent_avg_success = cursor.fetchone()[0] or 0.0
                
                return {
                    'total_executions': total_executions,
                    'high_success_executions': high_success_executions,
                    'success_rate': high_success_executions / total_executions if total_executions > 0 else 0,
                    'average_success_rating': avg_success_rating,
                    'domains_covered': domains_covered,
                    'patterns_discovered': patterns_discovered,
                    'recent_performance': recent_avg_success,
                    'performance_trend': 'improving' if recent_avg_success > avg_success_rating else 'stable',
                    'meta_learning_maturity': min(1.0, total_executions / 1000),  # Mature after 1000 executions
                    'last_updated': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Error getting learning summary: {e}")
            return {'error': str(e)}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old execution data while preserving learning patterns"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
                
                cursor.execute('''
                    DELETE FROM execution_feedback 
                    WHERE execution_id IN (
                        SELECT execution_id FROM pipeline_executions 
                        WHERE timestamp < ?
                    )
                ''', (cutoff_date,))
                
                cursor.execute('''
                    DELETE FROM pipeline_executions 
                    WHERE timestamp < ?
                ''', (cutoff_date,))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error cleaning up old data: {e}")
            return False


def create_dataset_characteristics(df: pd.DataFrame, target_column: str = None,
                                 domain: str = "general") -> DatasetCharacteristics:
    """Create dataset characteristics from DataFrame for meta-learning storage"""
    
    # Calculate hash for dataset identification
    data_str = f"{df.shape}_{list(df.columns)}_{df.dtypes.to_dict()}"
    dataset_hash = hashlib.md5(data_str.encode()).hexdigest()
    
    # Basic characteristics
    n_samples, n_features = df.shape
    n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
    n_numerical = len(df.select_dtypes(include=[np.number]).columns)
    n_text = len([col for col in df.columns if df[col].dtype == 'object' 
                  and df[col].astype(str).str.len().mean() > 50])
    n_datetime = len(df.select_dtypes(include=['datetime64', 'datetime']).columns)
    
    # Quality metrics
    missing_ratio = df.isnull().sum().sum() / (n_samples * n_features)
    
    # Target characteristics
    if target_column and target_column in df.columns:
        target_cardinality = df[target_column].nunique()
        target_type = "classification" if target_cardinality <= 20 else "regression"
        
        if target_type == "classification":
            value_counts = df[target_column].value_counts()
            class_imbalance_ratio = value_counts.max() / value_counts.min() if len(value_counts) > 1 else 1.0
        else:
            class_imbalance_ratio = 0.0
    else:
        target_cardinality = 0
        target_type = "unsupervised"
        class_imbalance_ratio = 0.0
    
    # Statistical characteristics
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        correlation_strength = abs(numeric_df.corr()).mean().mean()
        skewness_avg = abs(numeric_df.skew()).mean()
        kurtosis_avg = abs(numeric_df.kurtosis()).mean()
    else:
        correlation_strength = 0.0
        skewness_avg = 0.0
        kurtosis_avg = 0.0
    
    # Complexity scores
    task_complexity_score = min(1.0, (n_features * n_samples) / 1000000)
    feature_diversity_score = (n_categorical + n_numerical + n_text + n_datetime) / max(1, n_features)
    data_quality_score = max(0.0, 1.0 - missing_ratio * 2)
    
    return DatasetCharacteristics(
        dataset_hash=dataset_hash,
        n_samples=n_samples,
        n_features=n_features,
        n_categorical=n_categorical,
        n_numerical=n_numerical,
        n_text=n_text,
        n_datetime=n_datetime,
        missing_ratio=missing_ratio,
        target_type=target_type,
        target_cardinality=target_cardinality,
        class_imbalance_ratio=class_imbalance_ratio,
        correlation_strength=correlation_strength,
        skewness_avg=skewness_avg,
        kurtosis_avg=kurtosis_avg,
        domain=domain,
        task_complexity_score=task_complexity_score,
        feature_diversity_score=feature_diversity_score,
        data_quality_score=data_quality_score
    )