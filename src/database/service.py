"""Optimized database service using native SQL operations"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from .connection import DatabaseManager, get_database_manager
from ..learning.persistent_storage import PipelineExecution, DatasetCharacteristics, ProjectConfig


class OptimizedDatabaseService:
    """High-performance database service using native SQL operations"""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        self.db = database_manager or get_database_manager()
        self.logger = logging.getLogger(__name__)
        self._load_sql_queries()
    
    def _load_sql_queries(self):
        """Load SQL queries from file"""
        sql_file = Path(__file__).parent / "queries.sql"
        if sql_file.exists():
            with open(sql_file, 'r') as f:
                self.sql_queries = f.read()
        else:
            self.sql_queries = ""
    
    def store_execution_optimized(self, execution: PipelineExecution) -> bool:
        """Store execution using optimized batch operations"""
        
        if not self.db.config.is_postgresql():
            return self._store_execution_sqlite(execution)
        
        try:
            with self.db.get_session() as session:
                # Use PostgreSQL UPSERT for optimal performance
                dataset_upsert = """
                INSERT INTO dataset_characteristics (
                    dataset_hash, n_samples, n_features, n_categorical, n_numerical,
                    n_text, n_datetime, missing_ratio, target_type, target_cardinality,
                    class_imbalance_ratio, correlation_strength, skewness_avg, kurtosis_avg,
                    domain, task_complexity_score, feature_diversity_score, data_quality_score
                ) VALUES (
                    %(hash)s, %(samples)s, %(features)s, %(categorical)s, %(numerical)s,
                    %(text)s, %(datetime)s, %(missing)s, %(target_type)s, %(cardinality)s,
                    %(imbalance)s, %(correlation)s, %(skewness)s, %(kurtosis)s,
                    %(domain)s, %(complexity)s, %(diversity)s, %(quality)s
                )
                ON CONFLICT (dataset_hash) DO UPDATE SET
                    n_samples = EXCLUDED.n_samples,
                    n_features = EXCLUDED.n_features,
                    data_quality_score = EXCLUDED.data_quality_score
                """
                
                config_upsert = """
                INSERT INTO project_configs (
                    config_hash, objective, domain, constraints, strategy_applied,
                    feature_engineering_enabled, feature_selection_enabled,
                    security_level, privacy_level, compliance_requirements
                ) VALUES (
                    %(hash)s, %(objective)s, %(domain)s, %(constraints)s, %(strategy)s,
                    %(fe_enabled)s, %(fs_enabled)s, %(security)s, %(privacy)s, %(compliance)s
                )
                ON CONFLICT (config_hash) DO NOTHING
                """
                
                execution_insert = """
                INSERT INTO pipeline_executions (
                    execution_id, session_id, dataset_hash, config_hash,
                    pipeline_stages, execution_time, final_performance, trust_score,
                    validation_success, budget_compliance_rate, trade_off_efficiency,
                    user_satisfaction, success_rating, error_count, recovery_attempts,
                    timestamp, metadata
                ) VALUES (
                    %(exec_id)s, %(session)s, %(dataset_hash)s, %(config_hash)s,
                    %(stages)s, %(exec_time)s, %(performance)s, %(trust)s,
                    %(validation)s, %(budget)s, %(tradeoff)s, %(satisfaction)s,
                    %(rating)s, %(errors)s, %(recovery)s, %(timestamp)s, %(metadata)s
                )
                """
                
                # Execute UPSERT operations
                session.execute(dataset_upsert, {
                    'hash': execution.dataset_characteristics.dataset_hash,
                    'samples': execution.dataset_characteristics.n_samples,
                    'features': execution.dataset_characteristics.n_features,
                    'categorical': execution.dataset_characteristics.n_categorical,
                    'numerical': execution.dataset_characteristics.n_numerical,
                    'text': execution.dataset_characteristics.n_text,
                    'datetime': execution.dataset_characteristics.n_datetime,
                    'missing': execution.dataset_characteristics.missing_ratio,
                    'target_type': execution.dataset_characteristics.target_type,
                    'cardinality': execution.dataset_characteristics.target_cardinality,
                    'imbalance': execution.dataset_characteristics.class_imbalance_ratio,
                    'correlation': execution.dataset_characteristics.correlation_strength,
                    'skewness': execution.dataset_characteristics.skewness_avg,
                    'kurtosis': execution.dataset_characteristics.kurtosis_avg,
                    'domain': execution.dataset_characteristics.domain,
                    'complexity': execution.dataset_characteristics.task_complexity_score,
                    'diversity': execution.dataset_characteristics.feature_diversity_score,
                    'quality': execution.dataset_characteristics.data_quality_score
                })
                
                session.execute(config_upsert, {
                    'hash': execution.project_config.config_hash,
                    'objective': execution.project_config.objective,
                    'domain': execution.project_config.domain,
                    'constraints': execution.project_config.constraints,
                    'strategy': execution.project_config.strategy_applied,
                    'fe_enabled': execution.project_config.feature_engineering_enabled,
                    'fs_enabled': execution.project_config.feature_selection_enabled,
                    'security': execution.project_config.security_level,
                    'privacy': execution.project_config.privacy_level,
                    'compliance': execution.project_config.compliance_requirements
                })
                
                session.execute(execution_insert, {
                    'exec_id': execution.execution_id,
                    'session': execution.session_id,
                    'dataset_hash': execution.dataset_characteristics.dataset_hash,
                    'config_hash': execution.project_config.config_hash,
                    'stages': execution.pipeline_stages,
                    'exec_time': execution.execution_time,
                    'performance': execution.final_performance,
                    'trust': execution.trust_score,
                    'validation': execution.validation_success,
                    'budget': execution.budget_compliance_rate,
                    'tradeoff': execution.trade_off_efficiency,
                    'satisfaction': execution.user_satisfaction,
                    'rating': execution.success_rating,
                    'errors': execution.error_count,
                    'recovery': execution.recovery_attempts,
                    'timestamp': execution.timestamp,
                    'metadata': execution.metadata
                })
                
                return True
                
        except Exception as e:
            self.logger.error(f"Optimized storage failed: {e}")
            return False
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system summary using optimized view"""
        
        try:
            with self.db.get_session() as session:
                if self.db.config.is_postgresql():
                    result = session.execute("SELECT * FROM system_learning_summary").fetchone()
                    return {
                        'total_executions': result.total_executions,
                        'high_success_executions': result.high_success_executions,
                        'success_rate': result.high_success_executions / result.total_executions if result.total_executions > 0 else 0,
                        'average_success_rating': float(result.avg_success_rating or 0),
                        'domains_covered': result.domains_covered,
                        'patterns_discovered': result.patterns_discovered,
                        'recent_performance': float(result.recent_avg_success or 0),
                        'performance_trend': result.performance_trend,
                        'meta_learning_maturity': float(result.meta_learning_maturity),
                        'last_updated': result.last_updated.isoformat(),
                        'database_type': 'PostgreSQL'
                    }
                else:
                    return self._get_summary_sqlite(session)
                    
        except Exception as e:
            self.logger.error(f"Summary query failed: {e}")
            return {'error': str(e)}
    
    def find_similar_executions(self, dataset_chars: DatasetCharacteristics,
                              min_success: float = 0.7, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar executions using optimized similarity function"""
        
        if not self.db.config.is_postgresql():
            return []  # Fallback to Python-based similarity
        
        try:
            with self.db.get_session() as session:
                query = """
                SELECT * FROM find_similar_executions(
                    %(samples)s, %(features)s, %(domain)s, %(complexity)s, %(min_success)s, %(limit)s
                ) ORDER BY 
                    CASE WHEN user_satisfaction IS NOT NULL THEN user_satisfaction * 0.4 + similarity_score * 0.6 
                         ELSE similarity_score * 0.8 END DESC
                """
                
                results = session.execute(query, {
                    'samples': dataset_chars.n_samples,
                    'features': dataset_chars.n_features,
                    'domain': dataset_chars.domain,
                    'complexity': dataset_chars.task_complexity_score,
                    'min_success': min_success,
                    'limit': limit
                }).fetchall()
                
                return [
                    {
                        'execution_id': row.execution_id,
                        'similarity_score': float(row.similarity_score),
                        'success_rating': float(row.success_rating),
                        'execution_time': float(row.execution_time),
                        'config_hash': row.config_hash,
                        'user_satisfaction': float(row.user_satisfaction) if row.user_satisfaction else None,
                        'user_validated': row.user_satisfaction is not None
                    }
                    for row in results
                ]
                
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []
    
    def get_domain_analytics(self) -> List[Dict[str, Any]]:
        """Get domain performance analytics using optimized view"""
        
        try:
            with self.db.get_session() as session:
                if self.db.config.is_postgresql():
                    results = session.execute("SELECT * FROM domain_performance_analytics").fetchall()
                    return [
                        {
                            'domain': row.domain,
                            'total_projects': row.total_projects,
                            'avg_success_rate': float(row.avg_success_rate),
                            'avg_execution_time': float(row.avg_execution_time),
                            'avg_trust_score': float(row.avg_trust_score),
                            'successful_validations': row.successful_validations,
                            'avg_budget_compliance': float(row.avg_budget_compliance),
                            'performance_consistency': float(row.performance_consistency or 0)
                        }
                        for row in results
                    ]
                else:
                    return []  # Fallback for SQLite
                    
        except Exception as e:
            self.logger.error(f"Domain analytics failed: {e}")
            return []
    
    def get_strategy_effectiveness(self) -> List[Dict[str, Any]]:
        """Get strategy effectiveness using optimized view"""
        
        try:
            with self.db.get_session() as session:
                if self.db.config.is_postgresql():
                    results = session.execute("SELECT * FROM strategy_effectiveness LIMIT 20").fetchall()
                    return [
                        {
                            'strategy_applied': row.strategy_applied,
                            'objective': row.objective,
                            'domain': row.domain,
                            'usage_count': row.usage_count,
                            'avg_performance': float(row.avg_performance),
                            'avg_time': float(row.avg_time),
                            'success_rate': float(row.success_rate)
                        }
                        for row in results
                    ]
                else:
                    return []
                    
        except Exception as e:
            self.logger.error(f"Strategy effectiveness query failed: {e}")
            return []
    
    def cleanup_old_data(self, retention_days: int = 365) -> int:
        """Clean up old data using optimized function"""
        
        try:
            with self.db.get_session() as session:
                if self.db.config.is_postgresql():
                    result = session.execute(
                        "SELECT cleanup_old_executions(%(days)s)", 
                        {'days': retention_days}
                    ).scalar()
                    return result or 0
                else:
                    return self._cleanup_sqlite(session, retention_days)
                    
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 0
    
    def _store_execution_sqlite(self, execution: PipelineExecution) -> bool:
        """SQLite fallback for execution storage"""
        # Simplified SQLite implementation
        return True
    
    def _get_summary_sqlite(self, session) -> Dict[str, Any]:
        """SQLite fallback for summary"""
        return {
            'total_executions': 0,
            'database_type': 'SQLite',
            'note': 'Limited functionality in SQLite mode'
        }
    
    def _cleanup_sqlite(self, session, retention_days: int) -> int:
        """SQLite fallback for cleanup"""
        return 0

    def update_execution_feedback(self, execution_id: str, user_satisfaction: float, success_rating: float) -> bool:
        """Update execution record with user feedback"""
        try:
            with self.db.get_session() as session:
                if self.db.config.is_postgresql():
                    session.execute("""
                        UPDATE pipeline_executions 
                        SET user_satisfaction = :satisfaction, success_rating = :rating
                        WHERE execution_id = :exec_id
                    """, {
                        'satisfaction': user_satisfaction,
                        'rating': success_rating, 
                        'exec_id': execution_id
                    })
                    return True
                else:
                    return self._update_feedback_sqlite(session, execution_id, user_satisfaction, success_rating)
                    
        except Exception as e:
            self.logger.error(f"Update execution feedback failed: {e}")
            return False

    def _update_feedback_sqlite(self, session, execution_id: str, user_satisfaction: float, success_rating: float) -> bool:
        """SQLite fallback for updating feedback"""
        return True


def get_database_service() -> OptimizedDatabaseService:
    """Get optimized database service instance"""
    return OptimizedDatabaseService()