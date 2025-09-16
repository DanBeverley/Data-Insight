"""Adaptive Learning System for Continuous Model Improvement with Persistent Meta-Learning"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import pickle
import warnings
from datetime import datetime
import hashlib
import json
warnings.filterwarnings('ignore')

from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from ..data_quality.drift_monitor import ComprehensiveDriftMonitor, DriftResult
from ..model_selection.performance_validator import ProductionModelValidator, ModelPerformance, TaskType
from .persistent_storage import (
    PersistentMetaDatabase, 
    DatasetCharacteristics, 
    ProjectConfig, 
    PipelineExecution,
    create_dataset_characteristics
)

class LearningStrategy(Enum):
    REACTIVE = "reactive"
    PROACTIVE = "proactive"
    HYBRID = "hybrid"

class TriggerType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    TIME_BASED = "time_based"
    MANUAL = "manual"

@dataclass
class AdaptationTrigger:
    trigger_type: TriggerType
    threshold: float
    current_value: float
    triggered: bool
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearningEvent:
    event_id: str
    trigger: AdaptationTrigger
    original_performance: float
    new_performance: float
    adaptation_applied: str
    improvement: float
    timestamp: float
    success: bool

@dataclass
class AdaptiveConfig:
    learning_strategy: LearningStrategy = LearningStrategy.HYBRID
    performance_threshold: float = 0.05
    drift_threshold: float = 0.1
    min_samples_for_adaptation: int = 100
    adaptation_frequency_days: int = 7
    max_model_history: int = 10
    confidence_threshold: float = 0.7
    early_stopping_patience: int = 3

class AdaptiveLearningSystem:
    
    def __init__(self, config: Optional[AdaptiveConfig] = None, 
                 enable_persistence: bool = True,
                 db_path: Optional[str] = None):
        self.config = config or AdaptiveConfig()
        self.drift_monitor = ComprehensiveDriftMonitor()
        self.validator = ProductionModelValidator()
        
        # In-memory storage (for backward compatibility)
        self.model_history: List[Dict[str, Any]] = []
        self.performance_history: List[float] = []
        self.learning_events: List[LearningEvent] = []
        self.baseline_performance: Optional[float] = None
        
        # Persistent storage
        self.enable_persistence = enable_persistence
        if enable_persistence:
            self.meta_db = PersistentMetaDatabase(db_path or "data_insight_meta.db")
        else:
            self.meta_db = None

        # Raw execution data storage
        self.execution_history: List[Dict[str, Any]] = []

    def capture_execution_data(self, session_id: str, code: str, execution_time: float,
                             success: bool, output: str, error: str = "", context: Dict[str, Any] = None):
        """Store raw execution data without classification"""
        execution_data = {
            'session_id': session_id,
            'code': code,
            'execution_time': execution_time,
            'success': success,
            'output': output,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }

        self.execution_history.append(execution_data)

        # Keep only recent executions (last 1000)
        self.execution_history = self.execution_history[-1000:]

        # Also store in knowledge graph for unified access
        try:
            import sys, os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from knowledge_graph.service import KnowledgeGraphService
            graph_service = KnowledgeGraphService()
            graph_service.add_execution_data(execution_data)
        except Exception as e:
            print(f"Failed to store execution in knowledge graph: {e}")

    def get_execution_history(self, session_id: str = None, success_only: bool = False) -> List[Dict[str, Any]]:
        """Get raw execution history data"""
        history = self.execution_history

        if session_id:
            history = [ex for ex in history if ex['session_id'] == session_id]

        if success_only:
            history = [ex for ex in history if ex['success']]

        return history
        
    def monitor_and_adapt(self, current_model, X_new: pd.DataFrame, y_new: pd.Series,
                         task_type: TaskType, algorithm_name: str = "") -> Dict[str, Any]:
        
        adaptation_results = {
            'triggers_detected': [],
            'adaptations_applied': [],
            'performance_change': 0.0,
            'recommendation': 'continue_monitoring'
        }
        
        triggers = self._evaluate_triggers(current_model, X_new, y_new, task_type)
        adaptation_results['triggers_detected'] = [t.trigger_type.value for t in triggers if t.triggered]
        
        if not any(t.triggered for t in triggers):
            return adaptation_results
        
        if self.config.learning_strategy in [LearningStrategy.REACTIVE, LearningStrategy.HYBRID]:
            reactive_adaptation = self._apply_reactive_adaptation(
                current_model, X_new, y_new, triggers, task_type, algorithm_name
            )
            adaptation_results.update(reactive_adaptation)
        
        if self.config.learning_strategy in [LearningStrategy.PROACTIVE, LearningStrategy.HYBRID]:
            proactive_adaptation = self._apply_proactive_adaptation(
                current_model, X_new, y_new, task_type, algorithm_name
            )
            if proactive_adaptation['improvement'] > reactive_adaptation.get('improvement', 0):
                adaptation_results.update(proactive_adaptation)
        
        self._update_learning_history(adaptation_results, triggers)
        
        return adaptation_results
    
    def _evaluate_triggers(self, model, X: pd.DataFrame, y: pd.Series, 
                          task_type: TaskType) -> List[AdaptationTrigger]:
        
        triggers = []
        
        performance_trigger = self._evaluate_performance_trigger(model, X, y, task_type)
        triggers.append(performance_trigger)
        
        drift_trigger = self._evaluate_drift_trigger(X)
        triggers.append(drift_trigger)
        
        time_trigger = self._evaluate_time_trigger()
        triggers.append(time_trigger)
        
        return triggers
    
    def _evaluate_performance_trigger(self, model, X: pd.DataFrame, y: pd.Series,
                                    task_type: TaskType) -> AdaptationTrigger:
        
        try:
            current_performance = self._measure_performance(model, X, y, task_type)
            
            if self.baseline_performance is None:
                self.baseline_performance = current_performance
                performance_degradation = 0.0
                triggered = False
            else:
                performance_degradation = self.baseline_performance - current_performance
                triggered = performance_degradation > self.config.performance_threshold
            
            confidence = min(1.0, abs(performance_degradation) / self.config.performance_threshold) if triggered else 0.5
            
            return AdaptationTrigger(
                trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                threshold=self.config.performance_threshold,
                current_value=performance_degradation,
                triggered=triggered,
                confidence=confidence,
                metadata={
                    'baseline_performance': self.baseline_performance,
                    'current_performance': current_performance
                }
            )
            
        except Exception as e:
            return AdaptationTrigger(
                trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                threshold=self.config.performance_threshold,
                current_value=0.0,
                triggered=False,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _evaluate_drift_trigger(self, X: pd.DataFrame) -> AdaptationTrigger:
        
        try:
            if not hasattr(self, '_reference_data') or self._reference_data is None:
                self._reference_data = X.copy()
                return AdaptationTrigger(
                    trigger_type=TriggerType.DATA_DRIFT,
                    threshold=self.config.drift_threshold,
                    current_value=0.0,
                    triggered=False,
                    confidence=0.0,
                    metadata={'status': 'establishing_baseline'}
                )
            
            self.drift_monitor.fit_reference(self._reference_data)
            drift_results = self.drift_monitor.detect_drift(X)
            
            if not drift_results:
                max_drift_score = 0.0
                triggered = False
            else:
                max_drift_score = max(drift.drift_score for drift in drift_results)
                triggered = max_drift_score > self.config.drift_threshold
            
            confidence = min(1.0, max_drift_score / self.config.drift_threshold) if triggered else 0.5
            
            return AdaptationTrigger(
                trigger_type=TriggerType.DATA_DRIFT,
                threshold=self.config.drift_threshold,
                current_value=max_drift_score,
                triggered=triggered,
                confidence=confidence,
                metadata={
                    'drift_count': len(drift_results),
                    'drift_details': [d.description for d in drift_results[:3]]
                }
            )
            
        except Exception as e:
            return AdaptationTrigger(
                trigger_type=TriggerType.DATA_DRIFT,
                threshold=self.config.drift_threshold,
                current_value=0.0,
                triggered=False,
                confidence=0.0,
                metadata={'error': str(e)}
            )
    
    def _evaluate_time_trigger(self) -> AdaptationTrigger:
        
        current_time = time.time()
        
        if not hasattr(self, '_last_adaptation_time'):
            self._last_adaptation_time = current_time
            time_since_adaptation = 0.0
            triggered = False
        else:
            time_since_adaptation = (current_time - self._last_adaptation_time) / (24 * 3600)
            triggered = time_since_adaptation >= self.config.adaptation_frequency_days
        
        confidence = min(1.0, time_since_adaptation / self.config.adaptation_frequency_days) if triggered else 0.5
        
        return AdaptationTrigger(
            trigger_type=TriggerType.TIME_BASED,
            threshold=self.config.adaptation_frequency_days,
            current_value=time_since_adaptation,
            triggered=triggered,
            confidence=confidence,
            metadata={
                'days_since_adaptation': time_since_adaptation,
                'last_adaptation': self._last_adaptation_time if hasattr(self, '_last_adaptation_time') else None
            }
        )
    
    def _apply_reactive_adaptation(self, model, X: pd.DataFrame, y: pd.Series,
                                 triggers: List[AdaptationTrigger], task_type: TaskType,
                                 algorithm_name: str) -> Dict[str, Any]:
        
        if len(X) < self.config.min_samples_for_adaptation:
            return {
                'adaptation_type': 'reactive',
                'success': False,
                'reason': 'insufficient_data',
                'improvement': 0.0
            }
        
        original_performance = self._measure_performance(model, X, y, task_type)
        
        adapted_model = clone(model)
        adapted_model.fit(X, y)
        
        new_performance = self._measure_performance(adapted_model, X, y, task_type)
        improvement = new_performance - original_performance
        
        if improvement > 0:
            self._store_model_version(adapted_model, new_performance, algorithm_name)
            self.baseline_performance = new_performance
            
            return {
                'adaptation_type': 'reactive',
                'success': True,
                'improvement': improvement,
                'new_performance': new_performance,
                'adapted_model': adapted_model
            }
        else:
            return {
                'adaptation_type': 'reactive',
                'success': False,
                'reason': 'no_improvement',
                'improvement': improvement
            }
    
    def _apply_proactive_adaptation(self, model, X: pd.DataFrame, y: pd.Series,
                                  task_type: TaskType, algorithm_name: str) -> Dict[str, Any]:
        
        if len(X) < self.config.min_samples_for_adaptation:
            return {
                'adaptation_type': 'proactive',
                'success': False,
                'reason': 'insufficient_data',
                'improvement': 0.0
            }
        
        original_performance = self._measure_performance(model, X, y, task_type)
        
        best_model = model
        best_performance = original_performance
        adaptation_strategy = "none"
        
        strategies = [
            ('incremental_learning', self._incremental_learning_adaptation),
            ('hyperparameter_adjustment', self._hyperparameter_adaptation),
            ('ensemble_update', self._ensemble_adaptation)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                adapted_model = strategy_func(model, X, y, task_type)
                performance = self._measure_performance(adapted_model, X, y, task_type)
                
                if performance > best_performance:
                    best_model = adapted_model
                    best_performance = performance
                    adaptation_strategy = strategy_name
                    
            except Exception as e:
                continue
        
        improvement = best_performance - original_performance
        
        if improvement > 0:
            self._store_model_version(best_model, best_performance, algorithm_name)
            
            return {
                'adaptation_type': 'proactive',
                'success': True,
                'strategy': adaptation_strategy,
                'improvement': improvement,
                'new_performance': best_performance,
                'adapted_model': best_model
            }
        else:
            return {
                'adaptation_type': 'proactive',
                'success': False,
                'reason': 'no_improvement',
                'improvement': improvement
            }
    
    def _incremental_learning_adaptation(self, model, X: pd.DataFrame, y: pd.Series,
                                       task_type: TaskType):
        
        if hasattr(model, 'partial_fit'):
            adapted_model = clone(model)
            adapted_model.partial_fit(X, y)
            return adapted_model
        else:
            adapted_model = clone(model)
            adapted_model.fit(X, y)
            return adapted_model
    
    def _hyperparameter_adaptation(self, model, X: pd.DataFrame, y: pd.Series,
                                 task_type: TaskType):
        
        adapted_model = clone(model)
        current_params = model.get_params()
        
        if hasattr(model, 'learning_rate') and 'learning_rate' in current_params:
            new_lr = current_params['learning_rate'] * 0.9
            adapted_model.set_params(learning_rate=max(0.001, new_lr))
        
        if hasattr(model, 'n_estimators') and 'n_estimators' in current_params:
            new_estimators = min(current_params['n_estimators'] + 10, 200)
            adapted_model.set_params(n_estimators=new_estimators)
        
        adapted_model.fit(X, y)
        return adapted_model
    
    def _ensemble_adaptation(self, model, X: pd.DataFrame, y: pd.Series,
                           task_type: TaskType):
        
        if len(self.model_history) == 0:
            return clone(model)
        
        recent_models = [item['model'] for item in self.model_history[-3:]]
        recent_models.append(model)
        
        ensemble_predictions = []
        for m in recent_models:
            try:
                pred = m.predict(X)
                ensemble_predictions.append(pred)
            except:
                continue
        
        if len(ensemble_predictions) < 2:
            return clone(model)
        
        if task_type == TaskType.CLASSIFICATION:
            from scipy import stats
            ensemble_pred = stats.mode(np.array(ensemble_predictions), axis=0)[0].flatten()
        else:
            ensemble_pred = np.mean(ensemble_predictions, axis=0)
        
        class EnsembleWrapper:
            def __init__(self, models, task_type):
                self.models = models
                self.task_type = task_type
                
            def predict(self, X):
                predictions = []
                for model in self.models:
                    try:
                        pred = model.predict(X)
                        predictions.append(pred)
                    except:
                        continue
                
                if self.task_type == TaskType.CLASSIFICATION:
                    from scipy import stats
                    return stats.mode(np.array(predictions), axis=0)[0].flatten()
                else:
                    return np.mean(predictions, axis=0)
            
            def get_params(self):
                return {'ensemble_size': len(self.models)}
        
        return EnsembleWrapper(recent_models, task_type)
    
    def _measure_performance(self, model, X: pd.DataFrame, y: pd.Series, 
                           task_type: TaskType) -> float:
        
        try:
            if len(X) < 20:
                return 0.0
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_test)
            
            if task_type == TaskType.CLASSIFICATION:
                return accuracy_score(y_test, y_pred)
            else:
                return r2_score(y_test, y_pred)
                
        except Exception:
            return 0.0
    
    def _store_model_version(self, model, performance: float, algorithm_name: str):
        
        model_info = {
            'model': model,
            'performance': performance,
            'algorithm_name': algorithm_name,
            'timestamp': time.time(),
            'version': len(self.model_history) + 1
        }
        
        self.model_history.append(model_info)
        self.performance_history.append(performance)
        
        if len(self.model_history) > self.config.max_model_history:
            self.model_history.pop(0)
            self.performance_history.pop(0)
    
    def _update_learning_history(self, adaptation_results: Dict[str, Any], 
                               triggers: List[AdaptationTrigger]):
        
        event = LearningEvent(
            event_id=f"adapt_{len(self.learning_events)}_{int(time.time())}",
            trigger=triggers[0] if triggers else None,
            original_performance=self.baseline_performance or 0.0,
            new_performance=adaptation_results.get('new_performance', 0.0),
            adaptation_applied=adaptation_results.get('adaptation_type', 'none'),
            improvement=adaptation_results.get('improvement', 0.0),
            timestamp=time.time(),
            success=adaptation_results.get('success', False)
        )
        
        self.learning_events.append(event)
        
        if len(self.learning_events) > 100:
            self.learning_events.pop(0)
        
        self._last_adaptation_time = time.time()
    
    def get_learning_insights(self) -> Dict[str, Any]:
        
        if not self.learning_events:
            return {'status': 'no_learning_events'}
        
        successful_adaptations = [e for e in self.learning_events if e.success]
        total_improvement = sum(e.improvement for e in successful_adaptations)
        
        trigger_counts = {}
        for event in self.learning_events:
            if event.trigger:
                trigger_type = event.trigger.trigger_type.value
                trigger_counts[trigger_type] = trigger_counts.get(trigger_type, 0) + 1
        
        adaptation_success_rate = len(successful_adaptations) / len(self.learning_events)
        
        recent_events = self.learning_events[-10:]
        recent_success_rate = len([e for e in recent_events if e.success]) / len(recent_events) if recent_events else 0
        
        return {
            'total_adaptations': len(self.learning_events),
            'successful_adaptations': len(successful_adaptations),
            'success_rate': adaptation_success_rate,
            'recent_success_rate': recent_success_rate,
            'total_improvement': total_improvement,
            'average_improvement': total_improvement / len(successful_adaptations) if successful_adaptations else 0,
            'trigger_distribution': trigger_counts,
            'current_baseline': self.baseline_performance,
            'model_versions_stored': len(self.model_history),
            'learning_trend': self._calculate_learning_trend()
        }
    
    def _calculate_learning_trend(self) -> str:
        
        if len(self.performance_history) < 3:
            return 'insufficient_data'
        
        recent_performance = self.performance_history[-3:]
        
        if all(recent_performance[i] >= recent_performance[i-1] for i in range(1, len(recent_performance))):
            return 'improving'
        elif all(recent_performance[i] <= recent_performance[i-1] for i in range(1, len(recent_performance))):
            return 'declining'
        else:
            return 'stable'
    
    def rollback_to_best_model(self) -> Optional[Any]:
        
        if not self.model_history:
            return None
        
        best_model_info = max(self.model_history, key=lambda x: x['performance'])
        self.baseline_performance = best_model_info['performance']
        
        return best_model_info['model']
    
    def export_learning_state(self) -> Dict[str, Any]:
        base_state = {
            'baseline_performance': self.baseline_performance,
            'performance_history': self.performance_history,
            'learning_events_summary': [
                {
                    'event_id': e.event_id,
                    'success': e.success,
                    'improvement': e.improvement,
                    'timestamp': e.timestamp
                }
                for e in self.learning_events
            ],
            'config': self.config.__dict__,
            'insights': self.get_learning_insights()
        }
        
        # Add persistent storage summary if available
        if self.meta_db:
            base_state['persistent_summary'] = self.meta_db.get_system_learning_summary()
        
        return base_state
    
    def record_pipeline_execution(self, 
                                 dataset: pd.DataFrame,
                                 target_column: str,
                                 project_config: Dict[str, Any],
                                 pipeline_results: Dict[str, Any],
                                 session_id: str = None,
                                 domain: str = "general") -> bool:
        """Record complete pipeline execution for meta-learning"""
        
        if not self.meta_db:
            return False
        
        try:
            # Create dataset characteristics
            dataset_chars = create_dataset_characteristics(dataset, target_column, domain)
            
            # Create project configuration fingerprint
            config_str = json.dumps(project_config, sort_keys=True)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            proj_config = ProjectConfig(
                objective=project_config.get('objective', 'accuracy'),
                domain=domain,
                constraints=project_config.get('constraints', {}),
                config_hash=config_hash,
                strategy_applied=project_config.get('strategy_applied', 'default'),
                feature_engineering_enabled=project_config.get('feature_engineering_enabled', True),
                feature_selection_enabled=project_config.get('feature_selection_enabled', True),
                security_level=project_config.get('security_level', 'standard'),
                privacy_level=project_config.get('privacy_level', 'medium'),
                compliance_requirements=project_config.get('compliance_requirements', [])
            )
            
            # Create pipeline execution record
            execution = PipelineExecution(
                execution_id=f"exec_{session_id}_{int(time.time())}" if session_id else f"exec_{int(time.time())}",
                session_id=session_id or f"session_{int(time.time())}",
                dataset_characteristics=dataset_chars,
                project_config=proj_config,
                pipeline_stages=pipeline_results.get('pipeline_stages', []),
                execution_time=pipeline_results.get('execution_time', 0.0),
                final_performance=pipeline_results.get('performance_metrics', {}),
                trust_score=pipeline_results.get('trust_score', 0.0),
                validation_success=pipeline_results.get('validation_success', False),
                budget_compliance_rate=pipeline_results.get('budget_compliance_rate', 0.0),
                trade_off_efficiency=pipeline_results.get('trade_off_efficiency', 0.0),
                user_satisfaction=pipeline_results.get('user_satisfaction'),
                success_rating=pipeline_results.get('success_rating', 0.0),
                error_count=pipeline_results.get('error_count', 0),
                recovery_attempts=pipeline_results.get('recovery_attempts', 0),
                timestamp=datetime.now(),
                metadata=pipeline_results.get('metadata', {})
            )
            
            return self.meta_db.store_pipeline_execution(execution)
            
        except Exception as e:
            print(f"Failed to record pipeline execution: {e}")
            return False
    
    def get_strategy_recommendations(self, 
                                   dataset: pd.DataFrame,
                                   target_column: str = None,
                                   objective: str = "accuracy",
                                   domain: str = "general") -> Dict[str, Any]:
        """Get intelligent strategy recommendations based on historical learning"""
        
        if not self.meta_db:
            return self._get_heuristic_recommendations(dataset, objective, domain)
        
        try:
            # Create dataset characteristics for similarity matching
            dataset_chars = create_dataset_characteristics(dataset, target_column, domain)
            
            # Get recommendations from persistent storage
            recommendations = self.meta_db.get_recommended_strategies(
                dataset_chars, objective, domain
            )
            
            # Enhance with real-time adaptive insights
            if self.learning_events:
                recent_success_rate = len([e for e in self.learning_events[-10:] if e.success]) / min(10, len(self.learning_events))
                recommendations['adaptive_insights'] = {
                    'recent_adaptation_success_rate': recent_success_rate,
                    'learning_trend': self._calculate_learning_trend(),
                    'current_baseline_performance': self.baseline_performance
                }
            
            return recommendations
            
        except Exception as e:
            print(f"Failed to get strategy recommendations: {e}")
            return self._get_heuristic_recommendations(dataset, objective, domain)
    
    def _get_heuristic_recommendations(self, dataset: pd.DataFrame, 
                                     objective: str, domain: str) -> Dict[str, Any]:
        """Fallback heuristic recommendations when persistent storage unavailable"""
        n_samples, n_features = dataset.shape
        
        return {
            'recommended_strategies': [{
                'strategy_name': 'Heuristic_Default',
                'config': {
                    'feature_engineering_enabled': n_features < 100 and n_samples > 500,
                    'feature_selection_enabled': n_features > 20,
                    'security_level': 'standard',
                    'privacy_level': 'medium'
                },
                'expected_performance': 0.75,
                'expected_execution_time': n_samples * 0.001,
                'confidence': 0.6,
                'evidence': {'type': 'heuristic', 'dataset_size': n_samples}
            }],
            'meta_insights': {
                'recommendation_basis': 'heuristic_fallback',
                'dataset_complexity': 'high' if n_features > 50 else 'medium'
            },
            'confidence_score': 0.6
        }
    
    def learn_from_pipeline_feedback(self, execution_id: str, feedback: Dict[str, Any]) -> bool:
        """Learn from user feedback on pipeline execution quality"""
        
        if not self.meta_db:
            return False
        
        try:
            # Record user feedback
            success = self.meta_db.record_user_feedback(execution_id, feedback)
            
            # Update internal learning based on feedback
            if feedback.get('user_rating', 0) < 3.0:  # Poor rating
                # Reduce confidence in similar strategies
                self._adjust_strategy_confidence(execution_id, -0.1)
            elif feedback.get('user_rating', 0) >= 4.0:  # Good rating  
                # Increase confidence in similar strategies
                self._adjust_strategy_confidence(execution_id, 0.1)
            
            return success
            
        except Exception as e:
            print(f"Failed to process feedback: {e}")
            return False
    
    def _adjust_strategy_confidence(self, execution_id: str, adjustment: float):
        """Adjust confidence scores based on user feedback (placeholder for future enhancement)"""
        # This would query the database to find similar executions and adjust their confidence
        # For now, we just log the adjustment intent
        pass
    
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning insights across all recorded executions"""
        
        base_insights = self.get_learning_insights()
        
        if self.meta_db:
            persistent_insights = self.meta_db.get_system_learning_summary()
            return {
                'session_insights': base_insights,
                'global_insights': persistent_insights,
                'learning_maturity': persistent_insights.get('meta_learning_maturity', 0.0),
                'knowledge_base_size': persistent_insights.get('total_executions', 0),
                'cross_domain_learning': persistent_insights.get('domains_covered', 0)
            }
        else:
            return {
                'session_insights': base_insights,
                'global_insights': {'status': 'persistence_disabled'},
                'learning_maturity': 0.0
            }

def create_adaptive_system(config: Optional[AdaptiveConfig] = None) -> AdaptiveLearningSystem:
    return AdaptiveLearningSystem(config)