import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

# Lazy imports to avoid memory issues

XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    import xgboost
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

@dataclass
class AlgorithmConfig:
    name: str
    model_class: type
    default_params: Dict[str, Any]
    param_ranges: Dict[str, Tuple[Any, Any]]
    complexity_tier: str  # 'fast', 'balanced', 'complex'
    memory_requirement: str  # 'low', 'medium', 'high'
    interpretability: str  # 'high', 'medium', 'low'
    training_time: str  # 'fast', 'medium', 'slow'
    best_for: List[str]  # Dataset characteristics this algorithm excels at

class BaseAlgorithm(ABC):
    """Base class for algorithm wrappers"""
    
    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.training_time = 0.0
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseAlgorithm':
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        pass

class SklearnAlgorithm(BaseAlgorithm):
    """Wrapper for sklearn-compatible algorithms"""
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'SklearnAlgorithm':
        start_time = time.time()
        
        # Initialize model with parameters
        params = {**self.config.default_params, **kwargs}
        self.model = self.config.model_class(**params)
        
        # Fit model
        self.model.fit(X, y)
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # For binary classification, convert decision function to probabilities
            decision_scores = self.model.decision_function(X)
            if decision_scores.ndim == 1:  # Binary classification
                exp_scores = np.exp(decision_scores)
                proba_pos = exp_scores / (1 + exp_scores)
                return np.column_stack([1 - proba_pos, proba_pos])
        
        return None

class AlgorithmPortfolioManager:
    """Intelligent algorithm selection and management system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.algorithms_registry = None  # Initialize lazily
        self.performance_history: Dict[str, List[float]] = {}
        
    def _get_algorithms_registry(self):
        """Get algorithms registry, initialize if needed"""
        if self.algorithms_registry is None:
            self.algorithms_registry = self._initialize_algorithms()
        return self.algorithms_registry
        
    def _initialize_algorithms(self) -> Dict[str, AlgorithmConfig]:
        """Initialize the portfolio of available algorithms"""
        
        # Import sklearn classes only when needed
        from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        
        algorithms = {
            # Fast Algorithms
            'logistic_regression': AlgorithmConfig(
                name='logistic_regression',
                model_class=LogisticRegression,
                default_params={'random_state': 42, 'max_iter': 1000},
                param_ranges={
                    'C': (0.001, 100),
                    'penalty': (['l1', 'l2', 'elasticnet', 'none'],),
                    'solver': (['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'],)
                },
                complexity_tier='fast',
                memory_requirement='low',
                interpretability='high',
                training_time='fast',
                best_for=['linear_separable', 'high_dimensional', 'sparse_data']
            ),
            
            'linear_regression': AlgorithmConfig(
                name='linear_regression',
                model_class=LinearRegression,
                default_params={},
                param_ranges={
                    'fit_intercept': ([True, False],),
                    'normalize': ([True, False],)
                },
                complexity_tier='fast',
                memory_requirement='low',
                interpretability='high',
                training_time='fast',
                best_for=['linear_relationships', 'interpretable_models']
            ),
            
            'naive_bayes': AlgorithmConfig(
                name='naive_bayes',
                model_class=GaussianNB,
                default_params={},
                param_ranges={
                    'var_smoothing': (1e-12, 1e-1)
                },
                complexity_tier='fast',
                memory_requirement='low',
                interpretability='medium',
                training_time='fast',
                best_for=['small_datasets', 'text_data', 'categorical_features']
            ),
            
            # Balanced Algorithms
            'random_forest': AlgorithmConfig(
                name='random_forest',
                model_class=RandomForestClassifier,
                default_params={'random_state': 42, 'n_jobs': -1},
                param_ranges={
                    'n_estimators': (10, 500),
                    'max_depth': (3, 20),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'max_features': (['auto', 'sqrt', 'log2'],)
                },
                complexity_tier='balanced',
                memory_requirement='medium',
                interpretability='medium',
                training_time='medium',
                best_for=['general_purpose', 'mixed_features', 'robust_performance']
            ),
            
            'random_forest_regressor': AlgorithmConfig(
                name='random_forest_regressor',
                model_class=RandomForestRegressor,
                default_params={'random_state': 42, 'n_jobs': -1},
                param_ranges={
                    'n_estimators': (10, 500),
                    'max_depth': (3, 20),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'max_features': (['auto', 'sqrt', 'log2'],)
                },
                complexity_tier='balanced',
                memory_requirement='medium',
                interpretability='medium',
                training_time='medium',
                best_for=['general_purpose', 'non_linear', 'robust_performance']
            ),
            
            'decision_tree': AlgorithmConfig(
                name='decision_tree',
                model_class=DecisionTreeClassifier,
                default_params={'random_state': 42},
                param_ranges={
                    'max_depth': (3, 15),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'criterion': (['gini', 'entropy'],)
                },
                complexity_tier='balanced',
                memory_requirement='low',
                interpretability='high',
                training_time='fast',
                best_for=['interpretable_models', 'categorical_features', 'rule_extraction']
            ),
            
            # Complex Algorithms
            'gradient_boosting': AlgorithmConfig(
                name='gradient_boosting',
                model_class=GradientBoostingClassifier,
                default_params={'random_state': 42},
                param_ranges={
                    'n_estimators': (50, 300),
                    'learning_rate': (0.01, 0.3),
                    'max_depth': (3, 8),
                    'min_samples_split': (2, 20),
                    'min_samples_leaf': (1, 10),
                    'subsample': (0.6, 1.0)
                },
                complexity_tier='complex',
                memory_requirement='medium',
                interpretability='low',
                training_time='slow',
                best_for=['complex_patterns', 'high_performance', 'structured_data']
            ),
            
            'svm': AlgorithmConfig(
                name='svm',
                model_class=SVC,
                default_params={'random_state': 42, 'probability': True},
                param_ranges={
                    'C': (0.001, 100),
                    'kernel': (['linear', 'rbf', 'poly'],),
                    'gamma': (['scale', 'auto'],),
                    'degree': (2, 5)
                },
                complexity_tier='complex',
                memory_requirement='medium',
                interpretability='low',
                training_time='slow',
                best_for=['high_dimensional', 'non_linear', 'small_datasets']
            ),
            
            'neural_network': AlgorithmConfig(
                name='neural_network',
                model_class=MLPClassifier,
                default_params={'random_state': 42, 'max_iter': 1000},
                param_ranges={
                    'hidden_layer_sizes': ([(50,), (100,), (50, 50), (100, 50)],),
                    'activation': (['relu', 'tanh'],),
                    'alpha': (0.0001, 0.1),
                    'learning_rate': (['constant', 'adaptive'],),
                    'learning_rate_init': (0.0001, 0.01)
                },
                complexity_tier='complex',
                memory_requirement='high',
                interpretability='low',
                training_time='slow',
                best_for=['complex_patterns', 'large_datasets', 'non_linear']
            )
        }
        
        # Add regression variants
        algorithms['ridge'] = AlgorithmConfig(
            name='ridge',
            model_class=Ridge,
            default_params={'random_state': 42},
            param_ranges={'alpha': (0.1, 100), 'solver': (['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],)},
            complexity_tier='fast',
            memory_requirement='low',
            interpretability='high',
            training_time='fast',
            best_for=['regularization', 'multicollinearity', 'overfitting']
        )
        
        algorithms['lasso'] = AlgorithmConfig(
            name='lasso',
            model_class=Lasso,
            default_params={'random_state': 42, 'max_iter': 2000},
            param_ranges={'alpha': (0.001, 10)},
            complexity_tier='fast',
            memory_requirement='low',
            interpretability='high',
            training_time='fast',
            best_for=['feature_selection', 'sparse_solutions', 'high_dimensional']
        )
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            from xgboost import XGBClassifier, XGBRegressor
            algorithms['xgboost'] = AlgorithmConfig(
                name='xgboost',
                model_class=XGBClassifier,
                default_params={'random_state': 42, 'n_jobs': -1},
                param_ranges={
                    'n_estimators': (50, 300),
                    'learning_rate': (0.01, 0.3),
                    'max_depth': (3, 8),
                    'min_child_weight': (1, 10),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0)
                },
                complexity_tier='complex',
                memory_requirement='medium',
                interpretability='low',
                training_time='medium',
                best_for=['structured_data', 'competitions', 'high_performance']
            )
            
            algorithms['xgboost_regressor'] = AlgorithmConfig(
                name='xgboost_regressor',
                model_class=XGBRegressor,
                default_params={'random_state': 42, 'n_jobs': -1},
                param_ranges={
                    'n_estimators': (50, 300),
                    'learning_rate': (0.01, 0.3),
                    'max_depth': (3, 8),
                    'min_child_weight': (1, 10),
                    'subsample': (0.6, 1.0),
                    'colsample_bytree': (0.6, 1.0)
                },
                complexity_tier='complex',
                memory_requirement='medium',
                interpretability='low',
                training_time='medium',
                best_for=['structured_data', 'competitions', 'high_performance']
            )
        
        return algorithms
    
    def select_algorithms(self, dataset_characteristics, constraints: Optional[Dict[str, Any]] = None) -> List[str]:
        """Select optimal algorithms based on dataset characteristics and constraints"""
        
        constraints = constraints or {}
        max_training_time = constraints.get('max_training_time_minutes', 60)
        memory_limit_gb = constraints.get('memory_limit_gb', 8)
        require_interpretability = constraints.get('require_interpretability', False)
        require_probabilities = constraints.get('require_probabilities', False)
        performance_priority = constraints.get('performance_priority', 'balanced')  # 'speed', 'accuracy', 'balanced'
        
        # Start with recommended algorithms from dataset analysis
        candidates = dataset_characteristics.recommended_algorithms.copy()
        
        # Add algorithms based on specific criteria
        task_type = dataset_characteristics.task_type
        n_samples = dataset_characteristics.n_samples
        n_features = dataset_characteristics.n_features
        complexity_score = dataset_characteristics.complexity_score
        
        # Performance priority adjustments
        if performance_priority == 'speed':
            # Prioritize fast algorithms
            fast_algorithms = [name for name, config in self.algorithms_registry.items() 
                             if config.complexity_tier == 'fast' and self._is_compatible(config, task_type)]
            candidates = fast_algorithms + candidates
            
        elif performance_priority == 'accuracy':
            # Prioritize complex algorithms for better accuracy
            complex_algorithms = [name for name, config in self.algorithms_registry.items() 
                                if config.complexity_tier == 'complex' and self._is_compatible(config, task_type)]
            candidates = complex_algorithms + candidates
        
        # Filter by constraints
        filtered_candidates = []
        algorithms_registry = self._get_algorithms_registry()
        
        for algorithm_name in candidates:
            if algorithm_name not in algorithms_registry:
                continue
                
            config = algorithms_registry[algorithm_name]
            
            # Check interpretability requirement
            if require_interpretability and config.interpretability == 'low':
                continue
            
            # Check training time constraint
            if max_training_time < 10 and config.training_time == 'slow':
                continue
            
            # Check memory constraint
            memory_req = self._estimate_algorithm_memory_gb(config, n_samples, n_features)
            if memory_req > memory_limit_gb:
                continue
            
            # Check probability requirement
            if require_probabilities and not self._supports_probabilities(config):
                continue
                
            filtered_candidates.append(algorithm_name)
        
        # Remove duplicates while preserving order
        filtered_candidates = list(dict.fromkeys(filtered_candidates))
        
        # Ensure we have at least some algorithms
        if not filtered_candidates:
            # Fallback to simple, fast algorithms
            fallback_algorithms = ['logistic_regression' if task_type == 'classification' else 'linear_regression']
            filtered_candidates = fallback_algorithms
        
        # Limit to top algorithms based on priority and historical performance
        selected_algorithms = self._prioritize_algorithms(filtered_candidates, dataset_characteristics)
        
        return selected_algorithms[:5]  # Limit to top 5
    
    def _is_compatible(self, config: AlgorithmConfig, task_type: str) -> bool:
        """Check if algorithm is compatible with task type"""
        if task_type == 'classification':
            return 'Classifier' in config.model_class.__name__ or config.name in ['naive_bayes', 'svm']
        else:  # regression
            return 'Regressor' in config.model_class.__name__ or config.name in ['linear_regression', 'ridge', 'lasso']
    
    def _supports_probabilities(self, config: AlgorithmConfig) -> bool:
        """Check if algorithm supports probability predictions"""
        prob_methods = ['predict_proba', 'decision_function']
        return any(hasattr(config.model_class(), method) for method in prob_methods)
    
    def _estimate_algorithm_memory_gb(self, config: AlgorithmConfig, n_samples: int, n_features: int) -> float:
        """Estimate memory requirement for algorithm in GB"""
        base_memory_gb = (n_samples * n_features * 8) / (1024**3)  # Data in GB
        
        multipliers = {
            'low': 1.5,
            'medium': 3.0,
            'high': 6.0
        }
        
        multiplier = multipliers.get(config.memory_requirement, 3.0)
        return base_memory_gb * multiplier
    
    def _prioritize_algorithms(self, candidates: List[str], dataset_characteristics) -> List[str]:
        """Prioritize algorithms based on dataset characteristics and historical performance"""
        
        scored_algorithms = []
        
        algorithms_registry = self._get_algorithms_registry()
        
        for algorithm_name in candidates:
            if algorithm_name not in algorithms_registry:
                continue
                
            config = algorithms_registry[algorithm_name]
            score = self._calculate_algorithm_score(config, dataset_characteristics)
            scored_algorithms.append((algorithm_name, score))
        
        # Sort by score (higher is better)
        scored_algorithms.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, score in scored_algorithms]
    
    def _calculate_algorithm_score(self, config: AlgorithmConfig, dataset_characteristics) -> float:
        """Calculate suitability score for an algorithm"""
        score = 0.0
        
        # Base score from characteristics match
        if dataset_characteristics.linearity_score > 0.7 and 'linear' in config.name:
            score += 0.3
        
        if dataset_characteristics.complexity_score > 0.6 and config.complexity_tier == 'complex':
            score += 0.2
        
        if dataset_characteristics.n_samples < 1000 and config.training_time == 'fast':
            score += 0.2
        
        if dataset_characteristics.separability_score > 0.7 and 'tree' in config.name:
            score += 0.2
        
        # Historical performance bonus
        if config.name in self.performance_history:
            avg_performance = np.mean(self.performance_history[config.name])
            score += avg_performance * 0.3
        
        # Penalty for high complexity on simple datasets
        if dataset_characteristics.complexity_score < 0.3 and config.complexity_tier == 'complex':
            score -= 0.2
        
        return max(0.0, score)
    
    def create_algorithm(self, algorithm_name: str) -> BaseAlgorithm:
        """Create an algorithm instance"""
        algorithms_registry = self._get_algorithms_registry()
        if algorithm_name not in algorithms_registry:
            raise ValueError(f"Algorithm {algorithm_name} not found in registry")
        
        config = algorithms_registry[algorithm_name]
        return SklearnAlgorithm(config)
    
    def get_algorithm_info(self, algorithm_name: str) -> Optional[AlgorithmConfig]:
        """Get algorithm configuration information"""
        algorithms_registry = self._get_algorithms_registry()
        return algorithms_registry.get(algorithm_name)
    
    def update_performance_history(self, algorithm_name: str, performance_score: float):
        """Update historical performance for an algorithm"""
        if algorithm_name not in self.performance_history:
            self.performance_history[algorithm_name] = []
        
        self.performance_history[algorithm_name].append(performance_score)
        
        # Keep only recent performance (last 10 results)
        if len(self.performance_history[algorithm_name]) > 10:
            self.performance_history[algorithm_name] = self.performance_history[algorithm_name][-10:]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of available algorithms in portfolio"""
        algorithms_registry = self._get_algorithms_registry()
        summary = {
            'total_algorithms': len(algorithms_registry),
            'by_complexity': {},
            'by_task_type': {'classification': [], 'regression': []},
            'external_dependencies': {
                'xgboost': XGBOOST_AVAILABLE,
                'lightgbm': LIGHTGBM_AVAILABLE
            }
        }
        
        for name, config in algorithms_registry.items():
            # Count by complexity
            tier = config.complexity_tier
            if tier not in summary['by_complexity']:
                summary['by_complexity'][tier] = 0
            summary['by_complexity'][tier] += 1
            
            # Categorize by task type
            if 'Classifier' in config.model_class.__name__ or name in ['naive_bayes', 'svm']:
                summary['by_task_type']['classification'].append(name)
            else:
                summary['by_task_type']['regression'].append(name)
        
        return summary