"""Advanced Algorithm Portfolio with Intelligent Selection"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"

class ComplexityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class DataCharacteristics:
    n_samples: int
    n_features: int
    n_classes: Optional[int]
    has_categorical: bool
    has_missing: bool
    is_imbalanced: bool
    feature_correlation: float
    noise_level: float
    complexity_level: ComplexityLevel
    domain_hints: List[str] = field(default_factory=list)

@dataclass
class AlgorithmRecommendation:
    algorithm_name: str
    model_class: Any
    default_params: Dict[str, Any]
    suitability_score: float
    reasoning: List[str]
    computational_cost: str
    interpretability: str

class IntelligentAlgorithmPortfolio:
    
    def __init__(self):
        self.classification_algorithms = self._init_classification_algorithms()
        self.regression_algorithms = self._init_regression_algorithms()
        self.clustering_algorithms = self._init_clustering_algorithms()
        
    def _init_classification_algorithms(self) -> Dict[str, Dict]:
        return {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {'n_estimators': 100, 'random_state': 42},
                'strengths': ['robust', 'handles_missing', 'feature_importance', 'non_linear'],
                'weaknesses': ['memory_intensive'],
                'cost': 'medium',
                'interpretability': 'medium'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {'n_estimators': 100, 'random_state': 42},
                'strengths': ['high_accuracy', 'handles_missing', 'non_linear'],
                'weaknesses': ['overfitting_prone', 'slow_training'],
                'cost': 'high',
                'interpretability': 'low'
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {'random_state': 42, 'max_iter': 1000},
                'strengths': ['fast', 'interpretable', 'probabilistic', 'linear'],
                'weaknesses': ['assumes_linearity'],
                'cost': 'low',
                'interpretability': 'high'
            },
            'svm': {
                'model': SVC,
                'params': {'random_state': 42, 'probability': True},
                'strengths': ['high_accuracy', 'kernel_trick', 'robust'],
                'weaknesses': ['slow_large_data', 'memory_intensive'],
                'cost': 'high',
                'interpretability': 'low'
            },
            'knn': {
                'model': KNeighborsClassifier,
                'params': {'n_neighbors': 5},
                'strengths': ['simple', 'non_linear', 'no_assumptions'],
                'weaknesses': ['curse_dimensionality', 'slow_prediction'],
                'cost': 'medium',
                'interpretability': 'medium'
            },
            'naive_bayes': {
                'model': GaussianNB,
                'params': {},
                'strengths': ['fast', 'small_data', 'probabilistic'],
                'weaknesses': ['independence_assumption'],
                'cost': 'low',
                'interpretability': 'high'
            },
            'decision_tree': {
                'model': DecisionTreeClassifier,
                'params': {'random_state': 42, 'max_depth': 10},
                'strengths': ['interpretable', 'handles_missing', 'non_linear'],
                'weaknesses': ['overfitting_prone', 'unstable'],
                'cost': 'low',
                'interpretability': 'high'
            },
            'neural_network': {
                'model': MLPClassifier,
                'params': {'random_state': 42, 'max_iter': 500},
                'strengths': ['non_linear', 'flexible', 'high_capacity'],
                'weaknesses': ['black_box', 'requires_scaling', 'overfitting_prone'],
                'cost': 'high',
                'interpretability': 'low'
            },
            'ada_boost': {
                'model': AdaBoostClassifier,
                'params': {'random_state': 42, 'n_estimators': 50},
                'strengths': ['good_weak_learners', 'handles_noise'],
                'weaknesses': ['sensitive_outliers'],
                'cost': 'medium',
                'interpretability': 'medium'
            }
        }
    
    def _init_regression_algorithms(self) -> Dict[str, Dict]:
        return {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {'n_estimators': 100, 'random_state': 42},
                'strengths': ['robust', 'handles_missing', 'feature_importance', 'non_linear'],
                'weaknesses': ['memory_intensive'],
                'cost': 'medium',
                'interpretability': 'medium'
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'params': {'n_estimators': 100, 'random_state': 42},
                'strengths': ['high_accuracy', 'handles_missing', 'non_linear'],
                'weaknesses': ['overfitting_prone', 'slow_training'],
                'cost': 'high',
                'interpretability': 'low'
            },
            'linear_regression': {
                'model': LinearRegression,
                'params': {},
                'strengths': ['fast', 'interpretable', 'simple'],
                'weaknesses': ['assumes_linearity'],
                'cost': 'low',
                'interpretability': 'high'
            },
            'ridge': {
                'model': Ridge,
                'params': {'alpha': 1.0, 'random_state': 42},
                'strengths': ['handles_multicollinearity', 'regularized'],
                'weaknesses': ['assumes_linearity'],
                'cost': 'low',
                'interpretability': 'high'
            },
            'lasso': {
                'model': Lasso,
                'params': {'alpha': 1.0, 'random_state': 42},
                'strengths': ['feature_selection', 'regularized'],
                'weaknesses': ['assumes_linearity'],
                'cost': 'low',
                'interpretability': 'high'
            },
            'svr': {
                'model': SVR,
                'params': {},
                'strengths': ['kernel_trick', 'robust'],
                'weaknesses': ['slow_large_data', 'parameter_sensitive'],
                'cost': 'high',
                'interpretability': 'low'
            },
            'knn': {
                'model': KNeighborsRegressor,
                'params': {'n_neighbors': 5},
                'strengths': ['simple', 'non_linear', 'no_assumptions'],
                'weaknesses': ['curse_dimensionality', 'slow_prediction'],
                'cost': 'medium',
                'interpretability': 'medium'
            },
            'decision_tree': {
                'model': DecisionTreeRegressor,
                'params': {'random_state': 42, 'max_depth': 10},
                'strengths': ['interpretable', 'handles_missing', 'non_linear'],
                'weaknesses': ['overfitting_prone', 'unstable'],
                'cost': 'low',
                'interpretability': 'high'
            },
            'neural_network': {
                'model': MLPRegressor,
                'params': {'random_state': 42, 'max_iter': 500},
                'strengths': ['non_linear', 'flexible', 'high_capacity'],
                'weaknesses': ['black_box', 'requires_scaling', 'overfitting_prone'],
                'cost': 'high',
                'interpretability': 'low'
            },
            'ada_boost': {
                'model': AdaBoostRegressor,
                'params': {'random_state': 42, 'n_estimators': 50},
                'strengths': ['good_weak_learners', 'handles_noise'],
                'weaknesses': ['sensitive_outliers'],
                'cost': 'medium',
                'interpretability': 'medium'
            }
        }
    
    def _init_clustering_algorithms(self) -> Dict[str, Dict]:
        return {
            'kmeans': {
                'model': KMeans,
                'params': {'n_clusters': 3, 'random_state': 42},
                'strengths': ['fast', 'simple', 'spherical_clusters'],
                'weaknesses': ['assumes_spherical', 'requires_k'],
                'cost': 'low',
                'interpretability': 'high'
            },
            'dbscan': {
                'model': DBSCAN,
                'params': {'eps': 0.5, 'min_samples': 5},
                'strengths': ['arbitrary_shapes', 'handles_noise', 'auto_clusters'],
                'weaknesses': ['parameter_sensitive', 'varying_density'],
                'cost': 'medium',
                'interpretability': 'medium'
            },
            'hierarchical': {
                'model': AgglomerativeClustering,
                'params': {'n_clusters': 3},
                'strengths': ['no_k_assumption', 'deterministic', 'hierarchical'],
                'weaknesses': ['slow_large_data', 'memory_intensive'],
                'cost': 'high',
                'interpretability': 'high'
            }
        }
    
    def analyze_data_characteristics(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> DataCharacteristics:
        n_samples, n_features = X.shape
        n_classes = len(y.unique()) if y is not None else None
        
        has_categorical = any(X.dtypes == 'object') or any(X.dtypes == 'category')
        has_missing = X.isnull().any().any()
        
        is_imbalanced = False
        if y is not None and n_classes is not None:
            class_counts = y.value_counts()
            min_class_ratio = class_counts.min() / class_counts.max()
            is_imbalanced = min_class_ratio < 0.3
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        feature_correlation = 0
        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr().abs()
            feature_correlation = (corr_matrix.values.sum() - len(corr_matrix)) / (len(corr_matrix) ** 2 - len(corr_matrix))
        
        noise_level = self._estimate_noise_level(X)
        complexity_level = self._determine_complexity_level(n_samples, n_features, feature_correlation)
        
        return DataCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            has_categorical=has_categorical,
            has_missing=has_missing,
            is_imbalanced=is_imbalanced,
            feature_correlation=feature_correlation,
            noise_level=noise_level,
            complexity_level=complexity_level
        )
    
    def _estimate_noise_level(self, X: pd.DataFrame) -> float:
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 0.0
        
        noise_indicators = []
        for col in numeric_cols:
            series = X[col].dropna()
            if len(series) > 10:
                cv = series.std() / series.mean() if series.mean() != 0 else 0
                noise_indicators.append(min(cv, 1.0))
        
        return np.mean(noise_indicators) if noise_indicators else 0.0
    
    def _determine_complexity_level(self, n_samples: int, n_features: int, correlation: float) -> ComplexityLevel:
        if n_samples < 1000 and n_features < 10:
            return ComplexityLevel.LOW
        elif n_samples < 10000 and n_features < 100:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.HIGH
    
    def recommend_algorithms(self, data_chars: DataCharacteristics, 
                           task_type: TaskType, 
                           top_k: int = 5) -> List[AlgorithmRecommendation]:
        
        if task_type == TaskType.CLASSIFICATION:
            algorithms = self.classification_algorithms
        elif task_type == TaskType.REGRESSION:
            algorithms = self.regression_algorithms
        else:
            algorithms = self.clustering_algorithms
        
        recommendations = []
        for name, config in algorithms.items():
            score, reasoning = self._calculate_suitability_score(name, config, data_chars, task_type)
            
            recommendation = AlgorithmRecommendation(
                algorithm_name=name,
                model_class=config['model'],
                default_params=config['params'].copy(),
                suitability_score=score,
                reasoning=reasoning,
                computational_cost=config['cost'],
                interpretability=config['interpretability']
            )
            recommendations.append(recommendation)
        
        recommendations.sort(key=lambda x: x.suitability_score, reverse=True)
        return recommendations[:top_k]
    
    def _calculate_suitability_score(self, algorithm_name: str, config: Dict, 
                                   data_chars: DataCharacteristics, 
                                   task_type: TaskType) -> Tuple[float, List[str]]:
        
        score = 50.0
        reasoning = []
        strengths = config['strengths']
        weaknesses = config['weaknesses']
        
        if data_chars.n_samples < 1000:
            if 'fast' in strengths or 'simple' in strengths:
                score += 15
                reasoning.append("Good for small datasets")
            if 'slow_training' in weaknesses or 'memory_intensive' in weaknesses:
                score -= 10
                reasoning.append("May be overkill for small data")
        
        if data_chars.n_samples > 10000:
            if 'slow_large_data' in weaknesses:
                score -= 20
                reasoning.append("Not optimal for large datasets")
            if algorithm_name in ['random_forest', 'gradient_boosting']:
                score += 10
                reasoning.append("Scales well with large data")
        
        if data_chars.n_features > 50:
            if 'curse_dimensionality' in weaknesses:
                score -= 15
                reasoning.append("Struggles with high dimensions")
            if 'feature_selection' in strengths:
                score += 10
                reasoning.append("Handles high dimensions well")
        
        if data_chars.has_missing:
            if 'handles_missing' in strengths:
                score += 15
                reasoning.append("Naturally handles missing values")
            else:
                score -= 5
        
        if data_chars.is_imbalanced and task_type == TaskType.CLASSIFICATION:
            if algorithm_name in ['random_forest', 'gradient_boosting']:
                score += 10
                reasoning.append("Good for imbalanced data")
        
        if data_chars.feature_correlation > 0.7:
            if 'handles_multicollinearity' in strengths:
                score += 10
                reasoning.append("Handles correlated features")
            if algorithm_name == 'linear_regression':
                score -= 10
                reasoning.append("Sensitive to multicollinearity")
        
        if data_chars.noise_level > 0.3:
            if 'robust' in strengths:
                score += 10
                reasoning.append("Robust to noise")
            if 'sensitive_outliers' in weaknesses:
                score -= 10
                reasoning.append("Sensitive to noisy data")
        
        if data_chars.complexity_level == ComplexityLevel.HIGH:
            if 'non_linear' in strengths:
                score += 15
                reasoning.append("Can capture complex patterns")
            if 'assumes_linearity' in weaknesses:
                score -= 15
                reasoning.append("Too simple for complex data")
        
        if data_chars.complexity_level == ComplexityLevel.LOW:
            if 'simple' in strengths or 'interpretable' in strengths:
                score += 10
                reasoning.append("Appropriate complexity level")
            if algorithm_name in ['neural_network', 'svm']:
                score -= 10
                reasoning.append("May overcomplicate simple patterns")
        
        return max(0, min(100, score)), reasoning
    
    def get_ensemble_recommendations(self, base_recommendations: List[AlgorithmRecommendation]) -> Dict[str, Any]:
        if len(base_recommendations) < 2:
            return {}
        
        diverse_algorithms = []
        for rec in base_recommendations:
            if rec.interpretability != 'low' or len(diverse_algorithms) < 2:
                diverse_algorithms.append(rec)
            if len(diverse_algorithms) >= 3:
                break
        
        ensemble_strategies = {
            'voting': {
                'algorithms': [rec.algorithm_name for rec in diverse_algorithms[:3]],
                'reasoning': "Combines predictions from diverse algorithms",
                'complexity': 'medium'
            }
        }
        
        if len([rec for rec in base_recommendations if rec.computational_cost != 'high']) >= 2:
            ensemble_strategies['stacking'] = {
                'algorithms': [rec.algorithm_name for rec in base_recommendations[:2]],
                'meta_learner': 'logistic_regression',
                'reasoning': "Uses meta-learner to combine base models",
                'complexity': 'high'
            }
        
        return ensemble_strategies
    
    def create_model_instance(self, recommendation: AlgorithmRecommendation, custom_params: Optional[Dict] = None) -> Any:
        params = recommendation.default_params.copy()
        if custom_params:
            params.update(custom_params)
        
        try:
            return recommendation.model_class(**params)
        except Exception as e:
            fallback_params = {k: v for k, v in params.items() if k in ['random_state', 'n_estimators', 'max_iter']}
            return recommendation.model_class(**fallback_params)
    
    def quick_evaluation(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                        task_type: TaskType, cv_folds: int = 3) -> Dict[str, float]:
        
        try:
            if task_type == TaskType.CLASSIFICATION:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
                return {
                    'mean_accuracy': scores.mean(),
                    'std_accuracy': scores.std(),
                    'min_accuracy': scores.min(),
                    'max_accuracy': scores.max()
                }
            else:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
                return {
                    'mean_r2': scores.mean(),
                    'std_r2': scores.std(),
                    'min_r2': scores.min(),
                    'max_r2': scores.max()
                }
        except Exception:
            return {'error': True, 'mean_score': 0.0}

def select_best_algorithms(X: pd.DataFrame, y: Optional[pd.Series] = None, 
                          task_type: Optional[TaskType] = None,
                          top_k: int = 5) -> Tuple[List[AlgorithmRecommendation], DataCharacteristics]:
    
    portfolio = IntelligentAlgorithmPortfolio()
    
    if task_type is None:
        if y is None:
            task_type = TaskType.CLUSTERING
        else:
            task_type = TaskType.CLASSIFICATION if y.nunique() <= 20 else TaskType.REGRESSION
    
    data_chars = portfolio.analyze_data_characteristics(X, y)
    recommendations = portfolio.recommend_algorithms(data_chars, task_type, top_k)
    
    return recommendations, data_chars