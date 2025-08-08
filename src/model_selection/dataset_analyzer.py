import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DatasetCharacteristics:
    n_samples: int
    n_features: int
    n_classes: Optional[int]
    task_type: str
    imbalance_ratio: float
    missing_ratio: float
    categorical_ratio: float
    numerical_ratio: float
    linearity_score: float
    separability_score: float
    noise_level: float
    complexity_score: float
    memory_requirement_mb: float
    recommended_algorithms: List[str]
    constraints: Dict[str, Any]

class DatasetAnalyzer:
    """Intelligent dataset characterization for automated model selection"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analysis_cache: Dict[str, Any] = {}
        
    def analyze_dataset(self, X: pd.DataFrame, y: pd.Series, 
                       constraints: Optional[Dict[str, Any]] = None) -> DatasetCharacteristics:
        """Comprehensive dataset analysis for intelligent model selection"""
        
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("X must be DataFrame, y must be Series")
        
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        constraints = constraints or {}
        
        # Basic characteristics
        n_samples, n_features = X.shape
        task_type = self._determine_task_type(y)
        n_classes = len(y.unique()) if task_type == 'classification' else None
        
        # Data quality analysis
        missing_ratio = X.isnull().sum().sum() / (n_samples * n_features)
        categorical_ratio = sum(X.dtypes == 'object') / n_features
        numerical_ratio = 1.0 - categorical_ratio
        
        # Class imbalance analysis
        imbalance_ratio = self._calculate_imbalance_ratio(y, task_type)
        
        # Prepare numerical data for complexity analysis
        X_numerical = self._prepare_numerical_data(X, y)
        
        # Complexity analysis
        linearity_score = self._assess_linearity(X_numerical, y, task_type)
        separability_score = self._assess_separability(X_numerical, y, task_type)
        noise_level = self._estimate_noise_level(X_numerical, y, task_type)
        complexity_score = self._calculate_complexity_score(
            n_samples, n_features, linearity_score, separability_score, noise_level
        )
        
        # Resource requirements
        memory_requirement = self._estimate_memory_requirement(n_samples, n_features)
        
        # Algorithm recommendations
        recommended_algorithms = self._recommend_algorithms(
            task_type, n_samples, n_features, n_classes, complexity_score, 
            linearity_score, separability_score, constraints
        )
        
        return DatasetCharacteristics(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            task_type=task_type,
            imbalance_ratio=imbalance_ratio,
            missing_ratio=missing_ratio,
            categorical_ratio=categorical_ratio,
            numerical_ratio=numerical_ratio,
            linearity_score=linearity_score,
            separability_score=separability_score,
            noise_level=noise_level,
            complexity_score=complexity_score,
            memory_requirement_mb=memory_requirement,
            recommended_algorithms=recommended_algorithms,
            constraints=constraints
        )
    
    def _determine_task_type(self, y: pd.Series) -> str:
        """Determine if task is classification or regression"""
        if y.dtype in ['object', 'category'] or len(y.unique()) <= 20:
            return 'classification'
        
        # Check if values are integers and limited
        if y.dtype in ['int64', 'int32'] and len(y.unique()) <= 100:
            return 'classification'
        
        return 'regression'
    
    def _calculate_imbalance_ratio(self, y: pd.Series, task_type: str) -> float:
        """Calculate class imbalance ratio"""
        if task_type == 'regression':
            return 1.0
        
        class_counts = y.value_counts()
        if len(class_counts) < 2:
            return 1.0
        
        return class_counts.min() / class_counts.max()
    
    def _prepare_numerical_data(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Prepare numerical data for analysis"""
        X_num = X.copy()
        
        # Encode categorical variables
        label_encoders = {}
        for col in X_num.columns:
            if X_num[col].dtype == 'object':
                le = LabelEncoder()
                X_num[col] = le.fit_transform(X_num[col].astype(str))
                label_encoders[col] = le
        
        # Handle missing values
        for col in X_num.columns:
            if X_num[col].isnull().sum() > 0:
                X_num[col] = X_num[col].fillna(X_num[col].median())
        
        return X_num
    
    def _assess_linearity(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> float:
        """Assess linear separability/relationship in the data"""
        try:
            if len(X) > 5000:  # Sample for efficiency
                X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.8, random_state=42)
            else:
                X_sample, y_sample = X, y
            
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.model_selection import cross_val_score
            
            if task_type == 'classification':
                model = LogisticRegression(random_state=42, max_iter=1000)
                scoring = 'accuracy'
            else:
                model = LinearRegression()
                scoring = 'r2'
            
            cv_scores = cross_val_score(model, X_sample, y_sample, cv=3, scoring=scoring)
            linearity_score = np.mean(cv_scores)
            
            # Normalize to 0-1 scale
            return max(0.0, min(1.0, linearity_score))
            
        except Exception as e:
            logging.warning(f"Linearity assessment failed: {e}")
            return 0.5
    
    def _assess_separability(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> float:
        """Assess how well-separated classes/values are"""
        try:
            if task_type == 'classification' and len(y.unique()) > 1:
                # Use silhouette score as proxy for separability
                if len(X) > 1000:  # Sample for efficiency
                    sample_idx = np.random.choice(len(X), 1000, replace=False)
                    X_sample = X.iloc[sample_idx]
                    y_sample = y.iloc[sample_idx]
                else:
                    X_sample, y_sample = X, y
                
                if len(X_sample.columns) > 2:
                    # Reduce dimensionality for silhouette calculation
                    pca = PCA(n_components=min(10, len(X_sample.columns)))
                    X_reduced = pca.fit_transform(X_sample)
                else:
                    X_reduced = X_sample.values
                
                silhouette_avg = silhouette_score(X_reduced, y_sample)
                # Convert from [-1, 1] to [0, 1]
                return (silhouette_avg + 1) / 2
            
            else:  # Regression
                # For regression, assess data clustering tendency
                if len(X) > 1000:
                    sample_idx = np.random.choice(len(X), 1000, replace=False)
                    X_sample = X.iloc[sample_idx]
                else:
                    X_sample = X
                
                # Use k-means clustering and assess compactness
                kmeans = KMeans(n_clusters=min(5, len(X_sample)//10), random_state=42)
                cluster_labels = kmeans.fit_predict(X_sample)
                
                if len(np.unique(cluster_labels)) > 1:
                    separability = silhouette_score(X_sample, cluster_labels)
                    return (separability + 1) / 2
                else:
                    return 0.5
                    
        except Exception as e:
            logging.warning(f"Separability assessment failed: {e}")
            return 0.5
    
    def _estimate_noise_level(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> float:
        """Estimate noise level in the data"""
        try:
            # Use mutual information to estimate signal-to-noise ratio
            if len(X) > 2000:  # Sample for efficiency
                sample_idx = np.random.choice(len(X), 2000, replace=False)
                X_sample = X.iloc[sample_idx]
                y_sample = y.iloc[sample_idx]
            else:
                X_sample, y_sample = X, y
            
            if task_type == 'classification':
                mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
            
            # Calculate average mutual information
            avg_mi = np.mean(mi_scores) if len(mi_scores) > 0 else 0
            
            # Estimate noise level (inverse of signal strength)
            # Higher MI = lower noise
            noise_level = max(0.0, min(1.0, 1.0 - (avg_mi / (avg_mi + 0.1))))
            
            return noise_level
            
        except Exception as e:
            logging.warning(f"Noise estimation failed: {e}")
            return 0.3  # Default moderate noise level
    
    def _calculate_complexity_score(self, n_samples: int, n_features: int, 
                                  linearity_score: float, separability_score: float, 
                                  noise_level: float) -> float:
        """Calculate overall dataset complexity score"""
        
        # Size complexity
        size_complexity = min(1.0, (n_features * np.log(n_samples)) / 10000)
        
        # Feature-to-sample ratio
        ratio_complexity = min(1.0, n_features / max(n_samples, 1))
        
        # Pattern complexity (lower linearity and separability = higher complexity)
        pattern_complexity = (1.0 - linearity_score) * 0.4 + (1.0 - separability_score) * 0.3
        
        # Noise complexity
        noise_complexity = noise_level * 0.3
        
        # Combined complexity score
        complexity = (size_complexity * 0.2 + 
                     ratio_complexity * 0.3 + 
                     pattern_complexity * 0.3 + 
                     noise_complexity * 0.2)
        
        return min(1.0, complexity)
    
    def _estimate_memory_requirement(self, n_samples: int, n_features: int) -> float:
        """Estimate memory requirement in MB"""
        # Base memory for data storage (8 bytes per float)
        data_memory = (n_samples * n_features * 8) / (1024 * 1024)
        
        # Additional memory for model training (varies by algorithm)
        # Estimate for ensemble methods (higher memory requirement)
        training_memory = data_memory * 3
        
        return data_memory + training_memory
    
    def _recommend_algorithms(self, task_type: str, n_samples: int, n_features: int,
                            n_classes: Optional[int], complexity_score: float,
                            linearity_score: float, separability_score: float,
                            constraints: Dict[str, Any]) -> List[str]:
        """Recommend algorithms based on dataset characteristics"""
        
        recommendations = []
        
        # Time constraint
        max_time_minutes = constraints.get('max_training_time_minutes', 60)
        memory_limit_gb = constraints.get('memory_limit_gb', 8)
        require_interpretability = constraints.get('require_interpretability', False)
        
        # Algorithm selection logic
        if task_type == 'classification':
            # Linear models for high linearity
            if linearity_score > 0.7:
                recommendations.extend(['logistic_regression', 'linear_svm'])
            
            # Tree-based for interpretability
            if require_interpretability:
                recommendations.extend(['decision_tree', 'random_forest'])
            
            # Ensemble methods for complex data
            if complexity_score > 0.5 and n_samples > 1000:
                recommendations.extend(['random_forest', 'gradient_boosting', 'xgboost'])
            
            # Fast algorithms for large datasets or time constraints
            if n_samples > 10000 or max_time_minutes < 10:
                recommendations.extend(['logistic_regression', 'naive_bayes'])
            
            # Multi-class handling
            if n_classes and n_classes > 2:
                recommendations.extend(['random_forest', 'svm', 'neural_network'])
            
        else:  # Regression
            # Linear models for high linearity
            if linearity_score > 0.7:
                recommendations.extend(['linear_regression', 'ridge', 'lasso'])
            
            # Tree-based for non-linear patterns
            if linearity_score < 0.5:
                recommendations.extend(['random_forest_regressor', 'gradient_boosting_regressor'])
            
            # Ensemble for complex data
            if complexity_score > 0.5:
                recommendations.extend(['random_forest_regressor', 'xgboost_regressor'])
        
        # Remove duplicates and prioritize
        recommendations = list(dict.fromkeys(recommendations))
        
        # Default fallbacks
        if not recommendations:
            if task_type == 'classification':
                recommendations = ['random_forest', 'logistic_regression']
            else:
                recommendations = ['random_forest_regressor', 'linear_regression']
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def get_performance_expectations(self, characteristics: DatasetCharacteristics) -> Dict[str, Any]:
        """Estimate expected performance ranges based on dataset characteristics"""
        
        # Base performance expectations
        if characteristics.task_type == 'classification':
            if characteristics.n_classes == 2:
                base_accuracy = 0.5 + (characteristics.separability_score * 0.4)
            else:
                base_accuracy = (1.0 / characteristics.n_classes) + (characteristics.separability_score * 0.3)
            
            expected_performance = {
                'accuracy_range': (max(0.1, base_accuracy - 0.15), min(0.99, base_accuracy + 0.15)),
                'training_time_range': self._estimate_training_time(characteristics),
                'expected_best_algorithm': characteristics.recommended_algorithms[0] if characteristics.recommended_algorithms else 'random_forest'
            }
        else:
            # For regression, use R² expectations
            base_r2 = 0.3 + (characteristics.linearity_score * 0.4) - (characteristics.noise_level * 0.3)
            expected_performance = {
                'r2_range': (max(0.0, base_r2 - 0.2), min(0.95, base_r2 + 0.2)),
                'training_time_range': self._estimate_training_time(characteristics),
                'expected_best_algorithm': characteristics.recommended_algorithms[0] if characteristics.recommended_algorithms else 'random_forest_regressor'
            }
        
        return expected_performance
    
    def _estimate_training_time(self, characteristics: DatasetCharacteristics) -> Tuple[float, float]:
        """Estimate training time range in minutes"""
        n_samples = characteristics.n_samples
        n_features = characteristics.n_features
        complexity = characteristics.complexity_score
        
        # Base time estimate (very rough)
        base_time_seconds = (n_samples * n_features * complexity) / 10000
        
        # Convert to minutes
        base_time_minutes = base_time_seconds / 60
        
        # Range: fast algorithms to complex ensemble methods
        min_time = max(0.1, base_time_minutes * 0.1)
        max_time = base_time_minutes * 10
        
        return (min_time, min(120, max_time))  # Cap at 2 hours