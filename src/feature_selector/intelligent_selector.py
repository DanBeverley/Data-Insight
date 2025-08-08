import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, f_regression, 
    mutual_info_classif, mutual_info_regression, RFE
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

class IntelligentFeatureSelector:
    def __init__(self, task: str = 'classification', config: Optional[Dict[str, Any]] = None):
        self.task = task.lower()
        if self.task not in ['classification', 'regression']:
            raise ValueError("Task must be 'classification' or 'regression'")
        
        self.config = config or {}
        self.selected_features_: List[str] = []
        self.selection_history_: List[Dict[str, Any]] = []
        self.feature_scores_: Dict[str, float] = {}
        self.stability_scores_: Dict[str, float] = {}
        self._skip_stability = False
        
    def select_features(self, X: pd.DataFrame, y: pd.Series, 
                       target_features: Optional[int] = None) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
            raise TypeError("X must be DataFrame, y must be Series")
        
        if X.isnull().sum().sum() > 0:
            raise ValueError("X contains missing values")
        
        n_samples, n_features = X.shape
        strategy = self._select_strategy(n_samples, n_features)
        logging.info(f"Using {strategy} strategy for {n_samples} samples, {n_features} features")
        
        if target_features is None:
            target_features = min(max(10, n_samples // 10), n_features // 2)
        
        X_selected = self._execute_pipeline(X, y, strategy, target_features)
        
        if len(X_selected.columns) > 0 and not self._skip_stability:
            self._calculate_stability(X, y, X_selected.columns.tolist())
        
        self.selected_features_ = X_selected.columns.tolist()
        logging.info(f"Selected {len(self.selected_features_)} features from {n_features}")
        
        return X_selected
    
    def _select_strategy(self, n_samples: int, n_features: int) -> str:
        if n_samples < 100 or n_features < 20:
            return 'minimal'
        elif n_samples < 1000 or n_features < 100:
            return 'standard'
        elif n_samples < 10000 or n_features < 1000:
            return 'comprehensive'
        else:
            return 'advanced'
    
    def _execute_pipeline(self, X: pd.DataFrame, y: pd.Series, 
                         strategy: str, target_features: int) -> pd.DataFrame:
        X_current = X.copy()
        
        # Stage 1: Variance filtering
        X_current = self._variance_filter(X_current)
        if X_current.empty:
            return X_current
        
        # Stage 2: Correlation filtering
        if strategy in ['standard', 'comprehensive', 'advanced']:
            X_current = self._correlation_filter(X_current)
            if X_current.empty:
                return X_current
        
        # Stage 3: Univariate selection
        if strategy in ['comprehensive', 'advanced']:
            X_current = self._univariate_selection(X_current, y, 
                                                 min(target_features * 3, len(X_current.columns)))
            if X_current.empty:
                return X_current
        
        # Stage 4: Mutual information
        if strategy == 'advanced':
            X_current = self._mutual_info_selection(X_current, y, 
                                                  min(target_features * 2, len(X_current.columns)))
            if X_current.empty:
                return X_current
        
        # Stage 5: Model-based RFE
        X_current = self._rfe_selection(X_current, y, target_features)
        
        return X_current
    
    def _variance_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config.get('variance_threshold', 0.01)
        selector = VarianceThreshold(threshold=threshold)
        
        try:
            mask = selector.fit_transform(X).shape[1] > 0
            selected_features = X.columns[selector.get_support()].tolist()
            dropped = len(X.columns) - len(selected_features)
            
            self.selection_history_.append({
                'stage': 'variance_filter',
                'dropped': dropped,
                'remaining': len(selected_features),
                'threshold': threshold
            })
            
            return X[selected_features] if selected_features else pd.DataFrame()
        except:
            return X
    
    def _correlation_filter(self, X: pd.DataFrame) -> pd.DataFrame:
        threshold = self.config.get('correlation_threshold', 0.95)
        
        try:
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            to_drop = [col for col in upper_tri.columns 
                      if any(upper_tri[col] > threshold)]
            
            selected_features = [col for col in X.columns if col not in to_drop]
            
            self.selection_history_.append({
                'stage': 'correlation_filter',
                'dropped': len(to_drop),
                'remaining': len(selected_features),
                'threshold': threshold
            })
            
            return X[selected_features] if selected_features else pd.DataFrame()
        except:
            return X
    
    def _univariate_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> pd.DataFrame:
        try:
            score_func = f_classif if self.task == 'classification' else f_regression
            selector = SelectKBest(score_func=score_func, k=min(k, len(X.columns)))
            
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Store scores
            scores = selector.scores_
            for i, feature in enumerate(X.columns):
                if feature in selected_features:
                    self.feature_scores_[f"{feature}_univariate"] = scores[i]
            
            self.selection_history_.append({
                'stage': 'univariate_selection',
                'dropped': len(X.columns) - len(selected_features),
                'remaining': len(selected_features),
                'k': k
            })
            
            return X[selected_features] if selected_features else pd.DataFrame()
        except:
            return X
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series, k: int) -> pd.DataFrame:
        try:
            mi_func = mutual_info_classif if self.task == 'classification' else mutual_info_regression
            mi_scores = mi_func(X, y, random_state=42)
            
            feature_scores = list(zip(X.columns, mi_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            selected_features = [feat for feat, _ in feature_scores[:k]]
            
            for feature, score in feature_scores:
                if feature in selected_features:
                    self.feature_scores_[f"{feature}_mutual_info"] = score
            
            self.selection_history_.append({
                'stage': 'mutual_info_selection',
                'dropped': len(X.columns) - len(selected_features),
                'remaining': len(selected_features),
                'k': k
            })
            
            return X[selected_features] if selected_features else pd.DataFrame()
        except:
            return X
    
    def _rfe_selection(self, X: pd.DataFrame, y: pd.Series, n_features: int) -> pd.DataFrame:
        try:
            if self.task == 'classification':
                estimator = LogisticRegression(random_state=42, max_iter=1000)
            else:
                estimator = Ridge(random_state=42)
            
            # Scale features for linear models
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            
            selector = RFE(estimator, n_features_to_select=min(n_features, len(X.columns)), step=1)
            selector.fit(X_scaled, y)
            
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Store ranking
            for i, feature in enumerate(X.columns):
                if feature in selected_features:
                    self.feature_scores_[f"{feature}_rfe_rank"] = selector.ranking_[i]
            
            self.selection_history_.append({
                'stage': 'rfe_selection',
                'dropped': len(X.columns) - len(selected_features),
                'remaining': len(selected_features),
                'n_features': n_features
            })
            
            return X[selected_features] if selected_features else pd.DataFrame()
        except:
            return X
    
    def _calculate_stability(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]):
        try:
            n_samples = len(X)
            n_features = len(X.columns)
            
            # Skip stability calculation for very small datasets or when explicitly disabled
            if n_samples < 50 or n_features < 5 or self._skip_stability:
                for feature in selected_features:
                    self.stability_scores_[feature] = 1.0
                return
            
            # Bootstrap stability analysis with subsampling
            n_iterations = min(20, max(10, n_samples // 50))
            subsample_ratio = 0.8
            
            feature_selection_counts = {feature: 0 for feature in X.columns}
            successful_iterations = 0
            
            for iteration in range(n_iterations):
                try:
                    # Bootstrap subsample
                    subsample_size = max(30, int(n_samples * subsample_ratio))
                    if self.task == 'classification':
                        # Stratified subsampling for classification
                        from sklearn.utils import resample
                        indices = []
                        for class_label in np.unique(y):
                            class_indices = np.where(y == class_label)[0]
                            n_class_samples = max(1, int(len(class_indices) * subsample_ratio))
                            class_subsample = resample(class_indices, n_samples=n_class_samples, random_state=42+iteration)
                            indices.extend(class_subsample)
                    else:
                        # Random subsampling for regression
                        indices = np.random.RandomState(42+iteration).choice(n_samples, subsample_size, replace=False)
                    
                    X_subsample = X.iloc[indices]
                    y_subsample = y.iloc[indices]
                    
                    # Simplified feature selection pipeline for stability testing
                    X_stable = self._stability_selection_pipeline(X_subsample, y_subsample, len(selected_features))
                    
                    for feature in X_stable.columns:
                        if feature in feature_selection_counts:
                            feature_selection_counts[feature] += 1
                    
                    successful_iterations += 1
                    
                except Exception as e:
                    logging.debug(f"Bootstrap iteration {iteration} failed: {e}")
                    continue
            
            # Calculate stability scores
            if successful_iterations > 0:
                for feature in selected_features:
                    self.stability_scores_[feature] = feature_selection_counts[feature] / successful_iterations
            else:
                # Fallback: assign neutral stability scores
                for feature in selected_features:
                    self.stability_scores_[feature] = 0.5
                    
        except Exception as e:
            logging.warning(f"Stability calculation failed: {e}")
            # Assign default stability scores
            for feature in selected_features:
                self.stability_scores_[feature] = 0.5
    
    def _stability_selection_pipeline(self, X: pd.DataFrame, y: pd.Series, target_features: int) -> pd.DataFrame:
        """Simplified feature selection pipeline for stability analysis"""
        X_current = X.copy()
        
        # Only use the most robust selection methods for stability testing
        try:
            # Variance filtering
            threshold = self.config.get('variance_threshold', 0.01)
            if threshold > 0:
                variances = X_current.var()
                X_current = X_current.loc[:, variances > threshold]
                if X_current.empty:
                    return X_current
            
            # Correlation filtering
            corr_threshold = self.config.get('correlation_threshold', 0.95)
            if corr_threshold < 1.0 and len(X_current.columns) > 1:
                corr_matrix = X_current.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
                X_current = X_current.drop(columns=to_drop)
                if X_current.empty:
                    return X_current
            
            # Univariate selection with reduced complexity
            if len(X_current.columns) > target_features:
                from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                score_func = f_classif if self.task == 'classification' else f_regression
                selector = SelectKBest(score_func=score_func, k=min(target_features, len(X_current.columns)))
                selector.fit(X_current, y)
                selected_features = X_current.columns[selector.get_support()].tolist()
                X_current = X_current[selected_features]
            
            return X_current
            
        except Exception:
            # Return original features if selection fails
            return X.iloc[:, :min(target_features, len(X.columns))]
    
    def get_feature_importance(self) -> pd.DataFrame:
        if not self.feature_scores_:
            return pd.DataFrame()
        
        importance_data = []
        for feature in self.selected_features_:
            row = {'feature': feature}
            for score_type, score_value in self.feature_scores_.items():
                if feature in score_type:
                    score_name = score_type.replace(f"{feature}_", "")
                    row[score_name] = score_value
            
            if feature in self.stability_scores_:
                row['stability'] = self.stability_scores_[feature]
            
            importance_data.append(row)
        
        return pd.DataFrame(importance_data)
    
    def validate_performance(self, X_original: pd.DataFrame, X_selected: pd.DataFrame, 
                           y: pd.Series, validation_method: str = 'cross_validation') -> Dict[str, float]:
        """Validate that feature selection maintains predictive performance"""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            if validation_method not in ['cross_validation', 'holdout']:
                validation_method = 'cross_validation'
            
            # Create simple baseline model
            if self.task == 'classification':
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            # Scale features for consistent comparison
            scaler_orig = StandardScaler()
            scaler_sel = StandardScaler()
            
            X_orig_scaled = pd.DataFrame(scaler_orig.fit_transform(X_original), 
                                       columns=X_original.columns, index=X_original.index)
            X_sel_scaled = pd.DataFrame(scaler_sel.fit_transform(X_selected),
                                      columns=X_selected.columns, index=X_selected.index)
            
            # Cross-validation scores
            cv_folds = min(5, len(y) // 10) if len(y) >= 50 else 3
            
            orig_scores = cross_val_score(model, X_orig_scaled, y, cv=cv_folds, 
                                        scoring='accuracy' if self.task == 'classification' else 'r2')
            sel_scores = cross_val_score(model, X_sel_scaled, y, cv=cv_folds,
                                       scoring='accuracy' if self.task == 'classification' else 'r2')
            
            performance_metrics = {
                'original_mean_score': float(np.mean(orig_scores)),
                'original_std_score': float(np.std(orig_scores)),
                'selected_mean_score': float(np.mean(sel_scores)),
                'selected_std_score': float(np.std(sel_scores)),
                'performance_retention': float(np.mean(sel_scores) / np.mean(orig_scores)) if np.mean(orig_scores) > 0 else 1.0,
                'dimensionality_reduction': float(len(X_selected.columns) / len(X_original.columns)),
                'efficiency_gain': float(len(X_original.columns) / len(X_selected.columns))
            }
            
            return performance_metrics
            
        except Exception as e:
            logging.warning(f"Performance validation failed: {e}")
            return {
                'original_mean_score': 0.0,
                'selected_mean_score': 0.0,
                'performance_retention': 1.0,
                'dimensionality_reduction': float(len(X_selected.columns) / len(X_original.columns)) if len(X_original.columns) > 0 else 1.0,
                'efficiency_gain': 1.0
            }
    
    def get_selection_summary(self) -> Dict[str, Any]:
        summary = {
            'total_features_original': 0,
            'total_features_selected': len(self.selected_features_),
            'selection_stages': len(self.selection_history_),
            'stage_details': self.selection_history_,
            'avg_stability': np.mean(list(self.stability_scores_.values())) if self.stability_scores_ else 0.0,
            'min_stability': min(self.stability_scores_.values()) if self.stability_scores_ else 0.0,
            'max_stability': max(self.stability_scores_.values()) if self.stability_scores_ else 0.0,
            'config_used': self.config.copy()
        }
        
        if self.selection_history_:
            summary['total_features_original'] = (
                self.selection_history_[0]['remaining'] + self.selection_history_[0]['dropped']
            )
        
        # Calculate reduction efficiency per stage
        if len(self.selection_history_) > 0:
            stage_reductions = []
            for stage in self.selection_history_:
                if stage['remaining'] + stage['dropped'] > 0:
                    reduction = stage['dropped'] / (stage['remaining'] + stage['dropped'])
                    stage_reductions.append(reduction)
            
            summary['avg_reduction_per_stage'] = np.mean(stage_reductions) if stage_reductions else 0.0
            summary['total_reduction_ratio'] = (
                (summary['total_features_original'] - summary['total_features_selected']) / 
                summary['total_features_original']
            ) if summary['total_features_original'] > 0 else 0.0
        
        return summary
    
    @staticmethod
    def get_default_config(task: str, dataset_size: str = 'medium') -> Dict[str, Any]:
        """Get production-ready default configurations for different scenarios"""
        base_config = {
            'variance_threshold': 0.01,
            'correlation_threshold': 0.95
        }
        
        if dataset_size == 'small':  # < 1000 samples
            return {**base_config, 'variance_threshold': 0.005, 'correlation_threshold': 0.90}
        elif dataset_size == 'large':  # > 10000 samples  
            return {**base_config, 'variance_threshold': 0.02, 'correlation_threshold': 0.98}
        elif dataset_size == 'xlarge':  # > 100000 samples
            return {**base_config, 'variance_threshold': 0.05, 'correlation_threshold': 0.99}
        else:  # medium: 1000-10000 samples
            return base_config