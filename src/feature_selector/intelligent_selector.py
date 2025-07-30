"""
Intelligent Feature Selection Module for DataInsight AI

This module provides the `IntelligentFeatureSelector` class, which is responsible
for reducing the dimensionality of a feature set by selecting the most relevant
and informative features for a given machine learning task.
"""
import logging
from typing import Dict, Any, List, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectFromModel,
    mutual_info_classif,
    mutual_info_regression,
    RFE
)
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IntelligentFeatureSelector:
    """

    A class to perform multi-strategy feature selection.
    
    It combines several techniques to prune a feature set effectively:
    1. Low Variance Removal: Drops features with little to no variation.
    2. High Correlation Removal: Drops one of a pair of highly correlated features.
    3. Model-Based Selection: Uses a proxy model (e.g., RandomForest) to select
       features based on their importance.
    """
    def __init__(self, task:str, config:Optional[Dict[str, Any]]=None):
        if task not in ["classification", "regression"]:
            raise ValueError("Task must be 'classification' or 'regression'")
        self.task = task
        self.config = config or {"variance_threshold":0.01,
                                 "correlation_threshold":0.95,
                                 "max_features_model_based":50}
        self.selected_features_:List[Dict] = []
        self.feature_importances_:Optional[pd.DataFrame] = None

    def select_features(self, X:pd.DataFrame, y:pd.Series) -> pd.DataFrame:
        """
        Executes the full feature selection pipeline.

        Args:
            X: The input DataFrame of features. Must be fully numeric and imputed.
            y: The target Series.

        Returns:
            A DataFrame containing only the selected features.
        """
        logging.info(f"Starting intelligent feature selection for '{self.task}' task.")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise TypeError("All columns in X must be numeric. Please encode categorical features first.")
        if X.isnull().sum().sum() > 0:
            raise ValueError("X contains missing values. Please impute before feature selection.")
        
        X_variance = self._remove_low_variance(X)
        if X_variance.empty:
            logging.warning("No features remained after low variance removal. Returning empty DataFrame")
            self.selected_features_ = []
            return X_variance
        
        X_corr = self._remove_high_correlation(X_variance)
        if X_corr.empty:
            logging.warning("No features remained after high correlation removal. Returning empty DataFrame")
            self.selected_features_ = []
            return X_corr
        
        # Model-based selection (Final Pruning)
        X_final, importances = self._select_from_model(X_corr, y)
        self.selected_features_ = X_final.column.tolist()
        self.feature_importances_ = importances
        logging.info(f"Feature selection complete. Selected {len(self.select_features_)} features. ")
        return X_final
    
    def _remove_low_variance(self, X:pd.DataFrame) -> pd.DataFrame:
        """Removes features with variance below a threshold"""
        threshold = self.config.get("variance_threshold", 0.01)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        selected_cols = X.columns[selector.get_support()]
        dropped_cols = [col for col in X.columns if col not in selected_cols]
        
        logging.info(f"Variance Threshold ({threshold}): Dropped {len(dropped_cols)} features. Kept {len(selected_cols)}.")
        if dropped_cols:
            logging.debug(f"Dropped by variance: {dropped_cols}")
            
        return X[selected_cols]

    def _remove_high_correlation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Removes one of each pair of features with correlation above a threshold."""
        threshold = self.config.get("correlation_threshold", 0.95)
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        logging.info(f"Correlation Threshold ({threshold}): Dropped {len(to_drop)} features.")
        if to_drop:
            logging.debug(f"Dropped by correlation: {to_drop}")
            
        return X.drop(columns=to_drop)

    def _select_from_model(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Uses a RandomForest model to rank and select the top features."""
        max_features = self.config.get("max_features_model_based", 50)
        
        if self.task == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else: # regression
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
        model.fit(X, y)
        
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select the top `max_features`
        top_features = importances['feature'].head(max_features).tolist()
        
        logging.info(f"Model-based selection: Kept top {len(top_features)} features.")
        return X[top_features], importances.head(max_features)