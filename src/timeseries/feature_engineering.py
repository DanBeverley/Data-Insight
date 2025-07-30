"""
Time-Series Feature Engineering Transformers for DataInsight AI

This module provides a collection of custom scikit-learn transformers specifically
designed for creating features from time-series data. These transformers can be
seamlessly integrated into scikit-learn pipelines.

Key Components:
- DateTimeFeatureTransformer: Extracts features from datetime columns (e.g.,
  day of week, month, year, is_weekend).
- LagFeatureTransformer: Creates lagged versions of a time-series variable.
- RollingWindowTransformer: Creates rolling window statistics (e.g., moving
  average, moving standard deviation).
"""
import logging
from typing import List, Union, Dict, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DateTimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """Extract features from a datetime column"""
    def __init__(self, features:List[str]):
        if not features:
            raise ValueError("The 'features' list cannot be empty.")
        self.features = features
        self.feature_names_ = []
    
    def fit(self, X:pd.DataFrame, y=None):
        self.feature_names_ = [f"date_{feat}" for feat in self.features]
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        datetime_series = pd.to_datetime(X_copy.iloc[:,0])
        for feature in self.features:
            new_col_name = f"date_{feature}"
            try:
                X_copy[new_col_name] = getattr(datetime_series.dt, feature)
            except AttributeError:
                # Handle boolean attributes
                if feature.startswith("is_"):
                    X_copy[new_col_name] = getattr(datetime_series.dt, feature)
                else:
                    raise AttributeError(f"'{feature}' is not a valid pandas dt accessor")
        # Convert boolean flags to integers
        for col in X_copy.columns:
            if X_copy[col].dtype == "bool":
                X_copy[col] = X_copy[col].astype(int)

        return X_copy.drop(columns = X.columns)
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_
    
class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Creates lagged features for a time-series.
    """
    def __init__(self, lags: List[int]):
        if not all(isinstance(lag, int) and lag > 0 for lag in lags):
            raise ValueError("All lag values must be positive integers.")
        self.lags = lags
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        base_name = X.columns[0]
        self.feature_names_ = [f"{base_name}_lag_{lag}" for lag in self.lags]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        base_name = X.columns[0]
        for lag in self.lags:
            X_copy[f"{base_name}_lag_{lag}"] = X_copy[base_name].shift(lag)
        return X_copy.drop(columns=[base_name]) # Return only the new features

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_


class RollingWindowTransformer(BaseEstimator, TransformerMixin):
    """
    Creates rolling window aggregate features for a time-series.
    """
    def __init__(self, window_size: int, aggregations: List[str]):
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        self.window_size = window_size
        self.aggregations = aggregations
        self.feature_names_ = []

    def fit(self, X: pd.DataFrame, y=None):
        base_name = X.columns[0]
        self.feature_names_ = [f"{base_name}_rolling_{agg}_{self.window_size}" for agg in self.aggregations]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_copy = X.copy()
        base_name = X.columns[0]
        rolling_series = X_copy[base_name].rolling(window=self.window_size)

        for agg in self.aggregations:
            new_col_name = f"{base_name}_rolling_{agg}_{self.window_size}"
            X_copy[new_col_name] = rolling_series.agg(agg)

        return X_copy.drop(columns=[base_name])

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_