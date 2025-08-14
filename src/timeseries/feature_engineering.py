"""Time-Series Feature Engineering Transformers for DataInsight AI"""

import logging
from typing import List, Union, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateTimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """Extract temporal features from datetime columns."""
    
    def __init__(self, features: List[str] = None, datetime_column: str = None):
        self.features = features or ['year', 'month', 'day', 'dayofweek', 'hour', 'is_weekend']
        self.datetime_column = datetime_column
        self.feature_names_ = []
        self.datetime_col_index_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.datetime_column:
            if self.datetime_column not in X.columns:
                raise ValueError(f"Datetime column '{self.datetime_column}' not found")
            self.datetime_col_index_ = X.columns.tolist().index(self.datetime_column)
        else:
            datetime_cols = X.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) == 0:
                raise ValueError("No datetime columns found")
            self.datetime_col_index_ = X.columns.tolist().index(datetime_cols[0])
            
        self.feature_names_ = [f"dt_{feat}" for feat in self.features]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        datetime_series = pd.to_datetime(X.iloc[:, self.datetime_col_index_])
        result = pd.DataFrame(index=X.index)
        
        for feature in self.features:
            col_name = f"dt_{feature}"
            
            if feature == 'is_weekend':
                result[col_name] = (datetime_series.dt.dayofweek >= 5).astype(int)
            elif feature == 'is_month_start':
                result[col_name] = datetime_series.dt.is_month_start.astype(int)
            elif feature == 'is_month_end':
                result[col_name] = datetime_series.dt.is_month_end.astype(int)
            elif feature == 'quarter':
                result[col_name] = datetime_series.dt.quarter
            elif hasattr(datetime_series.dt, feature):
                value = getattr(datetime_series.dt, feature)
                result[col_name] = value.astype(int) if value.dtype == bool else value
            else:
                raise ValueError(f"Unknown datetime feature: {feature}")
                
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

class LagFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create lagged features for time-series forecasting."""
    
    def __init__(self, lags: List[int], target_column: str = None):
        self.lags = sorted([abs(lag) for lag in lags if lag != 0])
        self.target_column = target_column
        self.feature_names_ = []
        self.target_col_index_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.target_column:
            if self.target_column not in X.columns:
                raise ValueError(f"Target column '{self.target_column}' not found")
            self.target_col_index_ = X.columns.tolist().index(self.target_column)
            base_name = self.target_column
        else:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for lag features")
            self.target_col_index_ = X.columns.tolist().index(numeric_cols[0])
            base_name = numeric_cols[0]
            
        self.feature_names_ = [f"{base_name}_lag_{lag}" for lag in self.lags]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        target_series = X.iloc[:, self.target_col_index_]
        base_name = X.columns[self.target_col_index_]
        result = pd.DataFrame(index=X.index)
        
        for lag in self.lags:
            result[f"{base_name}_lag_{lag}"] = target_series.shift(lag)
            
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

class RollingWindowTransformer(BaseEstimator, TransformerMixin):
    """Create rolling window statistical features."""
    
    def __init__(self, window_size: int, aggregations: List[str] = None, target_column: str = None):
        self.window_size = max(1, int(window_size))
        self.aggregations = aggregations or ['mean', 'std', 'min', 'max']
        self.target_column = target_column
        self.feature_names_ = []
        self.target_col_index_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.target_column:
            if self.target_column not in X.columns:
                raise ValueError(f"Target column '{self.target_column}' not found")
            self.target_col_index_ = X.columns.tolist().index(self.target_column)
            base_name = self.target_column
        else:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for rolling features")
            self.target_col_index_ = X.columns.tolist().index(numeric_cols[0])
            base_name = numeric_cols[0]
            
        self.feature_names_ = [f"{base_name}_roll_{agg}_{self.window_size}" for agg in self.aggregations]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        target_series = X.iloc[:, self.target_col_index_]
        base_name = X.columns[self.target_col_index_]
        result = pd.DataFrame(index=X.index)
        
        rolling = target_series.rolling(window=self.window_size, min_periods=1)
        
        for agg in self.aggregations:
            col_name = f"{base_name}_roll_{agg}_{self.window_size}"
            try:
                result[col_name] = getattr(rolling, agg)()
            except AttributeError:
                result[col_name] = rolling.apply(agg)
                
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

class SeasonalDecompositionTransformer(BaseEstimator, TransformerMixin):
    """Extract seasonal components from time-series data."""
    
    def __init__(self, period: int = None, model: str = 'additive', target_column: str = None):
        self.period = period
        self.model = model
        self.target_column = target_column
        self.feature_names_ = []
        self.target_col_index_ = None

    def fit(self, X: pd.DataFrame, y=None):
        if self.target_column:
            if self.target_column not in X.columns:
                raise ValueError(f"Target column '{self.target_column}' not found")
            self.target_col_index_ = X.columns.tolist().index(self.target_column)
            base_name = self.target_column
        else:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for seasonal decomposition")
            self.target_col_index_ = X.columns.tolist().index(numeric_cols[0])
            base_name = numeric_cols[0]
            
        if self.period is None:
            self.period = min(12, len(X) // 4)  # Default to monthly or quarterly
            
        self.feature_names_ = [f"{base_name}_trend", f"{base_name}_seasonal", f"{base_name}_residual"]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            logging.warning("statsmodels not available, skipping seasonal decomposition")
            return pd.DataFrame(index=X.index)
            
        target_series = X.iloc[:, self.target_col_index_]
        base_name = X.columns[self.target_col_index_]
        result = pd.DataFrame(index=X.index)
        
        if len(target_series.dropna()) < 2 * self.period:
            logging.warning("Insufficient data for seasonal decomposition")
            result[f"{base_name}_trend"] = target_series
            result[f"{base_name}_seasonal"] = 0
            result[f"{base_name}_residual"] = 0
            return result
        
        try:
            decomposition = seasonal_decompose(
                target_series.dropna(), 
                model=self.model, 
                period=self.period,
                extrapolate_trend='freq'
            )
            
            result[f"{base_name}_trend"] = decomposition.trend.reindex(X.index)
            result[f"{base_name}_seasonal"] = decomposition.seasonal.reindex(X.index)
            result[f"{base_name}_residual"] = decomposition.resid.reindex(X.index)
            
        except Exception as e:
            logging.warning(f"Seasonal decomposition failed: {e}")
            result[f"{base_name}_trend"] = target_series
            result[f"{base_name}_seasonal"] = 0
            result[f"{base_name}_residual"] = 0
            
        return result.fillna(method='ffill').fillna(method='bfill').fillna(0)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_