"""Time-Series Pipeline Construction for DataInsight AI"""

import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .feature_engineering import (
    DateTimeFeatureTransformer,
    LagFeatureTransformer,
    RollingWindowTransformer,
    SeasonalDecompositionTransformer
)

def create_timeseries_pipeline(
    df: pd.DataFrame,
    target_column: str,
    datetime_column: str = None,
    config: Dict[str, Any] = None
) -> Pipeline:
    """Build comprehensive time-series forecasting pipeline."""
    
    config = config or {}
    
    # Auto-detect datetime column if not specified
    if not datetime_column:
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            datetime_column = datetime_cols[0]
    
    transformers = []
    
    # Datetime features
    if datetime_column and datetime_column in df.columns:
        dt_features = config.get('datetime_features', ['year', 'month', 'dayofweek', 'quarter', 'is_weekend'])
        dt_transformer = DateTimeFeatureTransformer(
            features=dt_features,
            datetime_column=datetime_column
        )
        transformers.append(('datetime', dt_transformer, [datetime_column]))
    
    # Lag features
    if config.get('lags', []):
        lag_transformer = LagFeatureTransformer(
            lags=config['lags'],
            target_column=target_column
        )
        transformers.append(('lags', lag_transformer, [target_column]))
    
    # Rolling window features
    if config.get('rolling_windows', []):
        for window_config in config['rolling_windows']:
            window_size = window_config.get('size', 7)
            aggregations = window_config.get('aggregations', ['mean', 'std'])
            
            rolling_transformer = RollingWindowTransformer(
                window_size=window_size,
                aggregations=aggregations,
                target_column=target_column
            )
            transformers.append(
                (f'rolling_{window_size}', rolling_transformer, [target_column])
            )
    
    # Seasonal decomposition
    if config.get('seasonal_decomposition', False):
        seasonal_transformer = SeasonalDecompositionTransformer(
            period=config.get('seasonal_period'),
            model=config.get('seasonal_model', 'additive'),
            target_column=target_column
        )
        transformers.append(('seasonal', seasonal_transformer, [target_column]))
    
    # Exogenous features (other numeric columns)
    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                   if col != target_column]
    
    if numeric_cols:
        exog_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('exogenous', exog_transformer, numeric_cols))
    
    if not transformers:
        raise ValueError("No valid transformers could be created for time-series pipeline")
    
    # Create the preprocessor
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Create the full pipeline
    model = config.get('model') or RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

def detect_datetime_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze DataFrame to suggest time-series configuration."""
    
    suggestions = {
        'datetime_columns': [],
        'potential_targets': [],
        'suggested_lags': [],
        'suggested_windows': [],
        'seasonal_period': None
    }
    
    # Find datetime columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Try to parse object columns as datetime
    for col in df.select_dtypes(include=['object']).columns:
        try:
            pd.to_datetime(df[col].head(100))
            datetime_cols.append(col)
        except:
            continue
    
    suggestions['datetime_columns'] = datetime_cols
    
    # Find numeric columns as potential targets
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    suggestions['potential_targets'] = numeric_cols
    
    # Suggest lags based on data length
    data_length = len(df)
    if data_length > 50:
        suggestions['suggested_lags'] = [1, 2, 3, 7, 14]
    elif data_length > 20:
        suggestions['suggested_lags'] = [1, 2, 3]
    else:
        suggestions['suggested_lags'] = [1]
    
    # Suggest rolling windows
    if data_length > 30:
        suggestions['suggested_windows'] = [
            {'size': 7, 'aggregations': ['mean', 'std']},
            {'size': 14, 'aggregations': ['mean', 'min', 'max']}
        ]
    elif data_length > 10:
        suggestions['suggested_windows'] = [
            {'size': 3, 'aggregations': ['mean']}
        ]
    
    # Suggest seasonal period
    if data_length > 24:
        suggestions['seasonal_period'] = 12  # Monthly
    elif data_length > 14:
        suggestions['seasonal_period'] = 7   # Weekly
    
    return suggestions

def create_timeseries_config(
    df: pd.DataFrame,
    target_column: str,
    datetime_column: str = None,
    auto_detect: bool = True
) -> Dict[str, Any]:
    """Generate optimal time-series configuration for given data."""
    
    if auto_detect:
        patterns = detect_datetime_patterns(df)
        
        # Use detected datetime column if not specified
        if not datetime_column and patterns['datetime_columns']:
            datetime_column = patterns['datetime_columns'][0]
        
        config = {
            'datetime_features': ['year', 'month', 'dayofweek', 'quarter', 'is_weekend'],
            'lags': patterns['suggested_lags'][:5],  # Limit to first 5
            'rolling_windows': patterns['suggested_windows'],
            'seasonal_decomposition': len(df) > 24,
            'seasonal_period': patterns['seasonal_period'],
            'seasonal_model': 'additive'
        }
    else:
        # Minimal configuration
        config = {
            'datetime_features': ['month', 'dayofweek'],
            'lags': [1, 2, 3],
            'rolling_windows': [{'size': 7, 'aggregations': ['mean']}],
            'seasonal_decomposition': False
        }
    
    return config