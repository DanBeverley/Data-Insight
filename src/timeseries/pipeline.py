"""
Time-Series Forecasting Pipeline Construction Module for DataInsight AI

This module provides a factory function to construct a complete pipeline for
time-series forecasting tasks. It integrates the feature engineering
transformers with standard preprocessing for exogenous variables.
"""

import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .feature_engineering import (
    DateTimeFeatureTransformer,
    LagFeatureTransformer,
    RollingWindowTransformer
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_timeseries_forecasting_pipeline(
    date_column: str,
    target_column: str,
    exogenous_numeric_cols: List[str],
    exogenous_categorical_cols: List[str],
    ts_config: Dict[str, Any],
    base_numeric_transformer: Pipeline,
    base_categorical_transformer: Pipeline,
    model: Optional[Any] = None
) -> Pipeline:
    """
    Builds a full pipeline for time-series forecasting.

    This function constructs a complex ColumnTransformer that:
    1. Extracts date features from the date column.
    2. Creates lag and rolling features from the target column.
    3. Applies standard preprocessing to any exogenous variables.

    The resulting feature set is then fed into a final estimator model.

    Args:
        date_column: The name of the column containing the date/time information.
        target_column: The name of the target variable to be forecasted.
        exogenous_numeric_cols: A list of numeric feature columns.
        exogenous_categorical_cols: A list of categorical feature columns.
        ts_config: A dictionary with time-series feature parameters from config.yml.
        base_numeric_transformer: The pre-configured pipeline for numeric features.
        base_categorical_transformer: The pre-configured pipeline for categorical features.
        model: An optional scikit-learn estimator. Defaults to RandomForestRegressor.

    Returns:
        A complete scikit-learn Pipeline for time-series forecasting.
    """
    logging.info("Constructing time-series forecasting pipeline...")

    dt_transformer = DateTimeFeatureTransformer(features=ts_config.get('datetime_features', []))
    lag_transformer = LagFeatureTransformer(lags=ts_config.get('lags', []))
    rolling_transformer = RollingWindowTransformer(
        window_size=ts_config.get('rolling_window_size', 1),
        aggregations=ts_config.get('rolling_aggregations', [])
    )

    target_feature_pipeline = Pipeline(steps=[
        ('lags', lag_transformer),
        ('rolling', rolling_transformer)
    ])

    transformer_list = []

    if ts_config.get('datetime_features'):
        transformer_list.append(('datetime_features', dt_transformer, [date_column]))

    if ts_config.get('lags') or ts_config.get('rolling_aggregations'):
        transformer_list.append(('target_features', target_feature_pipeline, [target_column]))

    if exogenous_numeric_cols:
        transformer_list.append(('exogenous_numeric', base_numeric_transformer, exogenous_numeric_cols))
    if exogenous_categorical_cols:
        transformer_list.append(('exogenous_categorical', base_categorical_transformer, exogenous_categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformer_list, remainder='drop')
    
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # The final pipeline
    # NOTE: The output of this pipeline will have NaNs at the beginning due to
    #       lags and rolling windows. The calling function (e.g., in the
    #       orchestrator or app) is responsible for dropping these NaNs from
    #       both X and y before fitting.
    final_pipeline = Pipeline(steps=[
        ('feature_engineering_preprocessor', preprocessor),
        ('model', model)
    ])

    logging.info("Time-series forecasting pipeline constructed successfully.")
    return final_pipeline