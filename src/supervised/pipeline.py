"""
Supervised Learning Pipeline Construction Module for DataInsight AI

This module provides specialized functions to build, augment, and analyze
pipelines for supervised learning tasks (classification and regression). It
encapsulates logic for handling common challenges like class imbalance and
provides utilities for model interpretation.

This module is designed to be called by the `WorkflowOrchestrator` after the
base preprocessor has been constructed.
"""

import logging
from typing import Optional, Union, Dict, Any, List

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from ..config import settings

RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_default_model(task: str) -> BaseEstimator:
    """
    Selects a robust default model based on the supervised learning task.

    Args:
        task: The supervised learning task, either 'classification' or 'regression'.

    Returns:
        An unfitted scikit-learn estimator instance.
    
    Raises:
        ValueError: If the task is not recognized.
    """
    if task == 'classification':
        logging.info("Selecting RandomForestClassifier as default model.")
        return RandomForestClassifier(random_state=RANDOM_STATE)
    elif task == 'regression':
        logging.info("Selecting RandomForestRegressor as default model.")
        return RandomForestRegressor(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown supervised learning task: '{task}'")


def create_supervised_pipeline(
    preprocessor: ColumnTransformer,
    task: str,
    target_series: Optional[pd.Series] = None,
    model: Optional[BaseEstimator] = None
) -> Union[Pipeline, ImbPipeline]:
    """
    Constructs a full supervised learning pipeline.

    This function takes a pre-configured preprocessor, determines the task,
    checks for class imbalance (for classification), and attaches a final
    estimator (model) to create a complete, model-ready pipeline.

    Args:
        preprocessor: The fitted or unfitted ColumnTransformer for feature processing.
        task: The supervised learning task: 'classification' or 'regression'.
        target_series: The target variable series (y). Required for imbalance check.
        model: An optional scikit-learn estimator. If None, a default will be used.

    Returns:
        A scikit-learn or imblearn Pipeline object, ready for fitting.
    """
    if model is None:
        model = _get_default_model(task)

    pipeline_steps = [
        ('preprocessor', preprocessor),
        # HOOK FOR PHASE 2: Feature Selection can be inserted here
    ]

    if task == 'classification':
        if target_series is None:
            raise ValueError("`target_series` is required for classification task to check for imbalance.")
        
        # Check for class imbalance
        imbalance_threshold = settings['pipeline_params']['imbalance_smote_threshold']
        minority_ratio = target_series.value_counts(normalize=True).min()
        
        if minority_ratio < imbalance_threshold:
            logging.info(f"Class imbalance detected (minority class ratio: {minority_ratio:.2f}). Adding SMOTE to pipeline.")
            pipeline_steps.append(('resampler', SMOTE(random_state=RANDOM_STATE)))
            pipeline_steps.append(('model', model))
            return ImbPipeline(pipeline_steps)
        else:
            logging.info("Target variable is balanced. Using standard pipeline.")

    pipeline_steps.append(('model', model))
    return ImbPipeline(pipeline_steps)


def calculate_permutation_importance(
    fitted_pipeline: Union[Pipeline, ImbPipeline],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Calculates and returns feature importances using permutation on a hold-out set.

    This method is model-agnostic and provides a reliable estimate of feature
    importance by measuring the performance drop when a feature's values are
    randomly shuffled.

    Args:
        fitted_pipeline: The already-fitted supervised learning pipeline.
        X_val: The validation features DataFrame (raw, not preprocessed).
        y_val: The validation target Series.
        feature_names: The list of column names *after* preprocessing.

    Returns:
        A pandas DataFrame with features, their mean importance, and standard deviation,
        sorted from most to least important.
    """
    logging.info("Calculating permutation importance on validation set...")
    try:
        result = permutation_importance(
            fitted_pipeline,
            X_val,
            y_val,
            n_repeats=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
        }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

        logging.info("Permutation importance calculation complete.")
        return importance_df

    except Exception as e:
        logging.error(f"Failed to calculate permutation importance: {e}")
        return pd.DataFrame(columns=['feature', 'importance_mean', 'importance_std'])


# For testing purposes
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import FunctionTransformer

    # 1. --- Setup a dummy environment ---
    # Dummy preprocessor (just passes through data)
    dummy_preprocessor = ColumnTransformer(
        [('passthrough', FunctionTransformer(), ['feature1', 'feature2'])],
        remainder='drop'
    )

    # 2. --- Test Case 1: Imbalanced Classification ---
    print("\n--- Testing Imbalanced Classification ---")
    X_imbalanced = pd.DataFrame({
        'feature1': range(100),
        'feature2': range(100, 200)
    })
    y_imbalanced = pd.Series([0] * 95 + [1] * 5) # 5% minority class

    imbalanced_pipeline = create_supervised_pipeline(
        preprocessor=dummy_preprocessor,
        task='classification',
        target_series=y_imbalanced
    )
    print("Pipeline Type:", type(imbalanced_pipeline).__name__)
    print("Pipeline Steps:", [name for name, _ in imbalanced_pipeline.steps])
    assert 'resampler' in imbalanced_pipeline.named_steps
    assert isinstance(imbalanced_pipeline.named_steps['resampler'], SMOTE)

    # 3. --- Test Case 2: Balanced Classification ---
    print("\n--- Testing Balanced Classification ---")
    y_balanced = pd.Series([0] * 50 + [1] * 50)

    balanced_pipeline = create_supervised_pipeline(
        preprocessor=dummy_preprocessor,
        task='classification',
        target_series=y_balanced
    )
    print("Pipeline Type:", type(balanced_pipeline).__name__)
    print("Pipeline Steps:", [name for name, _ in balanced_pipeline.steps])
    assert 'resampler' not in balanced_pipeline.named_steps

    # 4. --- Test Case 3: Regression ---
    print("\n--- Testing Regression ---")
    regression_pipeline = create_supervised_pipeline(
        preprocessor=dummy_preprocessor,
        task='regression'
    )
    print("Pipeline Type:", type(regression_pipeline).__name__)
    print("Pipeline Steps:", [name for name, _ in regression_pipeline.steps])
    assert isinstance(regression_pipeline.named_steps['model'], RandomForestRegressor)

    # 5. --- Test Permutation Importance ---
    print("\n--- Testing Permutation Importance ---")
    # Fit a pipeline
    X_train, X_val, y_train, y_val = train_test_split(X_imbalanced, y_imbalanced, test_size=0.3, random_state=RANDOM_STATE)
    imbalanced_pipeline.fit(X_train, y_train)

    # The feature names after the dummy preprocessor are the same
    feature_names_post_processing = ['feature1', 'feature2']

    importance_df = calculate_permutation_importance(
        fitted_pipeline=imbalanced_pipeline,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names_post_processing
    )
    print("Permutation Importance Results:")
    print(importance_df)