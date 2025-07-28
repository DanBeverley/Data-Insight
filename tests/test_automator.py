"""
Unit Tests for the WorkflowOrchestrator

This test suite validates the core logic of the `WorkflowOrchestrator`, ensuring
it correctly profiles data, classifies columns, and constructs appropriate
pipelines for different tasks and data characteristics.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from automator import WorkflowOrchestrator


# Test Fixtures: Reusable Data for Tests
@pytest.fixture
def messy_dataframe() -> pd.DataFrame:
    """Creates a complex, messy DataFrame for comprehensive testing."""
    data = {
        'transaction_id': [f'id_{i}' for i in range(100)], # ID column
        'user_age': [25, 30, 22, 45, 30, np.nan, 60, 22, 34, 55] * 10, # Numeric with NaNs
        'city': ['NY', 'SFO', 'SFO', 'NY', 'LA', 'chicago', 'la', 'New York', 'sf', 'Chicago'] * 10, # Messy Categorical
        'revenue': [10, 15, 8, 1500, 20, np.nan, 25, 2000, 12, 18] * 10, # Skewed numeric
        'product_sku': [f'sku_{i}' for i in range(10, 110)], # High-cardinality categorical
        'has_churned': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1] * 10, # Target variable (imbalanced)
        'notes': [None] * 100 # Column to be dropped (all NaNs)
    }
    return pd.DataFrame(data)

@pytest.fixture
def balanced_dataframe() -> pd.DataFrame:
    """Creates a simple, balanced DataFrame for testing classification without SMOTE."""
    data = {
        'feature1': np.random.rand(100),
        'feature2': ['A', 'B'] * 50,
        'target': [0, 1] * 50
    }
    return pd.DataFrame(data)


# Test Cases for WorkflowOrchestrator 

def test_initialization(messy_dataframe):
    """Test that the orchestrator initializes correctly and profiles data."""
    orchestrator = WorkflowOrchestrator(df=messy_dataframe, target_column='has_churned', task='classification')
    assert isinstance(orchestrator.df, pd.DataFrame)
    assert orchestrator.target_column == 'has_churned'
    assert 'user_age' in orchestrator.schema
    assert orchestrator.schema['notes']['missing_percent'] == 1.0
    assert orchestrator.schema['transaction_id']['cardinality'] == 1.0
    assert orchestrator.schema['revenue']['skewness'] > 2.0

def test_column_classification(messy_dataframe):
    """Test the core logic of classifying columns into their correct roles."""
    orchestrator = WorkflowOrchestrator(df=messy_dataframe, target_column='has_churned', task='classification')
    
    roles = orchestrator.column_roles
    
    assert 'notes' in roles['to_drop']
    assert 'transaction_id' in roles['id']
    assert 'transaction_id' in roles['to_drop'] # IDs should also be dropped
    assert 'revenue' in roles['skewed']
    assert 'user_age' in roles['numeric']
    assert 'city' in roles['categorical']
    assert 'product_sku' in roles['high_cardinality_cat']
    assert 'has_churned' not in roles['numeric'] # Ensure target is excluded

def test_pipeline_build_for_classification_imbalanced(messy_dataframe):
    """Test building a classification pipeline for imbalanced data (should include SMOTE)."""
    orchestrator = WorkflowOrchestrator(df=messy_dataframe, target_column='has_churned', task='classification')
    pipeline, _ = orchestrator.build()
    
    assert isinstance(pipeline, ImbPipeline)
    assert 'resampler' in pipeline.named_steps
    assert pipeline.named_steps['resampler'].__class__.__name__ == 'SMOTE'

def test_pipeline_build_for_classification_balanced(balanced_dataframe):
    """Test building a classification pipeline for balanced data (should NOT include SMOTE)."""
    orchestrator = WorkflowOrchestrator(df=balanced_dataframe, target_column='target', task='classification')
    pipeline, _ = orchestrator.build()

    assert isinstance(pipeline, Pipeline)
    assert not isinstance(pipeline, ImbPipeline) 
    assert 'resampler' not in pipeline.named_steps

def test_pipeline_build_for_regression(messy_dataframe):
    """Test building a regression pipeline (should not include SMOTE)."""
    orchestrator = WorkflowOrchestrator(df=messy_dataframe, target_column='revenue', task='regression')
    pipeline, _ = orchestrator.build()

    assert isinstance(pipeline, Pipeline)
    assert not isinstance(pipeline, ImbPipeline)
    assert 'resampler' not in pipeline.named_steps
    
    # Check that the target 'revenue' is correctly removed from the 'skewed' features list
    preprocessor = pipeline.named_steps['preprocessor']
    skewed_transformer_cols = [t[2] for t in preprocessor.transformers if t[0] == 'skewed'][0]
    assert 'revenue' not in skewed_transformer_cols

def test_pipeline_build_for_clustering(messy_dataframe):
    """Test building a pipeline for an unsupervised clustering task."""
    # Clustering has no target column
    orchestrator = WorkflowOrchestrator(df=messy_dataframe, target_column=None, task='clustering')
    pipeline, roles = orchestrator.build()

    assert isinstance(pipeline, Pipeline)
    assert 'resampler' not in pipeline.named_steps
    # Ensure all original columns (except dropped ones) are considered for transformation
    assert 'has_churned' in roles['numeric'] # 'has_churned' is now just another feature

def test_unsupported_task_raises_error(messy_dataframe):
    """Test that an unsupported task raises a ValueError."""
    orchestrator = WorkflowOrchestrator(df=messy_dataframe, target_column='has_churned', task='unknown_task')
    with pytest.raises(ValueError, match="Unsupported task: 'unknown_task'"):
        orchestrator.build()

def test_missing_target_for_supervised_task_raises_error(messy_dataframe):
    """Test that a supervised task without a target column raises a ValueError."""
    with pytest.raises(ValueError, match="Task 'classification' requires a target_column"):
        WorkflowOrchestrator(df=messy_dataframe, target_column=None, task='classification')

def test_end_to_end_transformation(messy_dataframe):
    """Test that the pipeline can successfully fit and transform the data without errors."""
    orchestrator = WorkflowOrchestrator(df=messy_dataframe, target_column='has_churned', task='classification')
    pipeline, _ = orchestrator.build()
    
    X = messy_dataframe.drop(columns=['has_churned'])
    y = messy_dataframe['has_churned']
    
    try:
        X_transformed = pipeline.fit_transform(X, y)
        assert isinstance(X_transformed, np.ndarray)
        # The number of rows should be larger than original due to SMOTE
        assert X_transformed.shape[0] > X.shape[0]
    except Exception as e:
        pytest.fail(f"Pipeline fit_transform failed with an exception: {e}")