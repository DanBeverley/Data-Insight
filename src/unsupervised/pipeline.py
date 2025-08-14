"""
Unsupervised Learning Pipeline Construction & Analysis Module for DataInsight AI

This module provides functions to construct pipelines for unsupervised learning
tasks and includes helper utilities to analyze the results, such as finding the
optimal number of clusters.

This is designed to be a self-contained unit for all unsupervised logic,
callable from the main application orchestrator.
"""
import logging
from typing import Dict, Any, List, Tuple, Union, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Mapping from string identifiers to corresponding model classes. Allows for easy extension
UNSUPERVISED_ALGORITHMS:Dict[str, BaseEstimator] = {"kmeans":KMeans,
                                                    "dbscan":DBSCAN,
                                                    "pca":PCA}

def create_unsupervised_pipeline(preprocessor:ColumnTransformer, algorithm:str,
                                 params:Dict[str, Any]) -> Pipeline:
    """
    Constructs an unsupervised learning pipeline.

    This factory function takes a pre-configured preprocessor, selects the
    specified unsupervised learning algorithm, configures it with the provided
    parameters, and attaches it to create a complete pipeline.

    Args:
        preprocessor: The scikit-learn ColumnTransformer for feature processing.
        algorithm: The string identifier for the desired algorithm
                   (e.g., 'kmeans', 'pca').
        params: A dictionary of parameters to pass to the algorithm's constructor
                (e.g., {'n_clusters': 3} for KMeans).

    Returns:
        A scikit-learn Pipeline object, ready for fitting.
    
    Raises:
        NotImplementedError: If the specified algorithm is not supported.
    """
    logging.info(f"Building unsupervised pipeline for algorithm: '{algorithm}'")
    
    if algorithm.lower() not in UNSUPERVISED_ALGORITHMS:
        raise NotImplementedError(f"Algorithm '{algorithm}' is not supported."
                                  f"Available options:{list(UNSUPERVISED_ALGORITHMS)}")
    
    EstimatorClass = UNSUPERVISED_ALGORITHMS[algorithm.lower()]
    estimator = EstimatorClass(**params)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                               ("estimator", estimator)])
    
    logging.info(f"Created pipeline with steps: ['preprocessor', 'estimator' ({EstimatorClass.__name__})]")
    return pipeline

def find_optimal_clusters_elbow(preprocessed_data:pd.DataFrame, k_range:Tuple[int, int]=(2,11)) -> Dict[int, float]:
    """
    Calculates the inertia for a range of k values for the Elbow Method.

    This helper function is intended to be called *before* the final pipeline
    is built, to help the user choose the optimal number of clusters for KMeans.

    Args:
        preprocessed_data: The data *after* it has been passed through the
                           preprocessor (imputed, scaled, encoded).
        k_range: A tuple specifying the (min, max) number of clusters to test.

    Returns:
        A dictionary mapping the number of clusters (k) to the inertia score.
    """
    logging.info(f"Running Elbow Method analysis for k in range {k_range}...")
    inertias = {}
    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters=k, init = "k-means++", n_init=10, random_state=42)
        kmeans.fit(preprocessed_data)
        inertias[k] = kmeans.inertia_
        logging.debug(f"k={k}, inertia={inertias[k]}")
    logging.info("Elbow Method analysis complete")
    return inertias


class UnsupervisedPipeline:
    """
    Wrapper class for unsupervised learning pipeline creation and management.
    
    Provides a unified interface for creating clustering and dimensionality reduction pipelines.
    """
    
    def __init__(self, task: str = 'clustering', config: Optional[Dict[str, Any]] = None):
        """
        Initialize UnsupervisedPipeline.
        
        Parameters
        ----------
        task : str
            Type of unsupervised learning task ('clustering' or 'dimensionality_reduction')
        config : dict, optional
            Configuration dictionary for pipeline parameters
        """
        self.task = task
        self.config = config or {}
        self.pipeline = None
        self.is_fitted = False
        
    def create_pipeline(self, preprocessor, n_clusters: int = None, algorithm: str = 'kmeans'):
        """
        Create an unsupervised learning pipeline.
        
        Parameters
        ----------
        preprocessor : sklearn transformer
            Preprocessing pipeline
        n_clusters : int, optional
            Number of clusters for clustering algorithms
        algorithm : str
            Algorithm to use ('kmeans', 'hierarchical', 'dbscan')
            
        Returns
        -------
        sklearn.Pipeline
            Configured pipeline
        """
        self.pipeline = create_unsupervised_pipeline(
            preprocessor=preprocessor,
            task=self.task,
            n_clusters=n_clusters,
            algorithm=algorithm
        )
        return self.pipeline
    
    def fit(self, X: pd.DataFrame, **kwargs):
        """Fit the pipeline."""
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_pipeline() first.")
        
        self.pipeline.fit(X, **kwargs)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        return self.pipeline.predict(X)
    
    def fit_predict(self, X: pd.DataFrame):
        """Fit and predict in one step."""
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_pipeline() first.")
        
        result = self.pipeline.fit_predict(X)
        self.is_fitted = True
        return result
    
    def transform(self, X: pd.DataFrame):
        """Transform data (for dimensionality reduction)."""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        return self.pipeline.transform(X)
    
    def get_cluster_centers(self):
        """Get cluster centers (for clustering algorithms that support it)."""
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        if hasattr(self.pipeline.named_steps.get('model', None), 'cluster_centers_'):
            return self.pipeline.named_steps['model'].cluster_centers_
        else:
            raise ValueError("Current algorithm doesn't support cluster centers")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        if self.pipeline is None:
            return {'status': 'not_created'}
        
        info = {
            'status': 'fitted' if self.is_fitted else 'created',
            'task': self.task,
            'steps': [name for name, _ in self.pipeline.steps],
            'model_type': type(self.pipeline.named_steps.get('model', None)).__name__
        }
        
        return info