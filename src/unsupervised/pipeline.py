"""
Unsupervised Learning Pipeline Construction & Analysis Module for DataInsight AI

This module provides functions to construct pipelines for unsupervised learning
tasks and includes helper utilities to analyze the results, such as finding the
optimal number of clusters.

This is designed to be a self-contained unit for all unsupervised logic,
callable from the main application orchestrator.
"""
import logging
from typing import Dict, Any, List, Tuple, Union

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