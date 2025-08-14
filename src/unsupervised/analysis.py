"""
Unsupervised Learning Analysis & Visualization Module for DataInsight AI

This module provides functions to analyze the output of unsupervised models,
primarily focusing on clustering evaluation and visualization. These helpers
are designed to provide users with actionable insights into the structure
discovered in their data.

It is intended to be called from the front-end (`app.py`) after a clustering
pipeline has been fitted.
"""
import logging
from typing import Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def generate_cluster_report(preprocessed_data:pd.DataFrame, labels:pd.Series) -> Dict[str, Any]:
    """
    Calculates key metrics to evaluate the quality of clustering results.

    Args:
        preprocessed_data: The data *after* it has been passed through the
                           preprocessor (imputed, scaled, encoded). This should
                           be the data that was fed into the clustering algorithm.
        labels: A Series or array containing the cluster label for each data point.

    Returns:
        A dictionary containing the number of clusters, noise points, and key
        clustering evaluation metrics.
    """
    if labels.nunique() <= 1:
        logging.warning("Cannot calculate clustering metrics with 1 or fewer clusters")
        return {"Number of Clusters": labels.nunique(),
                "Silhouette Score": "N/A",
                "Calinski-Harabasz Score":"N/A",
                "Davies-Bouldin Score":"N/A"}
    report = {"Number of Clusters":int(labels[labels != -1].nunique()),
              "Number of Noise Points":int((labels == -1).sum()),
              "Silhouette Score": silhouette_score(preprocessed_data, labels),
              "Calinski-Harabasz Score": calinski_harabasz_score(preprocessed_data, labels),
              "Davies-Bouldin Score":davies_bouldin_score(preprocessed_data, labels)}
    
    logging.info(f"Generated cluster report: {report}")
    return report

def plot_elbow_method(inertia_results:Dict[int, float]) -> go.Figure:
    """
    Creates an interactive plot of the Elbow Method results using Plotly.

    Args:
        inertia_results: A dictionary mapping k (number of clusters) to inertia.

    Returns:
        A Plotly Figure object.
    """
    k_values = list(inertia_results.keys())
    inertia_values = list(inertia_results.values())

    fig = go.Figure(data=go.Scatter(x=k_values, y=inertia_values, mode="lines+markers"))
    fig.update_layout(title = "Elbow Method for Optimal Number of Clusters",
                      xaxis_title="Number of Clusters (k)",
                      yaxis_title="Inertia (Within-cluster sum of squares)",
                      template="plotly_white")
    return fig

def plot_cluster_results_2d(data_2d:pd.DataFrame, labels:pd.Series, title:str = "Cluster Visualization (2D Projection)") -> go.Figure:
    """
    Creates an interactive 2D scatter plot of clustering results using Plotly.

    The input data should be 2-dimensional, typically the output of a
    dimensionality reduction technique like PCA or UMAP.

    Args:
        data_2d: A DataFrame with two columns representing the 2D coordinates.
        labels: A Series containing the cluster label for each data point.
        title: The title for the plot.

    Returns:
        A Plotly Figure object.
    """
    if data_2d.shape[1] != 2:
        raise ValueError(f"Input data must have exactly 2 columns for 2D plotting, but got {data_2d.shape[1]}")
    plot_df = data_2d.copy()
    plot_df.columns = ["Dimension 1", "Dimension 2"]
    plot_df["Cluster"] = labels.astype(str)

    fig = px.scatter(plot_df, 
                     x="Dimension 1",
                     y="Dimension 2",
                     color="Cluster",
                     title=title,
                     template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_traces(marker=dict(size=8, opacity = 0.8))
    fig.update_layout(legend_title_text="Cluster")
    return fig