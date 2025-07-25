"""
Data Cleaning Utilities for DataInsight AI

Key Components:
- SemanticCategoricalGrouper: A transformer that uses NLP sentence
  embeddings and clustering to automatically group semantically similar
  categorical values (e.g., "USA", "U.S.A.", "America").
"""
from collections import Counter
from typing import Dict, List, Optional
import hdbscan
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

try:
    import hdbscan
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: To use SementicCategoricalGrouper, please install required packages")
    print("pip install sentence-transformers hdbscan")

class SemanticCategoricalGrouper(BaseEstimator, TransformerMixin):
    """
    A transformer to automatically group semantically similar categorical features.

    This transformer uses pre-trained sentence embedding models to convert
    categorical levels into numerical vectors, then applies HDBSCAN clustering
    to group them. This is effective for cleaning messy, user-entered
    data with many variations of the same concept.

    Parameters
    ----------
    embedding_model_name : str, default='all-MiniLM-L6-v2'
        The name of the pre-trained model from the sentence-transformers library
        to use for creating embeddings.

    min_cluster_size : int, default=2
        The minimum number of samples in a group for it to be considered a
        cluster by HDBSCAN.
    """
    def __init__(self, embedding_model_name:str="all-MiniLM-L6-v2", min_cluster_size:int = 2):
        self.embedding_model_name = embedding_model_name
        self.min_cluster_size = min_cluster_size
        self.model_ :Optional[SentenceTransformer] = None
        self.mapping_:Dict[str, Dict[str, str]] = {}
    
    def fit(self, X:pd.DataFrame, y:Optional[pd.Series]=None):
        """
        Learns the semantic groupings for each categorical column in X.

        For each column, it gets unique values, embeds them, clusters them,
        and creates a mapping from raw values to a canonical group name.

        Parameters
        ----------
        X : pd.DataFrame
            The input data containing categorical columns to be processed.

        y : pd.Series, optional
            Ignored. Present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.model_ = SentenceTransformer(self.embedding_model_name)
        self.mappings_ = {}
        for col in X.columns:
            if X[col].dtype != "object" and not pd.api.types.is_categorical_dtype(X[col].dtype):
                continue # Skip non-categorical columns
            unique_values = X[col].dropna().unique().tolist()
            if len(unique_values) < self.min_cluster_size:
                # 1 to 1 mapping if not enough value for cluster
                self.mappings_[col] = {val:val for val in unique_values}
                continue
            # 1.Embed: Convert unique values to numerical vectors
            embeddings = self.model_.encode(unique_values)
            # 2.Cluster: Group similar vectors together
            clusterer = hdbscan.HDBSCAN(min_cluster_size = self.min_cluster_size,
                                        metrics = "euclidean",
                                        gen_min_span_tree = True)
            clusterer.fit(embeddings)
            labels = clusterer.labels_
            # 3. Create mapping: map original values to canonical group name
            col_mapping = self._create_canonical_map(unique_values, labels)
            self.mappings_[col] = col_mapping
        return self
    
    def _create_canonical_map(self, values:List[str], labels:List[str]) -> Dict[str, str]:
        """Helper to generate the value-to-group-name mapping from cluster labels."""

        df = pd.DataFrame({"value":values, "label":labels})
        mapping = {}
        # Values labeled -1 are noises and will map to themselves
        noise = df[df["label"] == -1]
        for val in noise['value']:
            mapping[val] = val
        # For each cluster, find the most common value to use as the canonical name
        clusters = df[df["label"]!=-1]
        for label_id in clusters["label"].unique():
            cluster_members = clusters[clusters["label"]==label_id]["value"].tolist()
            # Heuristic: most frequent item is the canonical name
            canonical_name = Counter(cluster_members).most_common(1)[0][0]
            for member in cluster_members:
                mapping[member] = canonical_name
        return mapping
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Applies the learned semantic groupings to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to transform.

        Returns
        -------
        pd.DataFrame
            The DataFrame with categorical values consolidated.
        """
        check_is_fitted(self, "mappings_")
        X_copy = X.copy()
        for col, mapping in self.mappings_.items():
            if col in X_copy.columns:
                original_col = X_copy[col].copy()
                X_copy[col] = X_copy[col].map(mapping)
                # Fill values not seen during fit (NaNs) with their original_value
                X_copy[col].fillna(original_col, inplace = True)
        return X_copy

                 