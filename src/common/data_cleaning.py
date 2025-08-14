"""
Data Cleaning Utilities for DataInsight AI

Key Components:
- DataCleaner: Main data cleaning orchestrator with multiple strategies
- SemanticCategoricalGrouper: A transformer that uses NLP sentence
  embeddings and clustering to automatically group semantically similar
  categorical values (e.g., "USA", "U.S.A.", "America").
"""
from collections import Counter
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import hdbscan
    from sentence_transformers import SentenceTransformer
    HAS_SEMANTIC_DEPS = True
except ImportError:
    HAS_SEMANTIC_DEPS = False
    hdbscan = None
    SentenceTransformer = None

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN_TEXT = True
except ImportError:
    HAS_SKLEARN_TEXT = False

class SemanticCategoricalGrouper(BaseEstimator, TransformerMixin):
    """
    Intelligent categorical grouper with multiple fallback strategies.
    
    Uses sentence embeddings when available, falls back to TF-IDF + KMeans,
    or simple string matching for semantic grouping of categorical values.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", min_cluster_size: int = 2):
        self.embedding_model_name = embedding_model_name
        self.min_cluster_size = min_cluster_size
        self.model_ = None
        self.mappings_ = {}
        self.fallback_mode = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learn semantic groupings using best available method."""
        self.mappings_ = {}
        
        if HAS_SEMANTIC_DEPS:
            self.fallback_mode = "semantic"  
            self.model_ = SentenceTransformer(self.embedding_model_name)
        elif HAS_SKLEARN_TEXT:
            self.fallback_mode = "tfidf"
        else:
            self.fallback_mode = "string_matching"
            
        for col in X.columns:
            if not (X[col].dtype == "object" or pd.api.types.is_categorical_dtype(X[col])):
                continue
                
            unique_values = X[col].dropna().unique().tolist()
            if len(unique_values) < self.min_cluster_size:
                self.mappings_[col] = {val: val for val in unique_values}
                continue
                
            if self.fallback_mode == "semantic":
                mapping = self._fit_semantic(unique_values)
            elif self.fallback_mode == "tfidf":
                mapping = self._fit_tfidf(unique_values)
            else:
                mapping = self._fit_string_matching(unique_values)
                
            self.mappings_[col] = mapping
            
        return self
    
    def _fit_semantic(self, unique_values: List[str]) -> Dict[str, str]:
        """Fit using sentence transformers and HDBSCAN."""
        embeddings = self.model_.encode(unique_values)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            algorithm='best'
        )
        labels = clusterer.fit_predict(embeddings)
        return self._create_canonical_map(unique_values, labels)
    
    def _fit_tfidf(self, unique_values: List[str]) -> Dict[str, str]:
        """Fallback using TF-IDF and KMeans."""
        if len(unique_values) < 3:
            return {val: val for val in unique_values}
            
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        try:
            embeddings = vectorizer.fit_transform(unique_values)
            n_clusters = min(len(unique_values) // 2, 10)
            if n_clusters < 2:
                return {val: val for val in unique_values}
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings.toarray())
            return self._create_canonical_map(unique_values, labels)
        except:
            return {val: val for val in unique_values}
    
    def _fit_string_matching(self, unique_values: List[str]) -> Dict[str, str]:
        """Simple string similarity grouping."""
        mapping = {}
        groups = {}
        
        for val in unique_values:
            val_clean = val.lower().strip()
            found_group = False
            
            for group_key, group_vals in groups.items():
                for existing_val in group_vals:
                    if self._string_similarity(val_clean, existing_val.lower().strip()) > 0.8:
                        groups[group_key].append(val)
                        mapping[val] = group_key
                        found_group = True
                        break
                if found_group:
                    break
            
            if not found_group:
                groups[val] = [val]
                mapping[val] = val
                
        return mapping
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity metric."""
        if s1 == s2:
            return 1.0
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
            
        # Simple character overlap ratio
        set1, set2 = set(s1), set(s2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
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
                X_copy[col] = X_copy[col].fillna(original_col)
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """
        Returns the feature names after transformation.
        Since this transformer modifies columns in place, the output feature
        names are the same as the input feature names.
        """
        if input_features is None:
            return [col for col in self.mappings_.keys()]
        return list(input_features)


class DataCleaner:
    """
    Comprehensive data cleaning orchestrator with multiple strategies.
    
    Handles missing values, outliers, data type conversions, and basic preprocessing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataCleaner with configuration.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with cleaning parameters
        """
        default_config = {
            'handle_missing': True,
            'missing_strategy': 'auto',  # 'auto', 'drop', 'mean', 'median', 'mode', 'forward_fill'
            'missing_threshold': 0.5,  # Drop columns with more than 50% missing
            'handle_outliers': False,
            'outlier_method': 'iqr',  # 'iqr', 'zscore'
            'outlier_threshold': 3,
            'normalize_text': True,
            'convert_dtypes': True,
            'remove_duplicates': True,
            'semantic_grouping': False
        }
        
        self.config = {**default_config, **(config or {})}
        self.semantic_grouper = None
        if self.config.get('semantic_grouping', False):
            try:
                self.semantic_grouper = SemanticCategoricalGrouper()
            except ImportError:
                print("Warning: Semantic grouping disabled due to missing dependencies")
        
        self.fitted_params = {}
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataCleaner':
        """
        Fit the data cleaner on training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.Series, optional
            Target variable
            
        Returns
        -------
        self : DataCleaner
        """
        self.fitted_params = {}
        
        # Store original dtypes
        self.fitted_params['original_dtypes'] = dict(X.dtypes)
        
        # Calculate missing value parameters
        if self.config['handle_missing']:
            missing_stats = {}
            for col in X.columns:
                if X[col].isnull().sum() > 0:
                    if X[col].dtype in ['object', 'category']:
                        missing_stats[col] = {'strategy': 'mode', 'value': X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown'}
                    elif X[col].dtype in ['float64', 'float32']:
                        missing_stats[col] = {'strategy': 'median', 'value': X[col].median()}
                    else:
                        missing_stats[col] = {'strategy': 'median', 'value': X[col].median()}
            
            self.fitted_params['missing_stats'] = missing_stats
        
        # Calculate outlier parameters
        if self.config['handle_outliers']:
            outlier_params = {}
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if self.config['outlier_method'] == 'iqr':
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_params[col] = {'lower': lower_bound, 'upper': upper_bound}
                elif self.config['outlier_method'] == 'zscore':
                    mean_val = X[col].mean()
                    std_val = X[col].std()
                    threshold = self.config['outlier_threshold']
                    outlier_params[col] = {
                        'mean': mean_val, 
                        'std': std_val, 
                        'threshold': threshold
                    }
            
            self.fitted_params['outlier_params'] = outlier_params
        
        # Fit semantic grouper if enabled
        if self.semantic_grouper is not None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                self.semantic_grouper.fit(X[categorical_cols])
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cleaning transformations to data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to clean
            
        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        if not self.is_fitted:
            raise ValueError("DataCleaner must be fitted before transform")
        
        X_clean = X.copy()
        
        # Remove duplicates
        if self.config['remove_duplicates']:
            initial_shape = X_clean.shape[0]
            X_clean = X_clean.drop_duplicates()
            if X_clean.shape[0] < initial_shape:
                print(f"Removed {initial_shape - X_clean.shape[0]} duplicate rows")
        
        # Handle missing values
        if self.config['handle_missing'] and 'missing_stats' in self.fitted_params:
            # First, drop columns with too many missing values
            cols_to_drop = []
            for col in X_clean.columns:
                missing_ratio = X_clean[col].isnull().sum() / len(X_clean)
                if missing_ratio > self.config['missing_threshold']:
                    cols_to_drop.append(col)
            
            if cols_to_drop:
                print(f"Dropping columns with high missing values: {cols_to_drop}")
                X_clean = X_clean.drop(columns=cols_to_drop)
            
            # Fill missing values
            for col, stats in self.fitted_params['missing_stats'].items():
                if col in X_clean.columns and X_clean[col].isnull().sum() > 0:
                    X_clean[col] = X_clean[col].fillna(stats['value'])
        
        # Handle outliers
        if self.config['handle_outliers'] and 'outlier_params' in self.fitted_params:
            for col, params in self.fitted_params['outlier_params'].items():
                if col not in X_clean.columns:
                    continue
                    
                if self.config['outlier_method'] == 'iqr':
                    # Cap outliers to bounds
                    X_clean[col] = X_clean[col].clip(
                        lower=params['lower'], 
                        upper=params['upper']
                    )
                elif self.config['outlier_method'] == 'zscore':
                    # Cap outliers based on z-score
                    z_scores = np.abs((X_clean[col] - params['mean']) / params['std'])
                    outlier_mask = z_scores > params['threshold']
                    if outlier_mask.sum() > 0:
                        median_val = X_clean[col].median()
                        X_clean.loc[outlier_mask, col] = median_val
        
        # Normalize text columns
        if self.config['normalize_text']:
            text_cols = X_clean.select_dtypes(include=['object']).columns
            for col in text_cols:
                if X_clean[col].dtype == 'object':
                    X_clean[col] = X_clean[col].astype(str).str.strip().str.lower()
                    # Replace empty strings with NaN
                    X_clean[col] = X_clean[col].replace('', np.nan)
        
        # Apply semantic grouping
        if self.semantic_grouper is not None:
            categorical_cols = X_clean.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                semantic_cols = [col for col in categorical_cols if col in self.semantic_grouper.mappings_]
                if semantic_cols:
                    X_clean[semantic_cols] = self.semantic_grouper.transform(X_clean[semantic_cols])
        
        # Convert data types
        if self.config['convert_dtypes']:
            X_clean = self._optimize_dtypes(X_clean)
        
        return X_clean
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform data in one step."""
        return self.fit(X, y).transform(X)
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type == 'object':
                # Try to convert to numeric if possible
                try:
                    numeric_series = pd.to_numeric(optimized_df[col], errors='coerce')
                    if not numeric_series.isnull().all():
                        optimized_df[col] = numeric_series
                        col_type = numeric_series.dtype
                except:
                    pass
            
            # Optimize numeric types
            if col_type in ['int64', 'int32', 'int16', 'int8']:
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
            
            elif col_type in ['float64', 'float32']:
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
        
        return optimized_df
    
    def get_cleaning_report(self, X_original: pd.DataFrame, X_cleaned: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a report of cleaning operations performed.
        
        Parameters
        ----------
        X_original : pd.DataFrame
            Original data before cleaning
        X_cleaned : pd.DataFrame
            Data after cleaning
            
        Returns
        -------
        dict
            Report with cleaning statistics
        """
        report = {
            'original_shape': X_original.shape,
            'cleaned_shape': X_cleaned.shape,
            'rows_removed': X_original.shape[0] - X_cleaned.shape[0],
            'columns_removed': X_original.shape[1] - X_cleaned.shape[1],
            'missing_values_before': X_original.isnull().sum().sum(),
            'missing_values_after': X_cleaned.isnull().sum().sum(),
            'duplicate_rows_removed': 0,
            'data_types_changed': [],
            'memory_usage_before': X_original.memory_usage(deep=True).sum(),
            'memory_usage_after': X_cleaned.memory_usage(deep=True).sum()
        }
        
        # Check for data type changes
        for col in X_original.columns:
            if col in X_cleaned.columns:
                if X_original[col].dtype != X_cleaned[col].dtype:
                    report['data_types_changed'].append({
                        'column': col,
                        'before': str(X_original[col].dtype),
                        'after': str(X_cleaned[col].dtype)
                    })
        
        # Calculate memory savings
        memory_saved = report['memory_usage_before'] - report['memory_usage_after']
        report['memory_saved_mb'] = memory_saved / (1024 * 1024)
        report['memory_savings_percent'] = (memory_saved / report['memory_usage_before']) * 100 if report['memory_usage_before'] > 0 else 0
        
        return report