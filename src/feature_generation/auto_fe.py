"""Automated Feature Engineering Module for DataInsight AI"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

try:
    import featuretools as ft
    from featuretools.primitives import (
        Count, Mean, Max, Min, Std, Sum,
        Day, Month, Year, Weekday, IsWeekend
    )
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    ft = None
    FEATURETOOLS_AVAILABLE = False
    logging.warning("Featuretools not available. Using basic feature engineering only.")

class AutomatedFeatureEngineer:
    """Orchestrate automated feature generation using multiple techniques."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.entityset: Optional[Any] = None
        self.generated_features: List = []
        self.feature_matrix: Optional[pd.DataFrame] = None

    def generate_features(
        self,
        dataframes: Dict[str, pd.DataFrame],
        relationships: List[Tuple[str, str, str, str]] = None,
        target_entity: str = None
    ) -> pd.DataFrame:
        """Generate features using available methods."""
        
        if not dataframes:
            raise ValueError("dataframes dictionary cannot be empty")
        
        # If single dataframe, use polynomial and interaction features
        if len(dataframes) == 1 and not relationships:
            df_name, df = list(dataframes.items())[0]
            return self._generate_single_table_features(df)
        
        # Use featuretools for relational data if available
        if FEATURETOOLS_AVAILABLE and relationships:
            return self._generate_relational_features(dataframes, relationships, target_entity)
        
        # Fallback to basic feature engineering
        logging.warning("Using basic feature engineering only")
        main_df = list(dataframes.values())[0]
        return self._generate_single_table_features(main_df)

    def _generate_single_table_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features from single table using polynomial and interaction features."""
        
        result_df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            logging.warning("Insufficient numeric columns for feature generation")
            return result_df
        
        # Polynomial features
        if self.config.get('polynomial_features', True):
            degree = self.config.get('polynomial_degree', 2)
            max_features = min(len(numeric_cols), 10)  # Limit to prevent explosion
            
            if max_features >= 2:
                poly = PolynomialFeatures(
                    degree=degree, 
                    include_bias=False,
                    interaction_only=False
                )
                
                poly_features = poly.fit_transform(df[numeric_cols[:max_features]])
                poly_feature_names = poly.get_feature_names_out(numeric_cols[:max_features])
                
                # Add only interaction features (skip original features)
                for i, name in enumerate(poly_feature_names):
                    if name not in numeric_cols:  # Skip original features
                        result_df[f"poly_{name}"] = poly_features[:, i]
        
        # Ratio features
        if self.config.get('ratio_features', True) and len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i + 1, min(i + 5, len(numeric_cols))):  # Limit combinations
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    
                    # Avoid division by zero
                    denominator = df[col2].replace(0, np.nan)
                    if not denominator.isna().all():
                        result_df[f"ratio_{col1}_{col2}"] = df[col1] / denominator
        
        # Binning features
        if self.config.get('binning_features', True):
            n_bins = self.config.get('n_bins', 5)
            for col in numeric_cols[:5]:  # Limit to first 5 columns
                try:
                    result_df[f"{col}_binned"] = pd.cut(
                        df[col], 
                        bins=n_bins, 
                        labels=False, 
                        duplicates='drop'
                    )
                except Exception:
                    continue
        
        # Statistical features
        if self.config.get('statistical_features', True):
            # Row-wise statistics for numeric columns
            numeric_data = df[numeric_cols]
            result_df['row_mean'] = numeric_data.mean(axis=1)
            result_df['row_std'] = numeric_data.std(axis=1)
            result_df['row_min'] = numeric_data.min(axis=1)
            result_df['row_max'] = numeric_data.max(axis=1)
            result_df['row_range'] = result_df['row_max'] - result_df['row_min']
        
        logging.info(f"Generated {len(result_df.columns) - len(df.columns)} new features")
        return result_df

    def _generate_relational_features(
        self,
        dataframes: Dict[str, pd.DataFrame],
        relationships: List[Tuple[str, str, str, str]],
        target_entity: str
    ) -> pd.DataFrame:
        """Generate features using featuretools DFS."""
        
        if not FEATURETOOLS_AVAILABLE:
            raise ImportError("Featuretools is required for relational feature generation")
        
        logging.info("Starting Deep Feature Synthesis...")
        
        self.entityset = ft.EntitySet(id="datainsight_es")
        
        # Add dataframes to entityset
        for entity_name, df_original in dataframes.items():
            df = df_original.copy()
            index_col = self._find_best_index_column(df, entity_name)
            
            self.entityset.add_dataframe(
                dataframe_name=entity_name,
                dataframe=df,
                index=index_col,
                make_index=False
            )
        
        # Add relationships
        for parent_entity, parent_variable, child_entity, child_variable in relationships:
            self.entityset.add_relationship(
                parent_dataframe_name=parent_entity,
                parent_column_name=parent_variable,
                child_dataframe_name=child_entity,
                child_column_name=child_variable
            )
        
        # Define primitives
        agg_primitives = [Sum, Mean, Max, Min, Std, Count]
        trans_primitives = [Day, Month, Year, Weekday, IsWeekend]
        
        # Run DFS
        self.feature_matrix, self.generated_features = ft.dfs(
            entityset=self.entityset,
            target_dataframe_name=target_entity,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            verbose=self.config.get('dfs_verbose', False),
            max_depth=self.config.get('dfs_max_depth', 2),
            n_jobs=self.config.get('dfs_n_jobs', 1)
        )
        
        logging.info(f"DFS complete. Generated {len(self.generated_features)} features")
        return self.feature_matrix.reset_index()

    def _find_best_index_column(self, df: pd.DataFrame, entity_name: str) -> str:
        """Identify the best primary key column for an entity."""
        
        scores = {}
        total_rows = len(df)
        
        if total_rows == 0:
            raise ValueError(f"DataFrame for entity '{entity_name}' is empty")
        
        for col in df.columns:
            scores[col] = 0
            series = df[col]
            
            # High score for unique columns
            if series.nunique() == total_rows:
                scores[col] += 100
            
            # Bonus for ID-like names
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['id', 'key', 'pk', 'uuid']):
                scores[col] += 50
            
            # Prefer integer or object types
            if pd.api.types.is_integer_dtype(series.dtype):
                scores[col] += 20
            elif pd.api.types.is_object_dtype(series.dtype):
                scores[col] += 15
            
            # Bonus for no nulls
            if series.isnull().sum() == 0:
                scores[col] += 10
        
        if not scores:
            raise ValueError(f"Could not score any columns for entity '{entity_name}'")
        
        best_column = max(scores, key=scores.get)
        
        if df[best_column].nunique() != total_rows:
            logging.warning(
                f"Index column '{best_column}' for '{entity_name}' is not fully unique. "
                "This may cause issues in feature generation."
            )
        
        return best_column

class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate interaction features between numeric columns."""
    
    def __init__(self, max_interactions: int = 20):
        self.max_interactions = max_interactions
        self.feature_names_ = []
        self.interaction_pairs_ = []

    def fit(self, X: pd.DataFrame, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Select top interactions based on correlation with target if available
        if y is not None and len(numeric_cols) > 1:
            correlations = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    interaction = X[col1] * X[col2]
                    corr = abs(np.corrcoef(interaction, y)[0, 1])
                    if not np.isnan(corr):
                        correlations.append((corr, col1, col2))
            
            # Sort by correlation and take top interactions
            correlations.sort(reverse=True)
            self.interaction_pairs_ = [
                (col1, col2) for _, col1, col2 in correlations[:self.max_interactions]
            ]
        else:
            # Random selection if no target
            import itertools
            pairs = list(itertools.combinations(numeric_cols, 2))
            self.interaction_pairs_ = pairs[:self.max_interactions]
        
        self.feature_names_ = [f"{col1}_x_{col2}" for col1, col2 in self.interaction_pairs_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        
        for col1, col2 in self.interaction_pairs_:
            if col1 in X.columns and col2 in X.columns:
                result[f"{col1}_x_{col2}"] = X[col1] * X[col2]
        
        return result

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

def create_feature_engineering_config(
    df: pd.DataFrame,
    task_type: str = 'regression',
    complexity: str = 'medium'
) -> Dict[str, Any]:
    """Generate feature engineering configuration based on data characteristics."""
    
    n_samples, n_features = df.shape
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    
    if complexity == 'high':
        config = {
            'polynomial_features': True,
            'polynomial_degree': 2,
            'ratio_features': True,
            'binning_features': True,
            'n_bins': 10,
            'statistical_features': True,
            'interaction_features': True,
            'max_interactions': min(50, numeric_cols * (numeric_cols - 1) // 2)
        }
    elif complexity == 'low':
        config = {
            'polynomial_features': False,
            'ratio_features': True,
            'binning_features': False,
            'statistical_features': True,
            'interaction_features': False
        }
    else:  # medium
        config = {
            'polynomial_features': numeric_cols <= 10,
            'polynomial_degree': 2,
            'ratio_features': True,
            'binning_features': numeric_cols <= 15,
            'n_bins': 5,
            'statistical_features': True,
            'interaction_features': numeric_cols >= 3,
            'max_interactions': min(20, max(5, numeric_cols))
        }
    
    # Adjust for small datasets
    if n_samples < 1000:
        config['polynomial_features'] = False
        config['binning_features'] = False
        config['max_interactions'] = min(10, config.get('max_interactions', 10))
    
    return config


# Alias for backward compatibility
AutoFeatureEngineer = AutomatedFeatureEngineer