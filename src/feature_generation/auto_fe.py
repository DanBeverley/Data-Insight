# src/feature_generation/auto_fe.py

"""
Automated Feature Engineering Module for DataInsight AI (Production-Grade)

This module provides the `AutomatedFeatureEngineer` class, responsible for
generating a rich set of new features from relational datasets using
techniques like Deep Feature Synthesis (DFS). It includes intelligent logic
to automatically identify primary key columns.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import featuretools as ft
import pandas as pd
from featuretools.primitives import (
    Count, Mean, Max, Min, Std, Sum,
    Day, Month, Year, Weekday, IsWeekend
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AutomatedFeatureEngineer:
    """
    A class to orchestrate automated feature generation using Featuretools.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.entityset: Optional[ft.EntitySet] = None
        self.generated_features: List[ft.FeatureBase] = []
        self.feature_matrix: Optional[pd.DataFrame] = None

    def generate_features(
        self,
        dataframes: Dict[str, pd.DataFrame],
        relationships: List[Tuple[str, str, str, str]],
        target_entity: str
    ) -> pd.DataFrame:
        """
        Generates features using Deep Feature Synthesis (DFS).

        Args:
            dataframes: A dictionary where keys are entity names (e.g., 'customers')
                        and values are the corresponding pandas DataFrames.
            relationships: A list of tuples defining relationships between entities.
                           Each tuple is (parent_entity, parent_variable,
                           child_entity, child_variable).
            target_entity: The name of the entity for which to generate features.

        Returns:
            A new DataFrame containing the original features plus all newly
            generated features.
        """
        logging.info("Starting automated feature generation with Deep Feature Synthesis...")
        if not dataframes or target_entity not in dataframes:
            raise ValueError("`dataframes` dictionary must not be empty and must contain the `target_entity`.")

        self.entityset = ft.EntitySet(id="datainsight_es")

        for entity_name, df_original in dataframes.items():
            df = df_original.copy()
            index_col = self._find_best_index_column(df, entity_name)
            
            self.entityset.add_dataframe(
                dataframe_name=entity_name,
                dataframe=df,
                index=index_col,
                make_index=False 
            )
        
        for parent_entity, parent_variable, child_entity, child_variable in relationships:
            self.entityset.add_relationship(
                parent_dataframe_name=parent_entity,
                parent_column_name=parent_variable,
                child_dataframe_name=child_entity,
                child_column_name=child_variable
            )

        agg_primitives = [Sum, Mean, Max, Min, Std, Count]
        trans_primitives = [Day, Month, Year, Weekday, IsWeekend]

        logging.info(f"Running DFS for target entity: '{target_entity}'")
        self.feature_matrix, self.generated_features = ft.dfs(
            entityset=self.entityset,
            target_dataframe_name=target_entity,
            agg_primitives=agg_primitives,
            trans_primitives=trans_primitives,
            verbose=self.config.get('dfs_verbose', False),
            max_depth=self.config.get('dfs_max_depth', 2),
            n_jobs=self.config.get('dfs_n_jobs', -1)
        )
        
        logging.info(f"DFS complete. Generated {len(self.generated_features)} features.")
        return self.feature_matrix.reset_index()

    def _find_best_index_column(self, df: pd.DataFrame, entity_name: str) -> str:
        """
        Intelligently identifies the best primary key (index) for a DataFrame.

        This sophisticated method scores each column based on uniqueness, naming
        conventions, data type, and nullability to find the most likely
        primary key.
        """
        logging.info(f"Finding best index for entity '{entity_name}'...")
        scores = {}
        total_rows = len(df)
        if total_rows == 0:
            raise ValueError(f"DataFrame for entity '{entity_name}' is empty.")

        for col in df.columns:
            scores[col] = 0
            series = df[col]
            
            if series.nunique() == total_rows:
                scores[col] += 100
            
            col_lower = str(col).lower()
            if 'id' in col_lower or 'key' in col_lower or 'pk' in col_lower or 'uuid' in col_lower:
                scores[col] += 50

            if pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype):
                scores[col] += 20
            elif pd.api.types.is_float_dtype(series.dtype):
                scores[col] -= 50

            if series.isnull().sum() == 0:
                scores[col] += 10
        
        if not scores:
            raise ValueError(f"Could not score any columns for entity '{entity_name}'.")

        best_column = max(scores, key=scores.get)
        logging.info(f"Best index for '{entity_name}' identified as '{best_column}' with score {scores[best_column]}.")
        
        if df[best_column].nunique() != total_rows:
             logging.warning(
                f"Column '{best_column}' chosen as index for '{entity_name}', "
                "but it is NOT fully unique. This may cause issues in feature generation."
            )
            
        return best_column