"""
The Core Automation Engine for DataInsight AI

This module contains the `PipelineAutomator`, the central class responsible for
inspecting raw data and dynamically constructing a tailored preprocessing
pipeline
"""
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .config import settings
from .supervised.pipeline import create_supervised_pipeline
from .unsupervised.pipeline import create_unsupervised_pipeline
from .utils import generate_lineage_report
from .common.data_cleaning import SemanticCategoricalGrouper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for category_encoders.TargetEncoder to ensure compatibility
    with scikit-learn's get_feature_names_out.
    """
    def __init__(self):
        self.encoder = TargetEncoder()
        self.feature_names_in_ = None

    def fit(self, X, y):
        self.encoder.fit(X, y)
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        return self.encoder.transform(X)

    def get_feature_names_out(self, input_features=None):
        return list(self.feature_names_in_)

class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()

@dataclass
class OrchestratorResult:
    """A structured object to hold the results of an orchestration run."""
    status:Status
    pipeline:Optional[Union[Pipeline, ImbPipeline]] = None
    processed_features:Optional[pd.DataFrame] = None
    aligned_target:Optional[pd.Series] = None
    lineage_report:Optional[Dict[str, Any]] = None
    error_message:Optional[str] = None
    column_roles:Dict[str, List[str]] = field(default_factory=dict)

class PipelineConstructionError(Exception):
    """Custom exception for failures during pipeline construction"""
    pass

class WorkflowOrchestrator:
    """
    Orchestrates the creation of an automated preprocessing pipeline.

    This class takes a pandas DataFrame, profiles it, classifies its columns
    based on heuristics defined in `config.yml`, and builds a scikit-learn
    ColumnTransformer and Pipeline tailored to the data's specific needs and
    the desired machine learning task.

    Attributes:
        df (pd.DataFrame): The raw input DataFrame.
        schema (Dict): A dictionary containing profiling metadata for each column.
        column_roles (Dict[str, List[str]]): A dictionary mapping column roles
            (e.g., 'numeric', 'skewed', 'categorical') to lists of column names.
    """
    def __init__(self, df:pd.DataFrame, target_column:str, task:str):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df:pd.DataFrame = df.copy()
        self.schema:Dict[str, Any] = {}
        self.column_roles:Dict[str, List[str]] = {}
        self.target_column:Optional[str] = target_column
        self.task:str = task
    
    def run(self) -> OrchestratorResult:
        """
        Executes the full preprocessing workflow from profiling to final output.

        Returns:
            An OrchestratorResult object containing the outcome of the run.
        """
        try:
            logging.info(f"Orchestrating for task: '{self.task}'")
            self._profile_data()
            self._classify_columns()
            self._prepare_feature_lists()

            preprocessor = self._build_preprocessor()
            if self.task in ["classification", "regression"]:
                y = self.df[self.target_column] if self.target_column else None
                pipeline = create_supervised_pipeline(preprocessor, self.task, y)
            elif self.task == "clustering":
                # Example: Defaulting to KMeans with 3 clusters for demo
                # In a real app, this would be configured in the UI
                params = {"n_clusters":3, "n_init":10, "random_state":42}
                pipeline = create_unsupervised_pipeline(preprocessor, "kmeans", params)
            else:
                raise NotImplementedError(f"Task '{self.task}' is not yet implemented")
            
            X = self.df.drop(columns=[self.target_column]) if self.target_column else self.df
            y = self.df[self.target_column] if self.target_column and self.task in ["classification", "regression"] else None
            
            preprocessor_step = pipeline.named_steps['preprocessor']
            if y is not None:
                preprocessor_step.fit(X, y)
            else:
                preprocessor_step.fit(X)
            
            X_transformed = preprocessor_step.transform(X)
            
            # Reconstruct DataFrame with proper column names
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            processed_features = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
            
            # Align the target variable
            y_aligned = self.df.loc[processed_features.index, self.target_column] if self.target_column else None

            # Drop NaNs from BOTH the processed features and the aligned target
            nan_rows_mask = processed_features.isnull().any(axis=1)
            if nan_rows_mask.any():
                logging.warning(f"Dropping {nan_rows_mask.sum()} rows with NaNs after transformation.")
                processed_features = processed_features.loc[~nan_rows_mask]
                if y_aligned is not None:
                    y_aligned = y_aligned.loc[~nan_rows_mask]

            # Generate lineage report
            lineage = generate_lineage_report(
                self.column_roles, pipeline, settings._config_data, self.task
            )

            logging.info("Workflow orchestration completed successfully.")
            return OrchestratorResult(
                status=Status.SUCCESS,
                pipeline=pipeline,
                processed_features=processed_features,
                aligned_target=y_aligned,
                lineage_report=lineage,
                column_roles=self.column_roles
            )

        except (PipelineConstructionError, NotImplementedError, ValueError, TypeError) as e:
            logging.error(f"A predictable error occurred during orchestration: {e}", exc_info=True)
            return OrchestratorResult(status=Status.FAILURE, error_message=str(e))
        except Exception as e:
            logging.error(f"An unexpected error occurred during orchestration: {e}", exc_info=True)
            return OrchestratorResult(
                status=Status.FAILURE,
                error_message="An unexpected internal error occurred. Please check the logs."
            )
    
    def _profile_data(self) -> None:
        """Analyzes the DataFrame to create a metadata for each column"""
        logging.info("Data Profiling...")
        for col in self.df.columns:
            series = self.df[col]
            self.schema[col] = {"dtype":series.dtype,
                                "missing_percent":series.isnull().sum()/len(series),
                                "unique_count":series.nunique(),
                                "cardinality":series.nunique()/len(series),
                                "skewness":series.skew() if pd.api.types.is_numeric_dtype(series) and len(series.dropna()) > 0 else 0.0}
        logging.info("Data profiling complete")
    
    def _classify_columns(self) -> None:
        """Classifies columns into roles based on the schema and config heurestics"""
        logging.info("Classifying column roles based on heuristics...")
        roles:Dict[str, List[str]] = {"to_drop":[],
                                      "numeric":[],
                                      "skewed":[],
                                      "categorical":[],
                                      "high_cardinality_cat":[],
                                      "id":[]}
        heuristics = settings['heuristics']
        feature_columns = [col for col in self.df.columns if col != self.target_column]

        for col in feature_columns:
            profile = self.schema[col]
            
            # Priority 1: Check for dropping first.
            if profile["missing_percent"] > heuristics["missing_value_drop_threshold"]:
                roles["to_drop"].append(col)
                continue

            # Priority 2: Check for ID columns based on cardinality.
            is_potential_id = np.isclose(profile["cardinality"], heuristics["id_cardinality_threshold"])
            is_float = pd.api.types.is_float_dtype(profile["dtype"])
            if is_potential_id and not is_float:
                roles["id"].append(col)
                continue

            # Priority 3: Handle all numeric types (skewed or normal).
            if pd.api.types.is_numeric_dtype(profile["dtype"]):
                if abs(profile["skewness"]) >= heuristics["skewed_feature_threshold"]:
                    roles["skewed"].append(col)
                else:
                    roles["numeric"].append(col)
                continue

            # Priority 4: Handle all categorical/object types.
            if pd.api.types.is_object_dtype(profile["dtype"]) or pd.api.types.is_categorical_dtype(profile["dtype"]):
                if profile["cardinality"] > heuristics["high_cardinality_threshold"]:
                    roles["high_cardinality_cat"].append(col)
                else:
                    roles["categorical"].append(col)
                continue
            
            # Fallback for unhandled dtypes
            logging.warning(f"Column '{col}' with dtype '{profile['dtype']}' was not classified and will be dropped.")
            roles["to_drop"].append(col)
        
        roles['to_drop'].extend(roles['id'])
        self.column_roles = roles
        logging.info(f"Column roles classified: { {k: len(v) for k, v in roles.items()} }")

    def _prepare_feature_lists(self) -> None:
        if self.target_column in self.df.columns:
            transform_roles = ["numeric", "skewed", "categorical", "high_cardinality_cat"]
            for role in transform_roles:
                if self.target_column in self.column_roles[role]:
                    self.column_roles[role].remove(self.target_column)
                    logging.warning(f"Target column '{self.target_column}' removed from '{role}' transform list.")

    def _build_numeric_transformer(self) -> Pipeline:
        """Builds the pipeline for standard numeric features"""
        imputer = SimpleImputer(strategy=settings['pipeline_params']['numeric_imputer_strategy'])
        imputer.set_output(transform="pandas")
        return Pipeline(steps=[
            ("imputer", imputer),
            ("scaler", StandardScaler())
        ])
    def _build_skewed_transformer(self) -> Pipeline:
        """Builds the pipeline for skewed numeric features"""
        imputer = SimpleImputer(strategy=settings['pipeline_params']['numeric_imputer_strategy'])
        imputer.set_output(transform="pandas")
        return Pipeline(steps=[
            ("imputer", imputer),
            ("power_transform", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler())
        ])
    
    def _build_categorical_transformer(self) -> Pipeline:
        """Builds the pipeline for categorical features, with optional semantic grouping"""
        params = settings['pipeline_params']
        adv_features = settings['advanced_features']

        imputer = SimpleImputer(strategy=params['categorical_imputer_strategy'], fill_value="missing")
        imputer.set_output(transform="pandas")
        cat_steps = [
            ("imputer", imputer),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None))
        ]
        
        if adv_features['use_semantic_categorical_grouping']:
            logging.info("Semantic grouping enabled. Prepending to categorical pipeline.")
            semantic_params = adv_features['semantic_grouping']
            grouper = SemanticCategoricalGrouper(
                embedding_model_name=semantic_params['embedding_model'],
                min_cluster_size=semantic_params['min_cluster_size']
            )
            cat_steps.insert(0, ("semantic_grouping", grouper))
            
        return Pipeline(steps=cat_steps)
    
    def _build_high_cardinality_transformer(self) -> Pipeline:
        """Builds the pipeline for cardinality categorical features"""
        encoder_name = settings['pipeline_params']['high_cardinality_encoder']
        if encoder_name == 'target':
            encoder = TargetEncoderWrapper()
        else:
            raise NotImplementedError(f"Encoder '{encoder_name}' is not yet implemented.")
            
        imputer = SimpleImputer(strategy='most_frequent')
        imputer.set_output(transform="pandas")
        return Pipeline(steps=[
            ('imputer', imputer),
            ('encoder', encoder)
        ])


    def _build_preprocessor(self) -> ColumnTransformer:
        """Assembles the main ColumnTransformer from various sub-pipelines."""
        transformers = []
        if self.column_roles['numeric']:
            transformers.append(('numeric', self._build_numeric_transformer(), self.column_roles['numeric']))
        if self.column_roles['skewed']:
            transformers.append(('skewed', self._build_skewed_transformer(), self.column_roles['skewed']))
        if self.column_roles['categorical']:
            transformers.append(('categorical', self._build_categorical_transformer(), self.column_roles['categorical']))
        if self.column_roles["high_cardinality_cat"]:
            transformers.append(("high_cardinality", self._build_high_cardinality_transformer(), self.column_roles["high_cardinality_cat"]))
        
        if not transformers:
            raise PipelineConstructionError("No columns were classified for preprocessing. Check data types and content.")

        return ColumnTransformer(transformers=transformers, remainder='drop')
            