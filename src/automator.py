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

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import settings
from supervised.pipeline import create_supervised_pipeline
from unsupervised.pipeline import create_unsupervised_pipeline
from utils import generate_lineage_report
from common.data_cleaning import SemanticCategoricalGrouper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()

@dataclass
class OrchestratorResult:
    """A structured object to hold the results of an orchestration run."""
    status:Status
    pipeline:Optional[Union[Pipeline, ImbPipeline]] = None
    processed_data:Optional[pd.DataFrame] = None
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
                raise NotImplementedError(f"Task '{self.task}' is not implemented")
            
            X = self.df.drop(columns=[self.target_column]) if self.target_column else self.df
            y = self.df[self.target_column] if self.target_column and self.task in ["classification", "regression"] else None
            if y is not None:
                X_transformed = pipeline.fit_transform(X,y)
            else:
                X_transformed = pipeline.fit_transform(X)
            
            # Reconstruct DataFrame with proper column names
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            processed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
            # Drop NaN rows created by time-series features 
            nan_rows_mask = processed_df.isnull().any(axis=1)
            if nan_rows_mask.any():
                logging.warning(f"Dropping {nan_rows_mask.sum()} rows with NaNs after transformation.")
                processed_df = processed_df[~nan_rows_mask]
                if y is not None:
                    y = y[~nan_rows_mask]
            
            if y is not None:
                 processed_df[self.target_column] = y.values

            # Generate lineage report
            lineage = generate_lineage_report(
                self.column_roles, pipeline, settings.config.dict(), self.task
            )

            logging.info("Workflow orchestration completed successfully.")
            return OrchestratorResult(
                status=Status.SUCCESS,
                pipeline=pipeline,
                processed_data=processed_df,
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
                                "skewness":series.skew() if pd.api.types.is_numeric_dtypes(series) else None}
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
        heuristic = settings.heuristics
        for col, profile in self.schema.items():
            if profile["missing_percent"] > heuristic["missing_value_drop_threshold"]:
                roles["to_drop"].append(col)
            elif profile["cardinality"] == heuristic["id_cardinality_threshold"]:
                roles["id"].append(col)
            elif pd.api.types.is_numeric_dtype(profile["dtype"]):
                if abs(profile["skewness"]) > heuristic["skewed_feature_threshold"]:
                    roles["skewed"].append(col)
                else:
                    roles["numeric"].append(col)
            elif pd.api.types.is_object_dtype(profile["dtype"]) or pd.api.types.is_categorical_dtype(profile["dtype"]):
                if profile["cardinality"] > heuristic["high_cardinality_threshold"]:
                    roles["high_cardinality_cat"].append(col)
                else:
                    roles['categorical'].append(col)
        
        roles['to_drop'].extend(roles['id'])
        self.column_roles = roles
        logging.info(f"Column roles classified: { {k: len(v) for k, v in roles.items()} }")

    def _prepare_feature_lists(self) -> None:
        """"Ensure the target column is not included in any feature transformation lists"""
        if self.target_column in self.df.columns:
            for role in self.column_roles:
                if self.target_column in self.column_roles[role]:
                    self.column_roles[role].remove(self.target_column)
                    logging.warning(f"Target column '{self.target_column}' removed from '{role}' transform list.")

    def _build_numeric_transformer(self) -> Pipeline:
        """Builds the pipeline for standard numeric features"""
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=settings.pipeline_params.numeric_imputer_strategy)),
            ("scaler", StandardScaler())
        ])
    def _build_skewed_transformer(self) -> Pipeline:
        """Builds the pipeline for skewed numeric features"""
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=settings.pipeline_params.numeric_imputer_strategy)),
            ("power_transform", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler())
        ])
    
    def _build_categorical_transformer(self) -> Pipeline:
        """Builds the pipeline for categorical features, with optional semantic grouping"""
        params = settings.pipeline_params
        adv_features = settings.advanced_features

        cat_steps = [
            ("imputer", SimpleImputer(strategy=params.categorical_imputer_strategy, fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
        
        if adv_features.use_semantic_categorical_grouping:
            logging.info("Semantic grouping enabled. Prepending to categorical pipeline.")
            semantic_params = adv_features.semantic_grouping
            grouper = SemanticCategoricalGrouper(
                embedding_model_name=semantic_params.embedding_model,
                min_cluster_size=semantic_params.min_cluster_size
            )
            cat_steps.insert(0, ("semantic_grouping", grouper))
            
        return Pipeline(steps=cat_steps)
    
    def _build_high_cardinality_transformer(self) -> Pipeline:
        """Builds the pipeline for cardinality categorical features"""
        encoder_name = settings.pipeline_params.high_cardinality_encoder
        if encoder_name == 'target':
            encoder = TargetEncoder()
        else:
            raise NotImplementedError(f"Encoder '{encoder_name}' is not yet implemented.")
            
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
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
            