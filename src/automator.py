"""
The Core Automation Engine for DataInsight AI

This module contains the `PipelineAutomator`, the central class responsible for
inspecting raw data and dynamically constructing a tailored preprocessing
pipeline
"""
import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import settings
from common.data_cleaning import SemanticCategoricalGrouper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
        self.schema:Dict = {}
        self.column_roles:Dict[str, List[str]] = {}
        self._profile_data()
        self._classify_columns()
        self._prepare_feature_lists()
    
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
        heuristic = settings["heuristics"]
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

    def _prepare_feature_list(self) -> None:
        """"Ensure the target column is not included in any feature transformation lists"""
        if self.target_column in self.df.columns:
            for role in self.column_roles:
                if self.target_column in self.column_roles[role]:
                    self.column_roles[role].remove(self.target_column)
                    logging.warning(f"Target column '{self.target_column}' removed from '{role}' transform list.")

    def _build_numeric_transformer(self) -> Pipeline:
        """Builds the pipeline for standard numeric features"""
        return Pipeline(steps=[("imputer", SimpleImputer(strategy=settings["pipeline_params"]["numeric_imputer_strategy"])),
                               ("scaler", StandardScaler())])
    def _build_skewed_transformer(self) -> Pipeline:
        """Builds the pipeline for skewed numeric features"""
        return Pipeline(steps=[("imputer", SimpleImputer(strategy=settings["pipeline_params"]["numeric_imputer_strategy"])),
                               ("power_transform", PowerTransformer(method="yeo-johnson")),
                               ("scaler", StandardScaler())])
    
    def _build_categorical_transformer(self) -> Pipeline:
        """Builds the pipeline for categorical features, with optional semantic grouping"""
        cat_steps = [("imputer", SimpleImputer(strategy=settings["pipeline_params"]["categorical_imputer_strategy"],
                                               fill_value="missing")),
                                               ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        if settings["advanced_features"]["use_semantic_categorical_grouping"]:
            logging.info("Semantic grouping enabled. Prepending to categorical pipeline")
            semantic_params = settings["advanced_features"].get("semantic_grouping, {}")
            grouper = SemanticCategoricalGrouper(embedding_model_name=semantic_params.get("embedding_model", "all-MiniLM-L6-v2"),
                                                 min_cluster_size=semantic_params.get("mini_cluster_size", 2))
            cat_steps.insert(0, ("semantic_grouping", grouper))
        return Pipeline(steps=cat_steps)
    
    def _build_high_cardinality_transformer(self) -> Pipeline:
        """Builds the pipeline for cardinality categorical features"""
        encoder_name = settings['pipeline_params']['high_cardinality_encoder']
        if encoder_name == 'target':
            encoder = TargetEncoder()
        else:
            # Placeholder for future encoders like HashingEncoder
            raise NotImplementedError(f"Encoder '{encoder_name}' is not yet implemented.")
            
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', encoder)
        ])


    def build(self) -> Tuple[Pipeline, Dict]:
        """
        The main public method that orchestrates the entire workflow.

        Returns:
            A tuple containing the fully configured scikit-learn pipeline and the
            column roles dictionary for the lineage report.
        """
        logging.info(f"Orchestrating workflow for task: '{self.task}'")
        # HOOK FOR PHASE 2: Feature Generation 
        # logging.info("[Future] Phase 2: Feature Generation step would be here.")

        # HOOK FOR PHASE 2: Feature Selection
        # logging.info("[Future] Phase 2: Feature Selection step would be here.")

        # PHASE 1: Build Core Preprocessor
        transformers: List[Tuple] = []
        if self.column_roles['numeric']:
            transformers.append(('numeric', self._build_numeric_transformer(), self.column_roles['numeric']))
        if self.column_roles['skewed']:
            transformers.append(('skewed', self._build_skewed_transformer(), self.column_roles['skewed']))
        if self.column_roles['categorical']:
            transformers.append(('categorical', self._build_categorical_transformer(), self.column_roles['categorical']))
        if self.column_roles["high_cardinality_cat"]:
            transformers.append(("high_cardinality", self._build_high_cardinality_transformer(),
                                 self.column_roles["high_cardinality_cat"]))  
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        # Task-specific augmentation
        if self.task == "classification":
            y = self.df[self.target_column]
            minority_ratio = y.value_counts(normalize=True).min()
            if minority_ratio < settings["pipeline_params"]["imbalance_smote_threshold"]:
                logging.info(f"Imbalance detected (ratio: {minority_ratio:.2f}). Adding SMOTE")
                final_pipeline = ImbPipeline([("preprocessor", preprocessor),
                                              ("resampler", SMOTE(random_state=42))])
            else:
                final_pipeline = Pipeline([("preprocessor", preprocessor)])
        elif self.task == "regression":
            final_pipeline = Pipeline([("preprocessor", preprocessor)])
        elif self.task == "clustering":
            logging.info("Building preprocessing pipeline for unsupervised clustering task")
            final_pipeline = Pipeline([("preprocessor", preprocessor)])
        # --- REPLACED TODO: Explicit error for unimplemented features ---
        elif self.task == 'timeseries':
            raise NotImplementedError("The 'timeseries' task is a planned feature and not yet implemented.")

        else:
            raise ValueError(f"Unsupported task: '{self.task}'")
        logging.info("Workflow orchestration complete. Pipeline is ready")
        return final_pipeline, self.column_roles
            