"""Core Automation Engine for DataInsight AI

Orchestrates automated preprocessing with optional feature engineering/selection."""
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer

try:
    from category_encoders import TargetEncoder
    HAS_CATEGORY_ENCODERS = True
except ImportError:
    HAS_CATEGORY_ENCODERS = False
    print("Warning: category_encoders not installed. Some advanced encoding features will be disabled.")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    ImbPipeline = Pipeline  # Fallback to regular sklearn Pipeline
    print("Warning: imbalanced-learn not installed. SMOTE and advanced sampling will be disabled.")

from .config import settings
from .supervised.pipeline import create_supervised_pipeline
from .unsupervised.pipeline import create_unsupervised_pipeline
from .timeseries.pipeline import create_timeseries_pipeline, create_timeseries_config
from .nlp.pipeline import create_nlp_pipeline, create_nlp_config
from .utils import generate_lineage_report
from .common.data_cleaning import SemanticCategoricalGrouper
from .feature_generation.auto_fe import AutomatedFeatureEngineer, create_feature_engineering_config
from .feature_selector.intelligent_selector import IntelligentFeatureSelector
from .intelligence.data_profiler import IntelligentDataProfiler
from .intelligence.domain_detector import DomainDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    """Wrapper for category_encoders.TargetEncoder with get_feature_names_out support."""
    def __init__(self):
        if HAS_CATEGORY_ENCODERS:
            self.encoder = TargetEncoder()
        else:
            # Fallback to OneHotEncoder if category_encoders is not available
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names_in_ = None
        self.fallback_mode = not HAS_CATEGORY_ENCODERS

    def fit(self, X, y):
        if self.fallback_mode:
            self.encoder.fit(X)
        else:
            self.encoder.fit(X, y)
        self.feature_names_in_ = X.columns
        return self

    def transform(self, X):
        result = self.encoder.transform(X)
        if self.fallback_mode and hasattr(result, 'toarray'):
            result = result.toarray() if hasattr(result, 'toarray') else result
        return pd.DataFrame(result, index=X.index, columns=self.get_feature_names_out())

    def get_feature_names_out(self, input_features=None):
        if self.fallback_mode and hasattr(self.encoder, 'get_feature_names_out'):
            return list(self.encoder.get_feature_names_out(self.feature_names_in_))
        return list(self.feature_names_in_)

class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()

@dataclass
class OrchestratorResult:
    """Results container for workflow orchestration."""
    status:Status
    pipeline:Optional[Union[Pipeline, ImbPipeline]] = None
    processed_features:Optional[pd.DataFrame] = None
    aligned_target:Optional[pd.Series] = None
    lineage_report:Optional[Dict[str, Any]] = None
    error_message:Optional[str] = None
    column_roles:Dict[str, List[str]] = field(default_factory=dict)

class PipelineConstructionError(Exception):
    """Exception raised when pipeline construction fails."""
    pass

class WorkflowOrchestrator:
    """Orchestrates automated preprocessing with intelligent analysis."""
    def __init__(self, df:pd.DataFrame, target_column:str, task:str):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df:pd.DataFrame = df.copy()
        self.schema:Dict[str, Any] = {}
        self.column_roles:Dict[str, List[str]] = {}
        self.target_column:Optional[str] = target_column
        self.task:str = task
        self.profiler = IntelligentDataProfiler()
        self.domain_detector = DomainDetector()
    
    def build(self) -> tuple[Pipeline, Dict[str, List[str]]]:
        """Simplified build method for backward compatibility."""
        result = self.run()
        return result.pipeline, result.column_roles
    
    def run(self) -> OrchestratorResult:
        """Executes complete preprocessing workflow with optional Phase 2 capabilities."""
        try:
            logging.info(f"Starting orchestration for task: '{self.task}'")
            
            # Phase 2: Optional automated feature generation  
            if hasattr(settings, 'feature_generation') and settings.get('feature_generation', {}).get('enabled', False):
                logging.info("Starting automated feature generation")
                fe_config = create_feature_engineering_config(self.df, self.task)
                engineer = AutomatedFeatureEngineer(config=fe_config)
                
                # Apply feature engineering to single table
                enhanced_df = engineer.generate_features({"main": self.df})
                logging.info(f"Generated {len(enhanced_df.columns) - len(self.df.columns)} new features")
                self.df = enhanced_df
            
            profile_result = self.profiler.profile_dataset(self.df)
            self.schema = profile_result['column_profiles']
            domain_info = self.domain_detector.detect_domain(self.df)
            
            self.column_roles = self._classify_columns_intelligent()
            self._prepare_feature_lists()

            # Dynamic pipeline construction based on intelligent analysis
            pipeline = self._build_pipeline_dynamically(self.task)
            
            X = self.df.drop(columns=[self.target_column]) if self.target_column else self.df
            y = self.df[self.target_column] if self.target_column and self.task in ["classification", "regression"] else None
            
            preprocessor_step = pipeline.named_steps['preprocessor']
            if y is not None:
                preprocessor_step.fit(X, y)
            else:
                preprocessor_step.fit(X)
            
            X_transformed = preprocessor_step.transform(X)
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            processed_features = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
            y_aligned = self.df.loc[processed_features.index, self.target_column] if self.target_column else None

            # Phase 2: Optional intelligent feature selection
            if (hasattr(settings, 'feature_selection') and 
                settings.get('feature_selection', {}).get('enabled', False) and 
                self.task in ['classification', 'regression'] and y_aligned is not None):
                
                logging.info("Starting intelligent feature selection")
                selector = IntelligentFeatureSelector(task=self.task, config=settings['feature_selection'])
                
                nan_mask = processed_features.isnull().any(axis=1)
                if nan_mask.any():
                    logging.warning(f"Dropping {nan_mask.sum()} rows with NaNs before feature selection")
                    X_clean = processed_features.dropna()
                    y_clean = y_aligned.loc[X_clean.index]
                else:
                    X_clean = processed_features
                    y_clean = y_aligned
                
                processed_features = selector.select_features(X_clean, y_clean)
                y_aligned = y_clean.loc[processed_features.index]
                logging.info(f"Feature selection complete. Final shape: {processed_features.shape}")
            
            elif processed_features.isnull().any().any():
                nan_rows_mask = processed_features.isnull().any(axis=1)
                logging.warning(f"Dropping {nan_rows_mask.sum()} rows with NaNs after transformation")
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
    
    def _classify_columns_intelligent(self) -> Dict[str, List[str]]:
        """Enhanced column classification with semantic understanding."""
        logging.info("Intelligent column classification...")
        
        roles = {
            "to_drop": [],
            "numeric": [],
            "skewed": [],
            "categorical": [],
            "high_cardinality_cat": [],
            "categorical_binary": [],
            "temporal": [],
            "text": [],
            "id": []
        }
        
        heuristics = settings['heuristics']
        feature_columns = [col for col in self.df.columns if col != self.target_column]
        
        for col in feature_columns:
            profile = self.schema[col]
            classification = self._classify_single_column(col, self.df[col], profile, heuristics)
            roles[classification].append(col)
        
        # Merge binary categorical with regular categorical for pipeline compatibility
        roles["categorical"].extend(roles["categorical_binary"])
        roles["to_drop"].extend(roles["id"])
        
        logging.info(f"Intelligent classification: {[f'{k}: {len(v)}' for k, v in roles.items() if v]}")
        return roles
    
    def _classify_single_column(self, col_name: str, series: pd.Series, profile, heuristics: Dict) -> str:
        """Classify individual column with enhanced intelligence."""
        col_lower = col_name.lower()
        
        # Extract missing percentage from series if profile doesn't have it
        missing_percent = getattr(profile, 'missing_percent', series.isnull().sum() / len(series))
        
        if missing_percent > heuristics["missing_value_drop_threshold"]:
            return "to_drop"
        
        # Use semantic type from intelligent profiler if available
        if hasattr(profile, 'semantic_type'):
            semantic_type = profile.semantic_type.value
            if semantic_type in ['primary_key', 'foreign_key', 'natural_key']:
                return "id"
            elif semantic_type in ['datetime_timestamp', 'datetime_date', 'datetime_partial']:
                return "temporal"
            elif semantic_type in ['text_short', 'text_long']:
                return "text"
        
        # Fallback to traditional classification
        if self._is_id_column(col_lower, {'cardinality': series.nunique()/len(series), 'unique_count': series.nunique()}):
            return "id"
        
        if self._is_temporal_column(col_lower, series):
            return "temporal"
        
        if self._is_text_column(series, {}):
            return "text"
        
        if pd.api.types.is_numeric_dtype(series):
            skewness = series.skew() if len(series.dropna()) > 0 else 0.0
            return self._classify_numeric_intelligent(col_lower, series, {'skewness': skewness}, heuristics)
        
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            cardinality = series.nunique() / len(series)
            unique_count = series.nunique()
            return self._classify_categorical_intelligent(series, {'cardinality': cardinality, 'unique_count': unique_count}, heuristics)
        
        return "to_drop"
    
    def _is_id_column(self, col_name: str, profile: Dict) -> bool:
        """Enhanced ID detection with semantic patterns."""
        id_patterns = ['id', 'key', 'uuid', 'identifier', '_id', 'pk']
        semantic_match = any(pattern in col_name for pattern in id_patterns)
        high_cardinality = profile["cardinality"] > 0.95
        return semantic_match or (high_cardinality and profile["unique_count"] > 100)
    
    def _is_temporal_column(self, col_name: str, series: pd.Series) -> bool:
        """Detect temporal columns."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        date_patterns = ['date', 'time', 'timestamp', 'created', 'updated', 'at']
        semantic_match = any(pattern in col_name for pattern in date_patterns)
        
        if semantic_match and pd.api.types.is_object_dtype(series):
            try:
                sample = series.dropna().head(10)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    return True
            except:
                pass
        
        return False
    
    def _is_text_column(self, series: pd.Series, profile: Dict) -> bool:
        """Detect text columns based on content analysis."""
        if not pd.api.types.is_object_dtype(series):
            return False
        
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False
        
        avg_length = sample.astype(str).str.len().mean()
        return avg_length > 20
    
    def _classify_numeric_intelligent(self, col_name: str, series: pd.Series, profile: Dict, heuristics: Dict) -> str:
        """Intelligent numeric classification."""
        if abs(profile["skewness"]) >= heuristics["skewed_feature_threshold"]:
            return "skewed"
        return "numeric"
    
    def _classify_categorical_intelligent(self, series: pd.Series, profile: Dict, heuristics: Dict) -> str:
        """Intelligent categorical classification."""
        unique_count = profile["unique_count"]
        
        if unique_count == 2:
            return "categorical_binary"
        
        if profile["cardinality"] > heuristics["high_cardinality_threshold"]:
            return "high_cardinality_cat"
        
        return "categorical"

    def _prepare_feature_lists(self) -> None:
        if self.target_column in self.df.columns:
            transform_roles = ["numeric", "skewed", "categorical", "high_cardinality_cat"]
            for role in transform_roles:
                if self.target_column in self.column_roles[role]:
                    self.column_roles[role].remove(self.target_column)
                    logging.warning(f"Target column '{self.target_column}' removed from '{role}' transform list.")

    def _build_numeric_transformer(self) -> Pipeline:
        """Creates transformer for standard numeric features."""
        imputer = SimpleImputer(strategy=settings['pipeline_params']['numeric_imputer_strategy'])
        imputer.set_output(transform="pandas")
        return Pipeline(steps=[
            ("imputer", imputer),
            ("scaler", StandardScaler())
        ])
    def _build_skewed_transformer(self) -> Pipeline:
        """Creates transformer for skewed numeric features."""
        imputer = SimpleImputer(strategy=settings['pipeline_params']['numeric_imputer_strategy'])
        imputer.set_output(transform="pandas")
        return Pipeline(steps=[
            ("imputer", imputer),
            ("power_transform", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler())
        ])
    
    def _build_categorical_transformer(self) -> Pipeline:
        """Creates transformer for categorical features with optional semantic grouping."""
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
        """Creates transformer for high-cardinality categorical features."""
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
        """Dynamic preprocessor construction based on intelligent classification."""
        transformers = []
        
        # Add transformers based on intelligent classification
        if self.column_roles.get('numeric'):
            transformers.append(('numeric', self._build_numeric_transformer(), self.column_roles['numeric']))
            
        if self.column_roles.get('skewed'):
            transformers.append(('skewed', self._build_skewed_transformer(), self.column_roles['skewed']))
            
        if self.column_roles.get('categorical'):
            transformers.append(('categorical', self._build_categorical_transformer(), self.column_roles['categorical']))
            
        if self.column_roles.get('high_cardinality_cat'):
            transformers.append(('high_cardinality', self._build_high_cardinality_transformer(), self.column_roles['high_cardinality_cat']))
            
        # Handle temporal columns with specialized transformers
        if self.column_roles.get('temporal'):
            transformers.append(('temporal', self._build_temporal_transformer(), self.column_roles['temporal']))
            
        # Handle text columns  
        if self.column_roles.get('text'):
            transformers.append(('text', self._build_text_transformer(), self.column_roles['text']))
        
        if not transformers:
            raise PipelineConstructionError("No columns classified for preprocessing")

        return ColumnTransformer(transformers=transformers, remainder='drop')
    
    def _build_temporal_transformer(self) -> Pipeline:
        """Build transformer for temporal features."""
        from .timeseries.feature_engineering import DateTimeFeatureTransformer
        
        return Pipeline(steps=[
            ('datetime_features', DateTimeFeatureTransformer(
                features=['year', 'month', 'day', 'dayofweek', 'quarter', 'is_weekend']
            ))
        ])
    
    def _build_text_transformer(self) -> Pipeline:
        """Build transformer for text features."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        
        # Simple text processing pipeline
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='')),
            ('tfidf', TfidfVectorizer(max_features=100, stop_words='english', lowercase=True))
        ])
        
    def _build_pipeline_dynamically(self, task_type: str) -> Pipeline:
        """Build complete pipeline with task-specific optimizations."""
        preprocessor = self._build_preprocessor()
        
        # Task-specific pipeline enhancements
        if task_type in ['classification', 'regression']:
            return self._build_supervised_pipeline(preprocessor, task_type)
        elif task_type == 'clustering':
            return self._build_clustering_pipeline(preprocessor)
        elif task_type == 'timeseries':
            return self._build_timeseries_pipeline(preprocessor)
        elif task_type == 'nlp':
            return self._build_nlp_pipeline(preprocessor)
        else:
            # Default preprocessing-only pipeline
            return Pipeline([('preprocessor', preprocessor)])
    
    def _build_supervised_pipeline(self, preprocessor: ColumnTransformer, task_type: str) -> Pipeline:
        """Build supervised learning pipeline with intelligent enhancements."""
        y = self.df[self.target_column] if self.target_column else None
        
        # Check for class imbalance in classification
        if task_type == 'classification' and y is not None:
            return create_supervised_pipeline(preprocessor, task_type, y)
        else:
            return create_supervised_pipeline(preprocessor, task_type)
    
    def _build_clustering_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Build clustering pipeline with optimal parameters."""
        # Estimate optimal number of clusters based on data size
        n_samples = len(self.df)
        optimal_k = min(max(2, int(np.sqrt(n_samples/2))), 10)
        
        params = {"n_clusters": optimal_k, "n_init": 10, "random_state": 42}
        return create_unsupervised_pipeline(preprocessor, "kmeans", params)
    
    def _build_timeseries_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Build time series pipeline with intelligent feature engineering."""
        ts_config = create_timeseries_config(self.df, self.target_column, auto_detect=True)
        return create_timeseries_pipeline(self.df, self.target_column, config=ts_config)
    
    def _build_nlp_pipeline(self, preprocessor: ColumnTransformer) -> Pipeline:
        """Build NLP pipeline with text-aware processing."""
        nlp_config = create_nlp_config(self.df, self.target_column, auto_detect=True)
        return create_nlp_pipeline(self.df, self.target_column, config=nlp_config)