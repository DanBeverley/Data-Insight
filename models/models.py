"""
Pydantic Models for DataInsight AI Configuration

This module defines the strict, type-hinted data structures for our application's
configuration, ensuring that all settings loaded from `config.yml` are valid,
correctly typed, and easy to access with auto-completion.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Union, Optional

class PathsConfig(BaseModel):
    data_dir: str
    raw_data_dir: str
    processed_data_dir: str
    models_dir: str
    reports_dir: str

class HeuristicsConfig(BaseModel):
    missing_value_drop_threshold: float = Field(..., gt=0, le=1)
    id_cardinality_threshold: float = Field(..., ge=0, le=1)
    high_cardinality_threshold: float = Field(..., ge=0, le=1)
    skewed_feature_threshold: float = Field(..., ge=0)

class PipelineParamsConfig(BaseModel):
    numeric_imputer_strategy: str
    categorical_imputer_strategy: str
    high_cardinality_encoder: str
    hashing_n_features: int = Field(..., gt=0)
    imbalance_smote_threshold: float = Field(..., ge=0, le=0.5)

    @validator('numeric_imputer_strategy')
    def validate_numeric_strategy(cls, v):
        if v not in ['mean', 'median', 'most_frequent']:
            raise ValueError("Invalid numeric imputer strategy")
        return v

    @validator('high_cardinality_encoder')
    def validate_hc_encoder(cls, v):
        if v not in ['target', 'hashing']:
            raise ValueError("Unsupported high-cardinality encoder")
        return v

class SemanticGroupingConfig(BaseModel):
    embedding_model: str
    min_cluster_size: int = Field(..., gt=1)

class AdvancedFeaturesConfig(BaseModel):
    use_semantic_categorical_grouping: bool
    semantic_grouping: SemanticGroupingConfig

class DriftParamsConfig(BaseModel):
    psi_threshold: float = Field(..., gt=0)
    ks_p_value_threshold: float = Field(..., ge=0, le=1)

class FeatureGenerationConfig(BaseModel):
    enabled: bool = False
    dfs_max_depth: int = 2

class FeatureSelectionConfig(BaseModel):
    enabled: bool = False
    variance_threshold: float = Field(0.01, ge=0)
    correlation_threshold: float = Field(0.95, gt=0, lt=1)
    max_features_model_based: int = Field(50, gt=0)

class AppConfig(BaseModel):
    """The root configuration model for the entire application."""
    app_version: str
    paths: PathsConfig
    heuristics: HeuristicsConfig
    pipeline_params: PipelineParamsConfig
    advanced_features: AdvancedFeaturesConfig
    drift_params: DriftParamsConfig
    feature_generation: FeatureGenerationConfig
    feature_selection: FeatureSelectionConfig

    class Config:
        validate_assignment = True