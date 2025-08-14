"""
Utility Functions for DataInsight AI

This module provides a collection of helper functions for common tasks such as
saving/loading machine learning artifacts and generating detailed lineage reports
for full reproducibility and auditability.
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from .common.data_cleaning import SemanticCategoricalGrouper

try:
    import ydata_profiling as pp
except ImportError:
    try:
        import pandas_profiling as pp
    except ImportError:
        pp = None
        logging.warning("ydata-profiling or pandas-profiling not found. EDA reports will be basic.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def save_pipeline(pipeline: Pipeline, file_path: Path) -> None:
    """
    Saves a scikit-learn pipeline to a file using joblib.

    Args:
        pipeline: The fitted scikit-learn pipeline object to save.
        file_path: The destination path (including filename) for the .joblib file.
    """
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, file_path)
        logging.info(f"Pipeline successfully saved to: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save pipeline to {file_path}: {e}")
        raise


def load_pipeline(file_path: Path) -> Pipeline:
    """
    Loads a scikit-learn pipeline from a joblib file.

    Args:
        file_path: The path to the .joblib file.

    Returns:
        The loaded scikit-learn pipeline object.
    """
    if not file_path.is_file():
        logging.error(f"Pipeline file not found at: {file_path}")
        raise FileNotFoundError(f"No pipeline file at the specified path: {file_path}")
    try:
        pipeline = joblib.load(file_path)
        logging.info(f"Pipeline successfully loaded from: {file_path}")
        return pipeline
    except Exception as e:
        logging.error(f"Failed to load pipeline from {file_path}: {e}")
        raise


def generate_lineage_report(
    column_roles: Dict[str, list],
    pipeline: Pipeline,
    config_snapshot: Dict[str, Any],
    task: str
) -> Dict[str, Any]:
    """
    Generates a detailed report on the preprocessing run for auditability.

    This function inspects the fitted pipeline to extract learned parameters
    and combines them with the column classifications and configuration settings.

    Args:
        column_roles: Dictionary mapping roles to column names.
        pipeline: The fitted scikit-learn pipeline.
        config_snapshot: A dictionary representation of the settings used for the run.
        task: The machine learning task performed (e.g., 'classification').

    Returns:
        A dictionary containing the detailed lineage report.
    """
    logging.info("Generating lineage report...")
    report = {
        "run_metadata": {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "task": task,
        },
        "column_processing_summary": column_roles,
        "pipeline_details": {},
        "configuration_snapshot": config_snapshot
    }

    if 'preprocessor' not in pipeline.named_steps:
        logging.warning("Preprocessor step not found in pipeline. Cannot detail steps.")
        return report

    preprocessor: ColumnTransformer = pipeline.named_steps['preprocessor']
    
    # Inspect each transformer within the ColumnTransformer
    for name, trans_pipeline, columns in preprocessor.transformers_:
        if not columns: 
            continue
        
        if not isinstance(trans_pipeline, Pipeline):
            logging.warning(f"Transformer '{name}' is not a Pipeline object, skipping detail extraction.")
            continue

        report["pipeline_details"][name] = {"columns": columns, "steps": {}}
        
        for step_name, step_obj in trans_pipeline.named_steps.items():
            step_details = {"class": step_obj.__class__.__name__}
            
            # Extract learned parameters based on the object type
            if isinstance(step_obj, SimpleImputer):
                step_details["strategy"] = step_obj.strategy
                step_details["learned_values"] = step_obj.statistics_.tolist()
            elif isinstance(step_obj, StandardScaler):
                step_details["learned_mean"] = step_obj.mean_.tolist()
                step_details["learned_scale"] = step_obj.scale_.tolist()
            elif isinstance(step_obj, PowerTransformer):
                step_details["learned_lambdas"] = step_obj.lambdas_.tolist()
            elif isinstance(step_obj, SemanticCategoricalGrouper):
                step_details["learned_mappings"] = step_obj.mappings_            
            report["pipeline_details"][name]["steps"][step_name] = step_details
            
    return report


def save_json_report(report_data: Dict, file_path: Path) -> None:
    """
    Saves a dictionary as a JSON file with pretty printing.

    Args:
        report_data: The dictionary to save.
        file_path: The destination path for the .json file.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        logging.info(f"JSON report successfully saved to: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON report to {file_path}: {e}")
        raise

def generate_eda_report(df: pd.DataFrame, title: str = "DataInsight AI - EDA Report") -> Dict[str, Any]:
    """Generate comprehensive EDA report for a DataFrame.
    
    Args:
        df: Input DataFrame to analyze
        title: Report title
        
    Returns:
        Dictionary containing EDA analysis results
    """
    logging.info("Generating EDA report...")
    
    # Basic statistics
    basic_stats = {
        "shape": df.shape,
        "memory_usage": df.memory_usage(deep=True).sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Numeric column analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_analysis = {}
    
    for col in numeric_cols:
        try:
            series = df[col].dropna()
            if len(series) > 0:
                numeric_analysis[col] = {
                    "count": len(series),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "25%": float(series.quantile(0.25)),
                    "50%": float(series.quantile(0.5)),
                    "75%": float(series.quantile(0.75)),
                    "max": float(series.max()),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                    "unique_count": int(series.nunique()),
                    "cardinality_ratio": float(series.nunique() / len(series))
                }
        except Exception as e:
            logging.warning(f"Could not analyze numeric column {col}: {e}")
    
    # Categorical column analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_analysis = {}
    
    for col in categorical_cols:
        try:
            series = df[col].dropna()
            if len(series) > 0:
                value_counts = series.value_counts()
                categorical_analysis[col] = {
                    "count": len(series),
                    "unique_count": int(series.nunique()),
                    "cardinality_ratio": float(series.nunique() / len(series)),
                    "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "top_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "top_10_values": value_counts.head(10).to_dict()
                }
        except Exception as e:
            logging.warning(f"Could not analyze categorical column {col}: {e}")
    
    # Correlation analysis for numeric columns
    correlation_matrix = None
    if len(numeric_cols) > 1:
        try:
            correlation_matrix = df[numeric_cols].corr().to_dict()
        except Exception as e:
            logging.warning(f"Could not compute correlation matrix: {e}")
    
    # Data quality issues
    quality_issues = []
    
    # Check for high missing values
    for col, missing_pct in basic_stats["missing_percentage"].items():
        if missing_pct > 50:
            quality_issues.append({
                "type": "high_missing_values",
                "column": col,
                "percentage": missing_pct,
                "severity": "high" if missing_pct > 80 else "medium"
            })
    
    # Check for high cardinality categorical columns
    for col, analysis in categorical_analysis.items():
        if analysis["cardinality_ratio"] > 0.9:
            quality_issues.append({
                "type": "high_cardinality_categorical",
                "column": col,
                "cardinality_ratio": analysis["cardinality_ratio"],
                "severity": "medium"
            })
    
    # Check for potential ID columns
    for col, analysis in numeric_analysis.items():
        if analysis["cardinality_ratio"] == 1.0:
            quality_issues.append({
                "type": "potential_id_column",
                "column": col,
                "severity": "low"
            })
    
    # Check for highly skewed columns
    for col, analysis in numeric_analysis.items():
        if abs(analysis["skewness"]) > 3:
            quality_issues.append({
                "type": "highly_skewed",
                "column": col,
                "skewness": analysis["skewness"],
                "severity": "medium"
            })
    
    # Compile final report
    eda_report = {
        "title": title,
        "generated_at": datetime.now().isoformat(),
        "basic_statistics": basic_stats,
        "numeric_analysis": numeric_analysis,
        "categorical_analysis": categorical_analysis,
        "correlation_matrix": correlation_matrix,
        "quality_issues": quality_issues,
        "recommendations": _generate_recommendations(quality_issues, basic_stats)
    }
    
    # Try to generate advanced profiling report if available
    if pp is not None:
        try:
            logging.info("Generating advanced profiling report...")
            profile = pp.ProfileReport(df, title=title, explorative=True)
            # Convert to dict format (simplified)
            eda_report["advanced_profile_available"] = True
            eda_report["profile_html"] = profile.to_html()
        except Exception as e:
            logging.warning(f"Could not generate advanced profile: {e}")
            eda_report["advanced_profile_available"] = False
    else:
        eda_report["advanced_profile_available"] = False
    
    logging.info("EDA report generation complete")
    return eda_report

def _generate_recommendations(quality_issues: list, basic_stats: dict) -> list:
    """Generate actionable recommendations based on EDA findings."""
    recommendations = []
    
    # Group issues by type
    issue_types = {}
    for issue in quality_issues:
        issue_type = issue["type"]
        if issue_type not in issue_types:
            issue_types[issue_type] = []
        issue_types[issue_type].append(issue)
    
    # Generate recommendations
    if "high_missing_values" in issue_types:
        high_missing = [issue for issue in issue_types["high_missing_values"] if issue["severity"] == "high"]
        if high_missing:
            recommendations.append({
                "type": "data_cleaning",
                "priority": "high",
                "action": "Consider dropping columns with >80% missing values",
                "affected_columns": [issue["column"] for issue in high_missing]
            })
    
    if "high_cardinality_categorical" in issue_types:
        recommendations.append({
            "type": "feature_engineering",
            "priority": "medium",
            "action": "Consider target encoding or grouping for high-cardinality categorical columns",
            "affected_columns": [issue["column"] for issue in issue_types["high_cardinality_categorical"]]
        })
    
    if "highly_skewed" in issue_types:
        recommendations.append({
            "type": "data_transformation",
            "priority": "medium",
            "action": "Apply log or power transformations to highly skewed numeric columns",
            "affected_columns": [issue["column"] for issue in issue_types["highly_skewed"]]
        })
    
    if "potential_id_column" in issue_types:
        recommendations.append({
            "type": "feature_selection",
            "priority": "low",
            "action": "Consider excluding ID columns from model training",
            "affected_columns": [issue["column"] for issue in issue_types["potential_id_column"]]
        })
    
    return recommendations