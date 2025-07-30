"""
Utility Functions for DataInsight AI

This module provides a collection of helper functions for common tasks such as
saving/loading machine learning artifacts and generating detailed lineage reports
for full reproducibility and auditability.
"""

import json
import logging
import joblib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from .common.data_cleaning import SemanticCategoricalGrouper
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