"""
Data Quality & Validation Framework for DataInsight AI

This module provides a comprehensive DataQualityValidator class that performs
a series of checks on an input DataFrame to ensure its integrity before it
enters the main preprocessing pipeline. It produces a structured, human-readable
report of its findings.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pandas as pd
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ValidationCheck(BaseModel):
    """Represents the outcome of a single validation check"""
    name:str
    passed:bool
    message:str
    details:Optional[Dict[str, Any]] = None

@dataclass
class ValidationReport:
    """A collection of all validation checks performed on a dataset"""
    is_valid:bool = True
    checks:List[ValidationCheck] = field(default_factory=list)

    def add_check(self, check:ValidationCheck):
        self.checks.append(check)
        if not check.passed:
            self.is_valid = False
    
class DataQualityValidator:
    """Performs a suite of data quality checks on a pandas DataFrame"""
    def __init__(self, df:pd.DataFrame, expected_schema:Optional[Dict[str,str]] = None):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Validator requires a non-empty pandas DataFrame")
        self.df = df
        self.expected_schema = expected_schema or {}
    
    def validate(self) -> ValidationReport:
        """
        Executes all defined validation checks and returns a comprehensive report.

        Returns:
            A ValidationReport object summarizing the results.
        """
        logging.info("Validating data quality...")
        report = ValidationReport()
        report.add_check(self._check_for_duplicates())
        report.add_check(self._check_for_high_missing_values())
        report.add_check(self._check_for_mixed_types())
        report.add_check(self._check_schema_compliance())
        report.add_check(self._check_outliers())
        logging.info(f"Validation complete. Status: {'Valid' if report.is_valid else 'Invalid'}")
        return report
    
    def _check_for_duplicates(self) -> ValidationCheck:
        """Checks for fully duplicate rows in the DataFrame."""
        num_duplicates = self.df.duplicated().sum()
        if num_duplicates > 0:
            return ValidationCheck(
                name="Duplicate Row Check",
                passed=False,
                message=f"Found {num_duplicates} duplicate rows in the dataset.",
                details={"duplicate_count": int(num_duplicates)}
            )
        return ValidationCheck(
            name="Duplicate Row Check",
            passed=True,
            message="No duplicate rows found."
        )

    def _check_for_high_missing_values(self) -> ValidationCheck:
        """Identifies columns with a high percentage of missing values."""
        missing_ratios = self.df.isnull().sum() / len(self.df)
        threshold = 0.95
        high_missing_cols = missing_ratios[missing_ratios >= threshold].to_dict()

        if high_missing_cols:
            return ValidationCheck(
                name="High Missing Values Check",
                passed=False,
                message=f"Found {len(high_missing_cols)} columns with >= {threshold:.0%} missing values.",
                details={col: f"{ratio:.2%}" for col, ratio in high_missing_cols.items()}
            )
        return ValidationCheck(
            name="High Missing Values Check",
            passed=True,
            message=f"No columns found with missing values above the {threshold:.0%} threshold."
        )

    def _check_for_mixed_types(self) -> ValidationCheck:
        """
        Detects columns with 'object' dtype that contain multiple underlying types.
        This is a common source of errors in data processing.
        """
        mixed_type_cols = {}
        object_cols = self.df.select_dtypes(include=['object']).columns

        for col in object_cols:
            unique_types = self.df[col].dropna().apply(type).nunique()
            if unique_types > 1:
                inferred_types = list(self.df[col].dropna().apply(type).unique())
                mixed_type_cols[col] = [t.__name__ for t in inferred_types]

        if mixed_type_cols:
            return ValidationCheck(
                name="Mixed Data Types Check",
                passed=False,
                message=f"Found {len(mixed_type_cols)} 'object' columns with mixed underlying data types.",
                details=mixed_type_cols
            )
        return ValidationCheck(
            name="Mixed Data Types Check",
            passed=True,
            message="No 'object' columns with mixed underlying types were detected."
        )
