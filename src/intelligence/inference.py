"""Inference utilities for intelligent automation"""

from typing import Optional, List
import pandas as pd


def infer_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Infer the most likely target column for ML tasks.

    Heuristics:
    1. Exact match for common names ('target', 'label', 'class', 'y', 'price', 'salary', 'churn').
    2. Last column if it's numeric or categorical with low cardinality.
    3. Column with 'target' or 'label' in the name (case-insensitive).

    Args:
        df: Input DataFrame

    Returns:
        Name of the inferred target column or None
    """
    columns = df.columns.tolist()
    lower_cols = [c.lower() for c in columns]

    # 1. Common exact matches
    common_names = {"target", "label", "class", "y", "actual", "price", "revenue", "salary", "churn", "survived"}
    for name in common_names:
        if name in lower_cols:
            return columns[lower_cols.index(name)]

    # 2. 'target' or 'label' in name
    for i, col in enumerate(lower_cols):
        if "target" in col or "label" in col:
            return columns[i]

    # 3. Last column (common convention in datasets)
    last_col = columns[-1]
    # Check if it looks like a valid target (not an ID or high-cardinality string)
    if pd.api.types.is_numeric_dtype(df[last_col]):
        return last_col
    elif pd.api.types.is_object_dtype(df[last_col]) or pd.api.types.is_categorical_dtype(df[last_col]):
        if df[last_col].nunique() < 20:  # Likely classification label
            return last_col

    return None
