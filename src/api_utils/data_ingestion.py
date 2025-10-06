from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import logging
import uuid


def create_validation_issues(validation_report) -> list:
    return [
        {
            "type": check.name,
            "description": check.message,
            "severity": "high" if "missing" in check.message.lower() or "null" in check.message.lower() else "medium",
            "affected_columns": list(check.details.keys()) if check.details else [],
            "details": check.details,
            "passed": check.passed
        }
        for check in validation_report.checks if not check.passed
    ]


def create_intelligence_summary(intelligence_profile: dict) -> dict:
    column_profiles = intelligence_profile.get('column_profiles', {})
    domain_analysis = intelligence_profile.get('domain_analysis', {})

    semantic_types = {col: profile.semantic_type.value
                    for col, profile in column_profiles.items()}

    detected_domains = domain_analysis.get('detected_domains', [])
    primary_domain = detected_domains[0] if detected_domains else None

    return {
        "semantic_types": semantic_types,
        "primary_domain": primary_domain.get('domain') if primary_domain else 'unknown',
        "domain_confidence": primary_domain.get('confidence', 0) if primary_domain else 0,
        "domain_analysis": domain_analysis,
        "key_insights": intelligence_profile.get('overall_recommendations', [])[:3] if intelligence_profile.get('overall_recommendations') else [],
        "relationships_found": len(intelligence_profile.get('relationship_analysis', {}).get('relationships', [])),
        "profiling_completed": True
    }


def process_dataframe_ingestion(
    df: pd.DataFrame,
    session_id: Optional[str],
    source_info: Dict[str, Any],
    enable_profiling: bool,
    data_profiler,
    session_store: Dict[str, Any]
) -> Dict[str, Any]:
    from ..data_quality.validator import DataQualityValidator

    validator = DataQualityValidator(df)
    validation_report = validator.validate()

    session_id = session_id or str(uuid.uuid4())
    session_data = {
        "dataframe": df,
        "validation_report": validation_report,
        "created_at": datetime.now().isoformat(),
        **source_info
    }

    response_data = {
        "session_id": session_id,
        "status": "success",
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "validation": {
            "is_valid": validation_report.is_valid,
            "issues": create_validation_issues(validation_report)
        },
        **{k: v for k, v in source_info.items() if k not in ['dataframe', 'validation_report']}
    }

    if enable_profiling:
        try:
            intelligence_profile = data_profiler.profile_dataset(df)
            intelligence_summary = create_intelligence_summary(intelligence_profile)
            response_data["intelligence_summary"] = intelligence_summary
            session_data["intelligence_profile"] = intelligence_profile
        except Exception as prof_e:
            response_data["intelligence_summary"] = {
                "profiling_completed": False,
                "profiling_error": str(prof_e),
                "key_insights": [],
                "semantic_types": {},
                "primary_domain": "unknown",
                "domain_confidence": 0,
                "domain_analysis": {"detected_domains": []},
                "relationships_found": 0
            }

    session_store[session_id] = session_data

    return response_data


def load_dataframe_to_sandbox(df: pd.DataFrame, session_id: str, execute_python_in_sandbox) -> Dict[str, Any]:
    csv_data = df.to_csv(index=False)

    load_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

csv_data = '''{csv_data}'''
df = pd.read_csv(StringIO(csv_data))
print(f"Dataset loaded successfully!")
print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
print(f"Columns: {{list(df.columns)}}")
print("First 5 rows:")
print(df.head())
"""

    print(f"DEBUG: Loading data into sandbox for session {session_id}")
    load_result = execute_python_in_sandbox(load_code, session_id)

    if load_result["success"]:
        save_code = """
df.to_csv('dataset.csv', index=False)
df.to_csv('data.csv', index=False)
df.to_csv('ds.csv', index=False)
print("Dataset saved as dataset.csv, data.csv, and ds.csv")
"""
        save_result = execute_python_in_sandbox(save_code, session_id)
        print(f"DEBUG: Dataset save result: {save_result['success']}")

    return load_result
