import logging
from typing import Dict, Any, BinaryIO
import pandas as pd
from .helpers import convert_pandas_output_to_html


def enhance_with_agent_profile(df: pd.DataFrame, session_id: str, filename: str, response_data: Dict[str, Any]) -> None:
    try:
        from src.intelligence.hybrid_data_profiler import generate_dataset_profile_for_agent

        data_profile = generate_dataset_profile_for_agent(
            df, context={"filename": filename, "upload_session": session_id}
        )

        pii_findings = data_profile.profile_metadata.get("pii_detection")
        if pii_findings and pii_findings.detected_columns:
            num_sensitive_cols = len(pii_findings.detected_columns)
            max_sensitivity = max(pii_findings.detected_columns.values()) if pii_findings.detected_columns else 0
            avg_sensitivity = (
                sum(pii_findings.detected_columns.values()) / len(pii_findings.detected_columns)
                if pii_findings.detected_columns
                else 0
            )

            privacy_concern = (
                pii_findings.privacy_score < 0.7
                or pii_findings.reidentification_risk > 0.3
                or max_sensitivity > 0.5
                or (avg_sensitivity > 0.4 and num_sensitive_cols >= 3)
                or pii_findings.risk_level in ["high", "critical"]
            )

            if privacy_concern:
                response_data["pii_detection"] = {
                    "pii_detected": True,
                    "risk_level": pii_findings.risk_level,
                    "privacy_score": round(pii_findings.privacy_score, 2),
                    "reidentification_risk": round(pii_findings.reidentification_risk, 2),
                    "recommendations": pii_findings.recommendations,
                    "detected_columns": pii_findings.detected_columns,
                    "detected_types": pii_findings.detected_types or [],
                    "requires_consent": True,
                    "message": f"Sensitive data detected in {len(pii_findings.detected_columns)} column(s). Apply privacy protection?",
                }

        response_data["profiling_summary"] = {
            "quality_score": (
                round(data_profile.quality_assessment.get("overall_score", 0), 1)
                if data_profile.quality_assessment.get("overall_score")
                else None
            ),
            "anomalies_detected": data_profile.anomaly_detection.get("summary", {}).get("total_anomalies", 0),
            "profiling_time": round(data_profile.profile_metadata.get("profiling_duration", 0), 2),
        }

        from src.api_utils.session_management import session_data_manager
        from .session_persistence import session_data_store

        session_data = session_data_manager.get_session(session_id)
        if not session_data:
            session_data = session_data_manager.create_session(session_id)

        session_data.update({"dataframe": df, "data_profile": data_profile, "filename": filename})

        session_data_store.save_session_data(session_id, session_data)

        store_in_knowledge_graph(session_id, filename, df, data_profile)

    except Exception as e:
        logging.warning(f"Failed to generate dataset profile: {e}")
        from src.api_utils.session_management import session_data_manager
        from .session_persistence import session_data_store

        session_data = session_data_manager.get_session(session_id)
        if not session_data:
            session_data = session_data_manager.create_session(session_id)

        session_data.update({"dataframe": df})
        session_data_store.save_session_data(session_id, session_data)


def store_in_knowledge_graph(session_id: str, filename: str, df: pd.DataFrame, data_profile) -> None:
    try:
        from src.knowledge_graph.service import SessionDataStorage

        storage = SessionDataStorage()
        storage.add_session(
            session_id,
            {
                "dataset_info": {
                    "filename": filename,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "domain": data_profile.dataset_insights.detected_domains,
                }
            },
        )
    except Exception as e:
        logging.warning(f"Failed to store in knowledge graph: {e}")


def load_data_to_agent_sandbox(
    df: pd.DataFrame,
    session_id: str,
    agent_sessions: Dict[str, Any],
    create_enhanced_agent_executor,
    response_data: Dict[str, Any],
) -> None:
    try:
        from data_scientist_chatbot.app.tools import execute_python_in_sandbox

        if session_id not in agent_sessions:
            agent_sessions[session_id] = create_enhanced_agent_executor(session_id)

        csv_data = df.to_csv(index=False)

        load_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

csv_data = '''{csv_data}'''
df = pd.read_csv(StringIO(csv_data))
print("Dataset loaded successfully!")
"""

        print(f"DEBUG: Loading data into sandbox for session {session_id}")
        load_result = execute_python_in_sandbox(load_code, session_id)

        if load_result["success"]:
            response_data["agent_analysis"] = None
            response_data["agent_session_id"] = session_id

            save_code = """
df.to_csv('dataset.csv', index=False)
df.to_csv('data.csv', index=False)
df.to_csv('ds.csv', index=False)
print("Dataset saved as dataset.csv, data.csv, and ds.csv")
"""
            execute_python_in_sandbox(save_code, session_id)
        else:
            error_msg = load_result.get("stderr", "Unknown error")
            raise Exception(f"Failed to load dataset into sandbox: {error_msg}")

    except Exception as agent_e:
        logging.error(f"Agent data loading failed: {agent_e}")
        raise


def validate_file_upload(file_content: BinaryIO, filename: str) -> bool:
    if not filename or not isinstance(filename, str):
        return False

    allowed_extensions = [".csv", ".xlsx", ".xls", ".json", ".parquet", ".txt"]
    file_ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

    if file_ext not in allowed_extensions:
        return False

    dangerous_extensions = [".php", ".exe", ".sh", ".bat", ".cmd", ".ps1", ".py", ".js", ".html", ".htm"]
    if any(ext in filename.lower() for ext in dangerous_extensions):
        return False

    try:
        content_sample = file_content.read(1024)
        file_content.seek(0)

        malicious_signatures = [b"<?php", b"<script", b"#!/bin/", b"#!python", b"MZ\x90\x00"]
        if any(sig in content_sample for sig in malicious_signatures):
            return False

    except Exception:
        return False

    return True


def handle_upload(df: pd.DataFrame, session_id: str, filename: str = "dataset.csv") -> Dict[str, Any]:
    from src.api_utils.session_management import session_data_manager

    # Ensure session exists
    if not session_data_manager.get_session(session_id):
        session_data_manager.create_session(session_id)

    response_data = {
        "status": "success",
        "message": "Dataset uploaded successfully",
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
    }

    load_data_to_agent_sandbox(df, session_id, {}, lambda sid: None, response_data)

    return response_data
