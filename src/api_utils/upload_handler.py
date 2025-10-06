import logging
from typing import Dict, Any
import pandas as pd
from .helpers import convert_pandas_output_to_html


def enhance_with_agent_profile(df: pd.DataFrame, session_id: str, filename: str, response_data: Dict[str, Any]) -> None:
    try:
        from intelligence.hybrid_data_profiler import generate_dataset_profile_for_agent
        data_profile = generate_dataset_profile_for_agent(df, context={'filename': filename, 'upload_session': session_id})

        pii_findings = data_profile.profile_metadata.get('pii_detection')
        if pii_findings and pii_findings.privacy_score < 0.7:
            response_data["pii_detection"] = {
                "pii_detected": True,
                "risk_level": pii_findings.risk_level,
                "privacy_score": round(pii_findings.privacy_score, 2),
                "reidentification_risk": round(pii_findings.reidentification_risk, 2),
                "recommendations": pii_findings.recommendations,
                "requires_consent": True,
                "message": f"Privacy concerns detected (Risk Level: {pii_findings.risk_level}). Would you like to apply privacy protection before analysis?"
            }

        response_data["profiling_summary"] = {
            "quality_score": round(data_profile.quality_assessment.get('overall_score', 0), 1) if data_profile.quality_assessment.get('overall_score') else None,
            "anomalies_detected": data_profile.anomaly_detection.get('summary', {}).get('total_anomalies', 0),
            "profiling_time": round(data_profile.profile_metadata.get('profiling_duration', 0), 2)
        }

        import builtins
        builtins._session_store[session_id] = {
            'dataframe': df,
            'data_profile': data_profile,
            'filename': filename
        }

        store_in_knowledge_graph(session_id, filename, df, data_profile)

    except Exception as e:
        logging.warning(f"Failed to generate dataset profile: {e}")
        import builtins
        builtins._session_store[session_id] = {"dataframe": df}


def store_in_knowledge_graph(session_id: str, filename: str, df: pd.DataFrame, data_profile) -> None:
    try:
        from knowledge_graph.service import SessionDataStorage
        storage = SessionDataStorage()
        storage.add_session(session_id, {
            'dataset_info': {
                'filename': filename,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'domain': data_profile.dataset_insights.detected_domains
            }
        })
    except Exception as e:
        logging.warning(f"Failed to store in knowledge graph: {e}")


def load_data_to_agent_sandbox(
    df: pd.DataFrame,
    session_id: str,
    agent_sessions: Dict[str, Any],
    create_enhanced_agent_executor,
    response_data: Dict[str, Any]
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
print(f"Dataset loaded successfully!")
print(f"Shape: {{df.shape[0]}} rows x {{df.shape[1]}} columns")
print(f"Columns: {{list(df.columns)}}")
print("First 5 rows:")
print(df.head())
"""

        print(f"DEBUG: Loading data into sandbox for session {session_id}")
        load_result = execute_python_in_sandbox(load_code, session_id)

        if load_result["success"]:
            html_output = convert_pandas_output_to_html(load_result['stdout'])
            enhanced_message = f"\n\n{html_output}"

            profile_summary = response_data.get("profiling_summary", {})
            if profile_summary.get("quality_score"):
                enhanced_message += f"\n📊 Quality Score: {profile_summary['quality_score']}/100"
            if profile_summary.get("anomalies_detected"):
                enhanced_message += f"\n⚠️ Detected {profile_summary['anomalies_detected']} anomalies"
            if profile_summary.get("profiling_time"):
                enhanced_message += f"\n⏱️ Analysis completed in {profile_summary['profiling_time']}s"

            response_data["agent_analysis"] = enhanced_message
            response_data["agent_session_id"] = session_id

            save_code = """
df.to_csv('dataset.csv', index=False)
df.to_csv('data.csv', index=False)
df.to_csv('ds.csv', index=False)
print("Dataset saved as dataset.csv, data.csv, and ds.csv")
"""
            execute_python_in_sandbox(save_code, session_id)
        else:
            error_msg = load_result.get('stderr', 'Unknown error')
            response_data["agent_analysis"] = f"⚠️ Data loaded but with issues: {error_msg}"
            response_data["agent_session_id"] = session_id

    except Exception as agent_e:
        logging.warning(f"Agent data loading failed: {agent_e}")
        response_data["agent_analysis"] = None
