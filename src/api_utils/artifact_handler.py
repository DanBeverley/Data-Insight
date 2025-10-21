import json
import tempfile
import os
import joblib
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from fastapi import HTTPException
from fastapi.responses import Response


def download_data_artifact(session_data: Dict[str, Any], session_id: str) -> Response:
    processed_data = None
    if "processed_data" in session_data:
        processed_data = session_data["processed_data"]
    elif "pipeline_result" in session_data:
        pipeline_result = session_data["pipeline_result"]
        if isinstance(pipeline_result, dict):
            processed_data = pipeline_result.get("final_data") or pipeline_result.get("processed_data")

    if processed_data is None:
        raise HTTPException(status_code=404, detail="Processed data not found")

    df = processed_data
    if session_data.get("aligned_target") is not None:
        df = df.copy()
        df["target"] = session_data["aligned_target"]

    csv_data = df.to_csv(index=False)
    return Response(
        content=csv_data.encode(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=processed_data_{session_id}.csv"},
    )


def download_pipeline_artifact(session_data: Dict[str, Any], session_id: str) -> Response:
    if "pipeline_result" not in session_data:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    pipeline_result = session_data["pipeline_result"]
    pipeline = None

    if isinstance(pipeline_result, dict):
        pipeline = pipeline_result.get("pipeline") or pipeline_result.get("final_pipeline")

    if pipeline is None:
        placeholder = {
            "session_id": session_id,
            "message": "Pipeline object not available",
            "pipeline_result_keys": (
                list(pipeline_result.keys()) if isinstance(pipeline_result, dict) else str(type(pipeline_result))
            ),
            "timestamp": datetime.now().isoformat(),
        }
        return Response(
            content=json.dumps(placeholder, indent=2).encode(),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=pipeline_info_{session_id}.json"},
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
            joblib.dump(pipeline, tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                pipeline_bytes = f.read()
            os.unlink(tmp_file.name)

        return Response(
            content=pipeline_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename=pipeline_{session_id}.joblib",
                "Content-Length": str(len(pipeline_bytes)),
            },
        )
    except Exception as e:
        pipeline_info = {
            "session_id": session_id,
            "pipeline_type": str(type(pipeline)),
            "pipeline_string": str(pipeline)[:1000],
            "serialization_error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
        return Response(
            content=json.dumps(pipeline_info, indent=2).encode(),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=pipeline_info_{session_id}.json"},
        )


def download_lineage_artifact(session_data: Dict[str, Any], session_id: str) -> Response:
    if "pipeline_result" not in session_data:
        raise HTTPException(status_code=404, detail="Lineage report not found")

    pipeline_result = session_data["pipeline_result"]
    lineage = None

    if isinstance(pipeline_result, dict):
        lineage = pipeline_result.get("lineage_report") or pipeline_result.get("lineage")

        if not lineage and "execution_summary" in pipeline_result:
            execution_summary = pipeline_result["execution_summary"]
            lineage = {
                "pipeline_execution": "completed",
                "timestamp": datetime.now().isoformat(),
                "total_stages": execution_summary.get("total_stages", 0),
                "successful_stages": execution_summary.get("successful_stages", 0),
                "execution_time": execution_summary.get("total_time", 0),
                "session_id": session_id,
                "stages_executed": list(pipeline_result.get("results", {}).keys()),
            }

        if not lineage:
            lineage = {
                "pipeline_execution": "completed",
                "timestamp": datetime.now().isoformat(),
                "steps_executed": list(pipeline_result.keys()),
                "session_id": session_id,
            }

    lineage_json = json.dumps(lineage, indent=2, default=str)
    return Response(
        content=lineage_json.encode(),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=lineage_report_{session_id}.json"},
    )


def download_intelligence_artifact(session_data: Dict[str, Any], session_id: str) -> Response:
    if "intelligence_profile" not in session_data:
        raise HTTPException(status_code=404, detail="Intelligence profile not found")

    intelligence_profile = session_data["intelligence_profile"]
    serializable_profile = {}

    if "column_profiles" in intelligence_profile:
        serializable_profile["column_profiles"] = {}
        for col, profile in intelligence_profile["column_profiles"].items():
            serializable_profile["column_profiles"][col] = {
                "semantic_type": profile.semantic_type.value,
                "confidence": profile.confidence,
                "evidence": profile.evidence,
                "recommendations": profile.recommendations,
            }

    for key in ["domain_analysis", "relationship_analysis", "overall_recommendations"]:
        if key in intelligence_profile:
            serializable_profile[key] = intelligence_profile[key]

    intelligence_json = json.dumps(serializable_profile, indent=2, default=str)
    return Response(
        content=intelligence_json.encode(),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=intelligence_report_{session_id}.json"},
    )


def download_pipeline_metadata(session_data: Dict[str, Any], session_id: str) -> Response:
    if "pipeline_result" not in session_data:
        raise HTTPException(status_code=404, detail="Pipeline metadata not found")

    pipeline_result = session_data["pipeline_result"]
    safe_metadata = {}

    if isinstance(pipeline_result, dict):
        for key, value in pipeline_result.items():
            if key in ["pipeline", "final_pipeline"]:
                safe_metadata[key] = f"<Pipeline object of type {type(value)}>"
            elif key in ["dataframe", "processed_data", "final_data", "enhanced_data"]:
                safe_metadata[key] = (
                    f"<DataFrame with shape {value.shape}>"
                    if hasattr(value, "shape")
                    else f"<Data object of type {type(value)}>"
                )
            elif hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, list, dict)):
                safe_metadata[key] = str(value)[:500]
            else:
                try:
                    json.dumps(value)
                    safe_metadata[key] = value
                except:
                    safe_metadata[key] = str(value)[:500]
    else:
        safe_metadata = {
            "pipeline_result_type": str(type(pipeline_result)),
            "pipeline_result_str": str(pipeline_result)[:1000],
        }

    safe_metadata["session_id"] = session_id
    safe_metadata["export_timestamp"] = datetime.now().isoformat()

    metadata_json = json.dumps(safe_metadata, indent=2, default=str)
    return Response(
        content=metadata_json.encode(),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=pipeline_metadata_{session_id}.json"},
    )


def download_enhanced_data(session_data: Dict[str, Any], session_id: str) -> Response:
    enhanced_df = session_data.get("enhanced_data")

    if not enhanced_df and "pipeline_result" in session_data:
        pipeline_result = session_data["pipeline_result"]
        if isinstance(pipeline_result, dict):
            enhanced_df = (
                pipeline_result.get("enhanced_data")
                or pipeline_result.get("final_data")
                or pipeline_result.get("processed_data")
            )

    if enhanced_df is None:
        enhanced_df = session_data.get("processed_data")

    if enhanced_df is None:
        raise HTTPException(status_code=404, detail="Enhanced data not found")

    csv_data = enhanced_df.to_csv(index=False)
    return Response(
        content=csv_data.encode(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=enhanced_data_{session_id}.csv"},
    )


ARTIFACT_HANDLERS = {
    "data": download_data_artifact,
    "pipeline": download_pipeline_artifact,
    "lineage": download_lineage_artifact,
    "intelligence": download_intelligence_artifact,
    "pipeline-metadata": download_pipeline_metadata,
    "enhanced-data": download_enhanced_data,
}


def handle_artifact_download(session_data: Dict[str, Any], session_id: str, artifact_type: str) -> Response:
    handler = ARTIFACT_HANDLERS.get(artifact_type)
    if not handler:
        raise HTTPException(status_code=400, detail="Invalid artifact type")
    return handler(session_data, session_id)
