from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime
from typing import Optional
import pandas as pd
from pathlib import Path
import zipfile
import tempfile
import os

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/{session_id}/preview")
async def get_data_preview(session_id: str, rows: int = 10):
    from ..api import session_store

    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    if "dataframe" not in session_store[session_id]:
        return {"data": [], "shape": [0, 0], "columns": [], "message": "No dataset uploaded"}

    df = session_store[session_id]["dataframe"]

    # Replace NaN/Infinity with None for JSON compliance
    df_clean = df.replace([float("inf"), float("-inf")], None).where(pd.notnull(df), None)

    preview_data = df_clean.head(rows).to_dict("records")

    return {"data": preview_data, "shape": df.shape, "columns": df.columns.tolist()}


@router.get("/{session_id}/uploads")
async def get_session_uploads(session_id: str):
    from ..api import session_store

    uploads = []
    seen_files = set()

    # 1. Check in-memory session store
    if session_id in session_store:
        session_data = session_store[session_id]
        datasets = session_data.get("datasets", {})
        for filename, df in datasets.items():
            uploads.append(
                {
                    "filename": filename,
                    "rows": len(df) if df is not None else 0,
                    "columns": len(df.columns) if df is not None else 0,
                }
            )
            seen_files.add(filename)

        if session_data.get("filename") and session_data["filename"] not in seen_files:
            df = session_data.get("dataframe")
            uploads.append(
                {
                    "filename": session_data["filename"],
                    "rows": len(df) if df is not None else 0,
                    "columns": len(df.columns) if df is not None else 0,
                }
            )
            seen_files.add(session_data["filename"])

    # 2. Check disk for persisted files (handle backend reboot)
    # Using relative path from project root (assumed CWD)
    try:
        upload_dir = Path(f"data/uploads/{session_id}")
        if upload_dir.exists():
            for file_path in upload_dir.glob("*"):
                if file_path.is_file() and file_path.name not in seen_files:
                    # Basic metadata since we can't load DF instantly
                    uploads.append(
                        {
                            "filename": file_path.name,
                            "rows": 0,  # Unknown without loading
                            "columns": 0,  # Unknown without loading
                        }
                    )
                    seen_files.add(file_path.name)
    except Exception as e:
        print(f"Error scanning upload dir: {e}")

    return {"files": uploads, "status": "success" if uploads else "no_files"}


@router.post("/{session_id}/eda")
async def generate_eda(session_id: str):
    from ..api import session_store

    if session_id not in session_store:
        return {"status": "error", "detail": "Session not found"}

    try:
        df = session_store[session_id]["dataframe"]

        eda_report = {
            "basic_info": {
                "shape": list(df.shape),
                "columns": df.columns.tolist(),
                "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
                "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
                "missing_percentage": {k: float(v) for k, v in (df.isnull().sum() / len(df) * 100).to_dict().items()},
            },
            "numeric_summary": {},
            "categorical_summary": {},
            "correlations": {},
        }

        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            desc = df[numeric_cols].describe()
            eda_report["numeric_summary"] = {
                col: {k: float(v) for k, v in desc[col].to_dict().items()} for col in numeric_cols
            }

            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                eda_report["correlations"] = {
                    col1: {
                        col2: float(corr_matrix.loc[col1, col2]) if not pd.isna(corr_matrix.loc[col1, col2]) else 0.0
                        for col2 in numeric_cols
                    }
                    for col1 in numeric_cols
                }

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns[:5]
        for col in categorical_cols:
            try:
                value_counts = df[col].value_counts().head()
            except TypeError:
                value_counts = df[col].astype(str).value_counts().head()
            eda_report["categorical_summary"][col] = {str(k): int(v) for k, v in value_counts.to_dict().items()}

        session_store[session_id]["eda_report"] = eda_report

        return {"status": "success", "report": eda_report, "message": "EDA report generated successfully"}

    except Exception as e:
        return {"status": "error", "detail": f"Error generating EDA: {str(e)}"}


@router.get("/{session_id}/download/{artifact_type}")
async def download_artifact(session_id: str, artifact_type: str):
    from ..api import session_store
    from ..api_utils.artifact_handler import handle_artifact_download

    if session_id not in session_store:
        return {"status": "error", "detail": "Session not found"}

    session_data = session_store[session_id]
    return handle_artifact_download(session_data, session_id, artifact_type)


@router.get("/{session_id}/profile")
async def get_intelligence_profile(session_id: str):
    from ..api import session_store, data_profiler
    import builtins

    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        df = session_store[session_id]["dataframe"]

        if "intelligence_profile" in session_store[session_id]:
            intelligence_profile = session_store[session_id]["intelligence_profile"]
        else:
            from src.api_utils.session_management import session_data_manager

            session_data = session_data_manager.get_session(session_id)
            if "data_profile" in session_data:
                data_profile = session_data["data_profile"]
                column_profiles = {}
                if hasattr(data_profile, "column_profiles"):
                    column_profiles = data_profile.column_profiles

                return {
                    "status": "success",
                    "column_profiles": column_profiles,
                    "dataset_summary": {"shape": df.shape, "columns": df.columns.tolist()},
                }

        intelligence_profile = data_profiler.profile_dataset(df)
        session_store[session_id]["intelligence_profile"] = intelligence_profile

        response_data = {}
        column_profiles = intelligence_profile.get("column_profiles", {})
        response_data["column_profiles"] = {
            col: {
                "semantic_type": profile.semantic_type.value,
                "confidence": profile.confidence,
                "evidence": profile.evidence,
                "recommendations": profile.recommendations,
            }
            for col, profile in column_profiles.items()
        }

        response_data["dataset_summary"] = {"shape": df.shape, "columns": df.columns.tolist()}

        return {
            "status": "success",
            "column_profiles": response_data["column_profiles"],
            "dataset_summary": response_data["dataset_summary"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling error: {str(e)}")


@router.post("/{session_id}/profile")
async def generate_intelligence_profile(session_id: str, request: dict):
    from ..api import session_store, data_profiler
    from ..api_utils.models import ProfilingRequest

    prof_request = ProfilingRequest(**request)

    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        df = session_store[session_id]["dataframe"]

        if "intelligence_profile" in session_store[session_id] and not prof_request.deep_analysis:
            intelligence_profile = session_store[session_id]["intelligence_profile"]
        else:
            intelligence_profile = data_profiler.profile_dataset(df)
            session_store[session_id]["intelligence_profile"] = intelligence_profile

        response_data = {}
        column_profiles = intelligence_profile.get("column_profiles", {})
        response_data["column_profiles"] = {
            col: {
                "semantic_type": profile.semantic_type.value,
                "confidence": profile.confidence,
                "evidence": profile.evidence,
                "recommendations": profile.recommendations,
            }
            for col, profile in column_profiles.items()
        }

        if prof_request.include_domain_detection:
            response_data["domain_analysis"] = intelligence_profile.get("domain_analysis", {})

        if prof_request.include_relationships:
            response_data["relationship_analysis"] = intelligence_profile.get("relationship_analysis", {})

        response_data["overall_recommendations"] = intelligence_profile.get("overall_recommendations", [])
        response_data["profiling_metadata"] = {
            "deep_analysis": prof_request.deep_analysis,
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "profile_timestamp": datetime.now().isoformat(),
        }

        return {"status": "success", "intelligence_profile": response_data, "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling error: {str(e)}")


@router.get("/{session_id}/feature-recommendations")
async def get_feature_recommendations(
    session_id: str,
    target_column: Optional[str] = None,
    max_recommendations: int = 10,
    priority_filter: Optional[str] = None,
):
    from ..api import session_store, data_profiler

    if session_id not in session_store:
        return {"status": "error", "detail": "Session not found"}

    try:
        df = session_store[session_id]["dataframe"]

        if "intelligence_profile" not in session_store[session_id]:
            intelligence_profile = data_profiler.profile_dataset(df)
            session_store[session_id]["intelligence_profile"] = intelligence_profile
        else:
            intelligence_profile = session_store[session_id]["intelligence_profile"]

        from ..intelligence.feature_intelligence import AdvancedFeatureIntelligence

        feature_intelligence = AdvancedFeatureIntelligence()

        fe_analysis = feature_intelligence.analyze_feature_engineering_opportunities(
            df, intelligence_profile, target_column
        )

        recommendations = fe_analysis.get("feature_engineering_recommendations", [])

        if priority_filter:
            recommendations = [rec for rec in recommendations if rec.priority == priority_filter]

        recommendations = recommendations[:max_recommendations]

        serializable_recommendations = [
            {
                "feature_type": rec.feature_type,
                "priority": rec.priority,
                "description": rec.description,
                "implementation": rec.implementation,
                "expected_benefit": rec.expected_benefit,
                "computational_cost": rec.computational_cost,
            }
            for rec in recommendations
        ]

        session_store[session_id]["feature_analysis"] = fe_analysis

        return {
            "status": "success",
            "recommendations": serializable_recommendations,
            "feature_selection_strategy": fe_analysis.get("feature_selection_strategy", {}),
            "scaling_strategy": fe_analysis.get("scaling_strategy", {}),
            "implementation_pipeline": fe_analysis.get("implementation_pipeline", []),
            "total_recommendations": len(serializable_recommendations),
            "recommendation_metadata": {
                "target_column": target_column,
                "priority_filter": priority_filter,
                "generated_at": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        return {"status": "error", "detail": f"Feature recommendation error: {str(e)}"}


@router.post("/{session_id}/apply-features")
async def apply_feature_recommendations(session_id: str, request: dict):
    from ..api import session_store
    from ..feature_generation.auto_fe import AutoFeatureEngineer

    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        df = session_store[session_id]["dataframe"]

        if "feature_analysis" not in session_store[session_id]:
            raise HTTPException(status_code=400, detail="No feature analysis found. Generate recommendations first.")

        feature_analysis = session_store[session_id]["feature_analysis"]
        auto_fe = AutoFeatureEngineer()
        enhanced_df = auto_fe.engineer_features(df, feature_analysis)

        session_store[session_id]["enhanced_data"] = enhanced_df

        impact_summary = {
            "original_features": len(df.columns),
            "engineered_features": len(enhanced_df.columns),
            "features_added": len(enhanced_df.columns) - len(df.columns),
            "feature_expansion_ratio": len(enhanced_df.columns) / len(df.columns),
            "memory_impact": enhanced_df.memory_usage().sum() / df.memory_usage().sum(),
        }

        return {
            "status": "success",
            "message": "Feature engineering applied successfully",
            "impact_summary": impact_summary,
            "enhanced_data_shape": enhanced_df.shape,
            "new_columns": [col for col in enhanced_df.columns if col not in df.columns],
            "artifacts": {"enhanced_data": f"/api/data/{session_id}/download/enhanced-data"},
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature application error: {str(e)}")


@router.get("/{session_id}/relationship-graph")
async def get_relationship_graph(session_id: str):
    from ..api import session_store

    if session_id not in session_store:
        return {"status": "error", "detail": "Session not found"}

    try:
        session_data = session_store[session_id]
        intelligence_profile = session_data.get("intelligence_profile")

        if not intelligence_profile:
            return {"status": "error", "detail": "Intelligence profile required. Run profiling first."}

        relationship_analysis = intelligence_profile.get("relationship_analysis", {})
        relationships = relationship_analysis.get("relationships", [])

        if not relationships:
            return {
                "status": "success",
                "message": "No relationships found in the data",
                "graph_data": {"nodes": [], "edges": [], "metadata": {"total_relationships": 0}},
            }

        from ..intelligence.relationship_discovery import RelationshipDiscovery

        relationship_discovery = RelationshipDiscovery()

        relationship_objects = []
        for rel_dict in relationships:

            class MockRelationship:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)

            relationship_objects.append(MockRelationship(**rel_dict))

        graph_data = relationship_discovery.generate_relationship_graph(relationship_objects)

        relationship_types = [rel.get("relationship_type", "unknown") for rel in relationships]
        type_counts = {}
        for rel_type in relationship_types:
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

        enhanced_graph = {
            "nodes": graph_data.get("nodes", []),
            "edges": graph_data.get("edges", []),
            "metadata": {
                "total_relationships": len(relationships),
                "relationship_type_counts": type_counts,
                "total_nodes": len(graph_data.get("nodes", [])),
                "graph_generated_at": datetime.now().isoformat(),
            },
            "visualization_config": {
                "node_size_range": [20, 60],
                "edge_width_range": [1, 5],
                "color_scheme": {
                    "primary_key": "#ff6b6b",
                    "foreign_key": "#4ecdc4",
                    "correlation": "#45b7d1",
                    "categorical": "#96ceb4",
                    "temporal": "#ffeaa7",
                    "default": "#ddd",
                },
            },
        }

        return {"status": "success", "graph_data": enhanced_graph}

    except Exception as e:
        return {"status": "error", "detail": f"Relationship graph error: {str(e)}"}


@router.get("/{session_id}/artifacts")
async def get_session_artifacts(session_id: str, category: Optional[str] = None):
    from ..api_utils.artifact_tracker import artifact_tracker, ArtifactCategory

    try:
        cat_filter = None
        if category:
            try:
                cat_filter = ArtifactCategory(category)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")

        artifacts_data = artifact_tracker.get_session_artifacts_with_urls(session_id, cat_filter)

        return {"status": "success", **artifacts_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching artifacts: {str(e)}")


@router.get("/{session_id}/artifacts/new")
async def get_new_artifacts(session_id: str, since: str):
    from ..api_utils.artifact_tracker import artifact_tracker
    import logging

    logger = logging.getLogger(__name__)

    try:
        logger.debug(f"Fetching new artifacts for session {session_id} since {since}")
        new_artifacts = artifact_tracker.get_new_artifacts(session_id, since)

        if new_artifacts is None:
            new_artifacts = []

        result = {"status": "success", "new_artifacts": new_artifacts, "count": len(new_artifacts)}
        if len(new_artifacts) > 0:
            logger.info(f"Successfully fetched {len(new_artifacts)} new artifacts")
        return result

    except Exception as e:
        logger.error(f"Error in get_new_artifacts endpoint: {str(e)}", exc_info=True)
        return {"status": "error", "new_artifacts": [], "count": 0, "error": str(e)}


@router.get("/{session_id}/artifacts/download/{artifact_id}")
async def download_artifact_file(session_id: str, artifact_id: str):
    from ..api_utils.artifact_tracker import artifact_tracker

    artifacts_data = artifact_tracker.get_session_artifacts(session_id)
    artifact = next((a for a in artifacts_data["artifacts"] if a["artifact_id"] == artifact_id), None)

    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    web_url = artifact["file_path"]
    if web_url.startswith("/static/"):
        relative_path = web_url[1:]
    else:
        relative_path = web_url

    file_path = Path(relative_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    media_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".csv": "text/csv",
        ".json": "application/json",
        ".pkl": "application/octet-stream",
        ".joblib": "application/octet-stream",
        ".h5": "application/octet-stream",
        ".pt": "application/octet-stream",
        ".pth": "application/octet-stream",
        ".zip": "application/zip",
        ".html": "text/html",
        ".pdf": "application/pdf",
    }

    media_type = media_type_map.get(file_path.suffix.lower(), "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=artifact["filename"],
        headers={"Content-Disposition": f"attachment; filename={artifact['filename']}"},
    )


@router.post("/{session_id}/artifacts/zip")
async def zip_selected_artifacts(session_id: str, request: dict):
    from ..api_utils.artifact_tracker import artifact_tracker

    artifact_ids = request.get("artifact_ids", [])
    description = request.get("description")

    if not artifact_ids:
        raise HTTPException(status_code=400, detail="No artifact IDs provided")

    artifacts_data = artifact_tracker.get_session_artifacts(session_id)
    all_artifacts = artifacts_data.get("artifacts", [])

    selected_artifacts = [a for a in all_artifacts if a["artifact_id"] in artifact_ids]

    if not selected_artifacts:
        raise HTTPException(status_code=404, detail="None of the specified artifacts were found")

    temp_dir = tempfile.mkdtemp()
    zip_filename = f"artifacts_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = Path(temp_dir) / zip_filename

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for artifact in selected_artifacts:
                file_path = Path(artifact["file_path"])

                if file_path.exists():
                    arcname = f"{artifact['category']}/{artifact['filename']}"
                    zipf.write(file_path, arcname=arcname)

        if not zip_path.exists() or zip_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Failed to create zip file")

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_filename,
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
            background=lambda: (
                os.unlink(zip_path) if os.path.exists(zip_path) else None,
                os.rmdir(temp_dir) if os.path.exists(temp_dir) else None,
            ),
        )

    except Exception as e:
        if zip_path.exists():
            os.unlink(zip_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        raise HTTPException(status_code=500, detail=f"Error creating zip file: {str(e)}")


@router.get("/{session_id}/artifacts/zip-all")
async def zip_all_artifacts(session_id: str, dataset: Optional[str] = None):
    from ..api_utils.artifact_tracker import artifact_tracker
    import shutil

    artifacts_data = artifact_tracker.get_session_artifacts(session_id)
    all_artifacts = artifacts_data.get("artifacts", [])

    if dataset:
        dataset_lower = dataset.lower()
        all_artifacts = [
            a
            for a in all_artifacts
            if a.get("source_dataset") == dataset
            or dataset_lower in a.get("filename", "").lower()
            or dataset_lower in a.get("description", "").lower()
        ]

    if not all_artifacts:
        detail = f"No artifacts found for dataset '{dataset}'" if dataset else "No artifacts found for this session"
        raise HTTPException(status_code=404, detail=detail)

    temp_dir = tempfile.mkdtemp()
    dataset_slug = f"_{dataset[:20]}" if dataset else ""
    zip_filename = f"artifacts{dataset_slug}_{session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = Path(temp_dir) / zip_filename

    files_added = 0
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for artifact in all_artifacts:
                file_path = Path(artifact.get("file_path", ""))

                if file_path.exists():
                    arcname = f"{artifact.get('category', 'misc')}/{artifact['filename']}"
                    zipf.write(file_path, arcname=arcname)
                    files_added += 1

        if files_added == 0:
            raise HTTPException(status_code=404, detail="No artifact files could be found on disk")

        if not zip_path.exists() or zip_path.stat().st_size == 0:
            raise HTTPException(status_code=500, detail="Failed to create zip file")

        def cleanup():
            try:
                if os.path.exists(str(zip_path)):
                    os.unlink(str(zip_path))
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass

        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_filename,
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"},
            background=cleanup,
        )

    except HTTPException:
        raise
    except Exception as e:
        if zip_path.exists():
            os.unlink(zip_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error creating zip file: {str(e)}")


@router.delete("/{session_id}/artifacts/{artifact_id}")
async def delete_artifact(session_id: str, artifact_id: str):
    from ..api_utils.artifact_tracker import artifact_tracker

    try:
        success = artifact_tracker.remove_artifact(session_id, artifact_id)

        if success:
            return {"status": "success", "message": "Artifact removed"}
        else:
            raise HTTPException(status_code=404, detail="Artifact not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing artifact: {str(e)}")


@router.get("/{session_id}/artifact/{filename:path}")
async def get_artifact_file(session_id: str, filename: str):
    from fastapi.responses import RedirectResponse, StreamingResponse
    from ..api_utils.artifact_tracker import artifact_tracker
    import io

    project_root = Path(__file__).parent.parent.parent
    local_path = project_root / "static" / "plots" / filename

    if local_path.exists():
        return FileResponse(local_path)

    try:
        artifacts_data = artifact_tracker.get_session_artifacts(session_id)
        artifacts = artifacts_data.get("artifacts", [])

        matching_artifact = None
        for artifact in artifacts:
            if artifact.get("filename") == filename:
                matching_artifact = artifact
                break

        if not matching_artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")

        if matching_artifact.get("presigned_url"):
            return RedirectResponse(url=matching_artifact["presigned_url"])

        if matching_artifact.get("blob_url"):
            return RedirectResponse(url=matching_artifact["blob_url"])

        blob_path = matching_artifact.get("blob_path")
        if blob_path:
            from src.storage.cloud_storage import get_cloud_storage

            storage = get_cloud_storage()
            if storage:
                presigned = storage.get_blob_url(blob_path, expires_in=86400)
                return RedirectResponse(url=presigned)

        raise HTTPException(status_code=404, detail="Artifact not available locally or in cloud")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching artifact: {str(e)}")


@router.get("/{session_id}/insights")
async def get_session_insights(session_id: str):
    """
    Get intelligent insights for a session using hybrid data profiler.
    No rigid pattern matching - uses ML-based semantic understanding.
    """
    from ..api import session_store
    from ..intelligence.hybrid_data_profiler import generate_dataset_profile_for_agent
    import logging

    logger = logging.getLogger(__name__)

    try:
        if session_id not in session_store:
            return {"status": "success", "insights": [], "message": "No data available for this session"}

        session_data = session_store[session_id]
        df = session_data.get("dataframe")
        data_profile = session_data.get("data_profile")

        if df is None:
            return {"status": "success", "insights": [], "message": "No dataset uploaded yet"}

        import builtins

        agent_insights = []
        if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
            agent_insights = builtins._session_store[session_id].get("agent_insights", [])
            print(f"DEBUG: get_session_insights found {len(agent_insights)} agent insights")

        system_insights = []
        if data_profile:
            try:
                summary = data_profile.dataset_insights
                system_insights.append(
                    {
                        "label": "Dataset Size",
                        "value": f"{summary.total_records:,} rows × {summary.total_features} cols",
                        "type": "summary",
                        "source": "System",
                    }
                )

                if summary.missing_data_percentage > 0:
                    system_insights.append(
                        {
                            "label": "Data Quality",
                            "value": f"{summary.missing_data_percentage}% missing data",
                            "type": "warning",
                            "source": "System",
                        }
                    )
                else:
                    system_insights.append(
                        {
                            "label": "Data Quality",
                            "value": "No missing values detected",
                            "type": "success",
                            "source": "System",
                        }
                    )

                # Memory usage - NOT available directly in DatasetInsights
                # system_insights.append({
                #     "label": "Memory Usage",
                #     "value": f"{summary.memory_usage_mb:.2f} MB",
                #     "type": "info",
                #     "source": "System"
                # })
            except Exception as e:
                logger.warning(f"Failed to extract system insights: {e}")

        combined_insights = agent_insights + system_insights

        if not combined_insights:
            combined_insights = [
                {"label": "Status", "value": "Waiting for analysis...", "type": "info", "source": "System"}
            ]

        return {
            "status": "success",
            "insights": combined_insights,
            "session_id": session_id,
            "source": "tiered_insight_system",
        }

    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/artifact/{filename}/code")
async def get_artifact_code(session_id: str, filename: str):
    from src.api_utils.code_registry import get_code_registry

    code_registry = get_code_registry()
    code_entry = code_registry.get_code_for_artifact(session_id, filename)

    if not code_entry:
        raise HTTPException(status_code=404, detail="Code not found for this artifact")

    return {"code": code_entry.get("code", ""), "description": code_entry.get("description", "")}


@router.get("/{session_id}/artifact/{filename}/export/{format}")
async def export_artifact(session_id: str, filename: str, format: str):
    from fastapi.responses import Response, StreamingResponse
    from src.api_utils.code_registry import get_code_registry
    from io import BytesIO

    code_registry = get_code_registry()
    code_entry = code_registry.get_code_for_artifact(session_id, filename)

    if format == "code":
        if not code_entry:
            raise HTTPException(status_code=404, detail="Code not found for this artifact")

        code_content = code_entry.get("code", "")
        py_filename = filename.rsplit(".", 1)[0] + ".py"

        return Response(
            content=code_content,
            media_type="text/x-python",
            headers={"Content-Disposition": f"attachment; filename={py_filename}"},
        )

    elif format == "notebook":
        if not code_entry:
            raise HTTPException(status_code=404, detail="Code not found for this artifact")

        import nbformat
        from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

        nb = new_notebook()
        nb.cells.append(new_markdown_cell(f"# Code for {filename}"))
        nb.cells.append(new_code_cell("import pandas as pd\nimport numpy as np\nimport plotly.express as px"))
        nb.cells.append(new_code_cell(code_entry.get("code", "")))

        nb_filename = filename.rsplit(".", 1)[0] + ".ipynb"

        return Response(
            content=nbformat.writes(nb),
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f"attachment; filename={nb_filename}"},
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")


@router.get("/{session_id}/code/export/{format}")
async def export_session_code(session_id: str, format: str):
    from fastapi.responses import Response
    from src.api_utils.code_registry import get_code_registry

    code_registry = get_code_registry()

    if format == "script":
        script_content = code_registry.export_as_script(session_id)
        if not script_content:
            raise HTTPException(status_code=404, detail="No code found for this session")

        return Response(
            content=script_content,
            media_type="text/x-python",
            headers={"Content-Disposition": f"attachment; filename=analysis_{session_id[:8]}.py"},
        )

    elif format == "notebook":
        notebook_content = code_registry.export_as_notebook(session_id)
        if not notebook_content:
            raise HTTPException(status_code=404, detail="No code found for this session")

        return Response(
            content=notebook_content,
            media_type="application/x-ipynb+json",
            headers={"Content-Disposition": f"attachment; filename=analysis_{session_id[:8]}.ipynb"},
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}. Use 'script' or 'notebook'")


@router.get("/{session_id}/code")
async def get_session_code(session_id: str):
    from src.api_utils.code_registry import get_code_registry

    code_registry = get_code_registry()
    codes = code_registry.get_session_code(session_id)

    return {"session_id": session_id, "code_entries": codes, "total": len(codes)}
