from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Optional
import pandas as pd

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/{session_id}/preview")
async def get_data_preview(session_id: str, rows: int = 10):
    from ..api import session_store

    if session_id not in session_store:
        raise HTTPException(status_code=404, detail="Session not found")

    df = session_store[session_id]["dataframe"]
    preview_data = df.head(rows).to_dict('records')

    return {
        "data": preview_data,
        "shape": df.shape,
        "columns": df.columns.tolist()
    }


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
                "missing_percentage": {k: float(v) for k, v in (df.isnull().sum() / len(df) * 100).to_dict().items()}
            },
            "numeric_summary": {},
            "categorical_summary": {},
            "correlations": {}
        }

        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            desc = df[numeric_cols].describe()
            eda_report["numeric_summary"] = {col: {k: float(v) for k, v in desc[col].to_dict().items()} for col in numeric_cols}

            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                eda_report["correlations"] = {
                    col1: {col2: float(corr_matrix.loc[col1, col2]) if not pd.isna(corr_matrix.loc[col1, col2]) else 0.0
                           for col2 in numeric_cols}
                    for col1 in numeric_cols
                }

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:5]
        for col in categorical_cols:
            value_counts = df[col].value_counts().head()
            eda_report["categorical_summary"][col] = {str(k): int(v) for k, v in value_counts.to_dict().items()}

        session_store[session_id]["eda_report"] = eda_report

        return {
            "status": "success",
            "report": eda_report,
            "message": "EDA report generated successfully"
        }

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
        column_profiles = intelligence_profile.get('column_profiles', {})
        response_data['column_profiles'] = {
            col: {
                'semantic_type': profile.semantic_type.value,
                'confidence': profile.confidence,
                'evidence': profile.evidence,
                'recommendations': profile.recommendations
            }
            for col, profile in column_profiles.items()
        }

        if prof_request.include_domain_detection:
            response_data['domain_analysis'] = intelligence_profile.get('domain_analysis', {})

        if prof_request.include_relationships:
            response_data['relationship_analysis'] = intelligence_profile.get('relationship_analysis', {})

        response_data['overall_recommendations'] = intelligence_profile.get('overall_recommendations', [])
        response_data['profiling_metadata'] = {
            'deep_analysis': prof_request.deep_analysis,
            'total_columns': len(df.columns),
            'total_rows': len(df),
            'profile_timestamp': datetime.now().isoformat()
        }

        return {
            "status": "success",
            "intelligence_profile": response_data,
            "session_id": session_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profiling error: {str(e)}")


@router.get("/{session_id}/feature-recommendations")
async def get_feature_recommendations(
    session_id: str,
    target_column: Optional[str] = None,
    max_recommendations: int = 10,
    priority_filter: Optional[str] = None
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

        recommendations = fe_analysis.get('feature_engineering_recommendations', [])

        if priority_filter:
            recommendations = [rec for rec in recommendations if rec.priority == priority_filter]

        recommendations = recommendations[:max_recommendations]

        serializable_recommendations = [
            {
                'feature_type': rec.feature_type,
                'priority': rec.priority,
                'description': rec.description,
                'implementation': rec.implementation,
                'expected_benefit': rec.expected_benefit,
                'computational_cost': rec.computational_cost
            }
            for rec in recommendations
        ]

        session_store[session_id]["feature_analysis"] = fe_analysis

        return {
            "status": "success",
            "recommendations": serializable_recommendations,
            "feature_selection_strategy": fe_analysis.get('feature_selection_strategy', {}),
            "scaling_strategy": fe_analysis.get('scaling_strategy', {}),
            "implementation_pipeline": fe_analysis.get('implementation_pipeline', []),
            "total_recommendations": len(serializable_recommendations),
            "recommendation_metadata": {
                "target_column": target_column,
                "priority_filter": priority_filter,
                "generated_at": datetime.now().isoformat()
            }
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
            "memory_impact": enhanced_df.memory_usage().sum() / df.memory_usage().sum()
        }

        return {
            "status": "success",
            "message": "Feature engineering applied successfully",
            "impact_summary": impact_summary,
            "enhanced_data_shape": enhanced_df.shape,
            "new_columns": [col for col in enhanced_df.columns if col not in df.columns],
            "artifacts": {
                "enhanced_data": f"/api/data/{session_id}/download/enhanced-data"
            }
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

        relationship_analysis = intelligence_profile.get('relationship_analysis', {})
        relationships = relationship_analysis.get('relationships', [])

        if not relationships:
            return {
                "status": "success",
                "message": "No relationships found in the data",
                "graph_data": {
                    "nodes": [],
                    "edges": [],
                    "metadata": {"total_relationships": 0}
                }
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

        relationship_types = [rel.get('relationship_type', 'unknown') for rel in relationships]
        type_counts = {}
        for rel_type in relationship_types:
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1

        enhanced_graph = {
            "nodes": graph_data.get('nodes', []),
            "edges": graph_data.get('edges', []),
            "metadata": {
                "total_relationships": len(relationships),
                "relationship_type_counts": type_counts,
                "total_nodes": len(graph_data.get('nodes', [])),
                "graph_generated_at": datetime.now().isoformat()
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
                    "default": "#ddd"
                }
            }
        }

        return {
            "status": "success",
            "graph_data": enhanced_graph
        }

    except Exception as e:
        return {"status": "error", "detail": f"Relationship graph error: {str(e)}"}
