"""Session context utilities"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
try:
    from core.logger import logger
except ImportError:
    from ..core.logger import logger


def get_artifacts_context(session_id: str) -> str:
    """Get current session artifacts for brain agent context"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
        from src.api_utils.artifact_tracker import artifact_tracker

        artifacts_data = artifact_tracker.get_session_artifacts(session_id)
        artifacts = artifacts_data.get("artifacts", [])

        if not artifacts:
            return ""

        categories = artifacts_data.get("categories", {})
        artifact_lines = []

        for cat_key, cat_data in categories.items():
            cat_artifacts = cat_data.get("artifacts", [])
            if cat_artifacts:
                artifact_lines.append(f"{cat_data['icon']} {cat_data['label']} ({len(cat_artifacts)}):")
                for artifact in cat_artifacts[:10]:
                    artifact_id = artifact.get("artifact_id", "")
                    filename = artifact.get("filename", "")
                    file_path = artifact.get("file_path", "")
                    description = artifact.get("description", "")

                    desc_part = f" - {description}" if description and description != filename else ""
                    artifact_lines.append(f"  • {filename} (ID: {artifact_id}){desc_part}")
                    artifact_lines.append(f"    Path: {file_path}")
                if len(cat_artifacts) > 10:
                    artifact_lines.append(f"  ... and {len(cat_artifacts) - 10} more")

        return f"""
AVAILABLE ARTIFACTS ({len(artifacts)} total):
{chr(10).join(artifact_lines)}

To embed images: use the Path shown above
To zip artifacts: use zip_artifacts tool with the ID values (not filenames)
"""
    except Exception as e:
        logger.warning(f"Failed to get artifacts context: {e}")
        return ""


def get_data_context(session_id: str, query: str = None) -> str:
    """
    Get data context, optionally using RAG for specific queries.
    If query is provided, retrieves relevant columns/insights.
    If no query, provides a high-level summary (not a full dump).
    """
    try:
        from src.api_utils.session_management import session_data_manager
        import pandas as pd
        import numpy as np
        from .rag_context import RAGContextManager

        session_data = session_data_manager.get_session(session_id)

        if not session_data:
            return "No dataset available. Please upload data first."

        unified_context = session_data.get("unified_context")
        if unified_context:
            df = session_data.get("dataframe")
            if df is not None:
                extra = f"\n\nUNIFIED DATAFRAME STATS:\n• Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n• ALL COLUMNS: {df.columns.tolist()}"
                return unified_context + extra
            return unified_context

        df = session_data.get("dataframe")
        data_profile = session_data.get("data_profile")

        if df is None:
            return "No dataset available. Please upload data first."

        # [PERFORMANCE FIX] Disable strict RAG indexing on critial path to prevent 2-minute latency
        # rag_manager = None
        # if "rag_manager" not in session_data:
        #     try:
        #         rag_manager = RAGContextManager(session_id)
        #         # rag_manager.index_dataset(df, data_profile) # DISABLED for speed
        #         session_data["rag_manager"] = rag_manager
        #     except Exception as e:
        #         logger.warning(f"Failed to init RAG: {e}")
        # else:
        #     rag_manager = session_data["rag_manager"]

        # If query is provided and RAG is active, use semantic retrieval
        # if query and rag_manager:
        #     return rag_manager.retrieve_context(query)

        # Fallback / Default: High-Level Summary with FULL column visibility
        context_parts = []

        # 1. Dataset Overview
        context_parts.append(
            f"""
DATASET OVERVIEW:
• Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
• Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
• Missing: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%"""
        )

        # 2. FULL Column List (Critical for Hands Agent)
        all_cols = df.columns.tolist()
        col_list_str = ", ".join(all_cols[:200])  # Limit to 200 to be safe but comprehensive
        if len(all_cols) > 200:
            col_list_str += f"... (+{len(all_cols)-200} more)"

        context_parts.append(f"\nALL COLUMNS: [{col_list_str}]")

        # Rich Context Injection from HybridDataProfiler
        if data_profile and hasattr(data_profile, "ai_agent_context"):
            ctx = data_profile.ai_agent_context

            # 1. Overview & Quality
            quality = ctx.get("data_quality", {})
            anomalies = ctx.get("anomalies", {})

            context_parts.append(
                f"""
DATASET INTELLIGENCE:
• Quality Score: {quality.get("overall_score", "N/A")}/100
• Anomalies: {anomalies.get("total_count", 0)} detected
"""
            )

            # 2. Analysis Recommendations
            recs = ctx.get("analysis_recommendations", [])
            if recs:
                context_parts.append("RECOMMENDED ANALYSIS:\n• " + "\n• ".join(recs[:5]))

            # 3. RICH COLUMN INTELLIGENCE (Crucial for Planning)
            if "column_details" in ctx:
                col_details = []
                for col, info in list(ctx["column_details"].items())[:30]:  # Limit to top 30 cols to save tokens
                    dtype_str = info.get("semantic_type", info.get("dtype", "unknown"))

                    # Numeric Stats
                    stats_str = ""
                    if "mean" in info:
                        stats_str = (
                            f"| Mean: {info['mean']:.2f} | Range: [{info.get('min'):.2f}, {info.get('max'):.2f}]"
                        )

                    # Sample Values
                    samples = info.get("sample_values", [])
                    sample_str = f"Samples: {samples}" if samples else ""

                    col_details.append(f"- {col} ({dtype_str}) {stats_str} {sample_str}")

                if col_details:
                    context_parts.append("\nCOLUMN PROFILES (Top 30):\n" + "\n".join(col_details))

        else:
            pass  # Columns already added above

        return "\n".join(context_parts)

    except Exception as e:
        logger.error(f"Failed to get data context: {e}", exc_info=True)
        return "Error loading dataset context"
