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


def _build_multi_dataset_context(datasets_dict: dict, session_data: dict) -> str:
    """Build rich context for multiple datasets so Brain doesn't need tool calls."""
    import pandas as pd

    parts = [f"AVAILABLE DATASETS ({len(datasets_dict)} tables loaded):"]

    all_columns = {}
    for filename, df in datasets_dict.items():
        if df is None:
            continue

        base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
        cols = df.columns.tolist()
        all_columns[base_name] = set(cols)

        dtypes_summary = df.dtypes.value_counts().to_dict()
        dtype_str = ", ".join([f"{v} {k}" for k, v in dtypes_summary.items()])

        col_preview = ", ".join(cols[:12])
        if len(cols) > 12:
            col_preview += f"... (+{len(cols)-12} more)"

        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if df.size > 0 else 0

        parts.append(
            f"""
📊 {filename}:
   Shape: {df.shape[0]:,} rows × {df.shape[1]} cols | Types: {dtype_str} | Missing: {missing_pct:.1f}%
   Columns: [{col_preview}]"""
        )

    # Detect potential foreign key relationships
    relationships = []
    table_names = list(all_columns.keys())
    for i, t1 in enumerate(table_names):
        for t2 in table_names[i + 1 :]:
            common = all_columns[t1] & all_columns[t2]
            if common:
                relationships.append(f"   • {t1} ↔ {t2}: shared columns [{', '.join(sorted(common)[:5])}]")

    if relationships:
        parts.append("\n🔗 DETECTED RELATIONSHIPS:")
        parts.extend(relationships[:10])

    parts.append("\n💡 All datasets are available in sandbox as: datasets['customers'], datasets['orders'], etc.")

    return "\n".join(parts)


def get_data_context(session_id: str, query: str = None) -> str:
    """Get data context for Brain agent. Supports multiple datasets."""
    try:
        from src.api_utils.session_management import session_data_manager
        import pandas as pd
        import numpy as np
        from .rag_context import RAGContextManager

        session_data = session_data_manager.get_session(session_id)

        if not session_data:
            return "No dataset available. Please upload data first."

        # Multi-dataset support: check datasets dict first
        datasets_dict = session_data.get("datasets", {})
        if datasets_dict and len(datasets_dict) > 1:
            return _build_multi_dataset_context(datasets_dict, session_data)

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

            # 2. Statistical Summary (df.describe equivalent)
            stat_summary = ctx.get("statistical_summary", {})
            if stat_summary:
                stat_lines = ["STATISTICAL SUMMARY:"]
                for col, stats in list(stat_summary.items())[:10]:
                    stat_str = ", ".join(
                        [f"{k}: {v}" for k, v in stats.items() if k in ["count", "mean", "std", "min", "max", "50%"]]
                    )
                    if stat_str:
                        stat_lines.append(f"  • {col}: {stat_str}")
                if len(stat_lines) > 1:
                    context_parts.append("\n".join(stat_lines))

            # 3. Top Correlations
            correlations = ctx.get("top_correlations", [])
            if correlations:
                corr_lines = ["TOP CORRELATIONS:"]
                for c in correlations[:5]:
                    corr_lines.append(
                        f"  • {c['columns'][0]} ↔ {c['columns'][1]}: {c['correlation']} ({c['strength']})"
                    )
                context_parts.append("\n".join(corr_lines))

            # 4. Skewed Columns
            skewed = ctx.get("skewed_columns", {})
            if skewed:
                skew_lines = ["SKEWED COLUMNS (may need transformation):"]
                for col, info in list(skewed.items())[:5]:
                    skew_lines.append(f"  • {col}: {info['type']} (skew={info['skewness']})")
                context_parts.append("\n".join(skew_lines))

            # 5. Categorical Distributions
            cat_dist = ctx.get("categorical_distributions", {})
            if cat_dist:
                cat_lines = ["CATEGORICAL VALUE FREQUENCIES:"]
                for col, info in list(cat_dist.items())[:5]:
                    top_vals = ", ".join([f"{v['value']}({v['percentage']}%)" for v in info["top_values"][:3]])
                    cat_lines.append(f"  • {col} ({info['unique_count']} unique): {top_vals}")
                context_parts.append("\n".join(cat_lines))

            # 6. Outliers
            outliers = ctx.get("outlier_analysis", {})
            if outliers:
                out_lines = ["OUTLIER ANALYSIS (IQR method):"]
                for col, info in list(outliers.items())[:5]:
                    out_lines.append(f"  • {col}: {info['outlier_count']} outliers ({info['outlier_percentage']}%)")
                context_parts.append("\n".join(out_lines))

            # 7. Analysis Recommendations
            recs = ctx.get("analysis_recommendations", [])
            if recs:
                context_parts.append("RECOMMENDED ANALYSIS:\n• " + "\n• ".join(recs[:5]))

            # 8. Code Generation Hints
            hints = ctx.get("code_generation_hints", {})
            if hints:
                hint_parts = []
                if hints.get("highly_correlated_pairs"):
                    hint_parts.append(f"Highly correlated: {hints['highly_correlated_pairs']}")
                if hints.get("columns_needing_transformation"):
                    hint_parts.append(f"Need transformation: {hints['columns_needing_transformation']}")
                if hints.get("columns_with_outliers"):
                    hint_parts.append(f"Have outliers: {hints['columns_with_outliers']}")
                if hint_parts:
                    context_parts.append("CODE HINTS:\n• " + "\n• ".join(hint_parts))

            # 9. Column Profiles (condensed)
            if "column_details" in ctx:
                col_details = []
                for col, info in list(ctx["column_details"].items())[:20]:
                    dtype_str = info.get("semantic_type", info.get("dtype", "unknown"))
                    stats_str = ""
                    if "mean" in info and info["mean"] is not None:
                        stats_str = (
                            f"| Mean: {info['mean']:.2f} | Range: [{info.get('min', 0):.2f}, {info.get('max', 0):.2f}]"
                        )
                    col_details.append(f"  • {col} ({dtype_str}) {stats_str}")

                if col_details:
                    context_parts.append("COLUMN PROFILES:\n" + "\n".join(col_details))

        return "\n".join(context_parts)

    except Exception as e:
        logger.error(f"Failed to get data context: {e}", exc_info=True)
        return "Error loading dataset context"
