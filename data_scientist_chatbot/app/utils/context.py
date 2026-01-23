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


def _profile_single_table(df, table_name: str) -> dict:
    """Phase 1: Profile a single table with rich statistics."""
    import numpy as np

    profile = {
        "name": table_name,
        "shape": df.shape,
        "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if df.size > 0 else 0,
        "columns": {},
        "numeric_cols": [],
        "categorical_cols": [],
        "pk_candidates": [],
    }

    for col in df.columns:
        col_info = {"dtype": str(df[col].dtype), "null_pct": df[col].isnull().mean() * 100}

        if df[col].dtype in ["int64", "float64", "int32", "float32"]:
            profile["numeric_cols"].append(col)
            col_info.update(
                {
                    "mean": round(df[col].mean(), 2) if not df[col].isnull().all() else None,
                    "std": round(df[col].std(), 2) if not df[col].isnull().all() else None,
                    "min": df[col].min() if not df[col].isnull().all() else None,
                    "max": df[col].max() if not df[col].isnull().all() else None,
                }
            )
            if df[col].nunique() == len(df) and df[col].notnull().all():
                profile["pk_candidates"].append(col)
        else:
            profile["categorical_cols"].append(col)
            unique_count = df[col].nunique()
            col_info["unique_count"] = unique_count
            if unique_count <= 20:
                top_vals = df[col].value_counts().head(5)
                col_info["top_values"] = [f"{v}({c})" for v, c in top_vals.items()]
            if unique_count == len(df) and df[col].notnull().all():
                profile["pk_candidates"].append(col)

        profile["columns"][col] = col_info

    return profile


def _detect_relationships(profiles: dict, datasets: dict) -> list:
    """Phase 2: Detect FK/PK relationships and compute join quality."""
    relationships = []
    table_names = list(profiles.keys())

    for i, t1 in enumerate(table_names):
        for t2 in table_names[i + 1 :]:
            p1, p2 = profiles[t1], profiles[t2]
            df1, df2 = datasets[t1], datasets[t2]

            common_cols = set(p1["columns"].keys()) & set(p2["columns"].keys())

            for col in common_cols:
                # Detect join type by uniqueness
                unique1 = df1[col].nunique()
                unique2 = df2[col].nunique()
                len1, len2 = len(df1), len(df2)

                if unique1 == len1 and unique2 < len2:
                    join_type = "1:N"
                    pk_table, fk_table = t1, t2
                elif unique2 == len2 and unique1 < len1:
                    join_type = "1:N"
                    pk_table, fk_table = t2, t1
                elif unique1 == len1 and unique2 == len2:
                    join_type = "1:1"
                    pk_table, fk_table = t1, t2
                else:
                    join_type = "N:M"
                    pk_table, fk_table = t1, t2

                # Join coverage
                vals1, vals2 = set(df1[col].dropna()), set(df2[col].dropna())
                matched = len(vals1 & vals2)
                coverage = matched / max(len(vals1), len(vals2), 1) * 100
                orphans = len(vals1 ^ vals2)

                relationships.append(
                    {
                        "tables": (t1, t2),
                        "column": col,
                        "type": join_type,
                        "pk_table": pk_table,
                        "fk_table": fk_table,
                        "coverage": round(coverage, 1),
                        "orphans": orphans,
                    }
                )

    return relationships


def _compute_cross_table_stats(datasets: dict, relationships: list) -> dict:
    """Phase 3: Compute cross-table correlations after logical join."""
    import numpy as np

    cross_stats = {"correlations": [], "unified_columns": {}}

    if not relationships:
        return cross_stats

    for rel in relationships[:3]:
        t1, t2 = rel["tables"]
        join_col = rel["column"]

        try:
            df1, df2 = datasets.get(t1), datasets.get(t2)
            if df1 is None or df2 is None:
                continue

            merged = df1.merge(df2, on=join_col, how="inner", suffixes=(f"_{t1}", f"_{t2}"))

            # Cross-table correlation between numeric columns
            nums = merged.select_dtypes(include=[np.number]).columns.tolist()
            if len(nums) >= 2:
                corr_matrix = merged[nums].corr()
                for c1 in nums:
                    for c2 in nums:
                        if c1 != c2 and c1.split("_")[-1] != c2.split("_")[-1]:
                            corr_val = corr_matrix.loc[c1, c2]
                            if abs(corr_val) > 0.5:
                                cross_stats["correlations"].append(
                                    {
                                        "col1": c1,
                                        "col2": c2,
                                        "correlation": round(corr_val, 3),
                                        "tables": f"{t1} × {t2}",
                                    }
                                )
        except Exception:
            continue

    cross_stats["correlations"] = sorted(
        cross_stats["correlations"], key=lambda x: abs(x["correlation"]), reverse=True
    )[:10]

    return cross_stats


def _build_semantic_layer(profiles: dict, datasets: dict) -> dict:
    """Phase 4: Map business terms and detect domain."""
    semantic = {"domain": "general", "business_terms": {}, "descriptions": []}

    all_cols = []
    for p in profiles.values():
        all_cols.extend(p["columns"].keys())
    all_cols_lower = [c.lower() for c in all_cols]

    # Domain detection
    finance_terms = ["price", "revenue", "cost", "profit", "amount", "balance", "payment"]
    ecommerce_terms = ["order", "product", "customer", "cart", "shipping", "sku"]
    healthcare_terms = ["patient", "diagnosis", "medication", "treatment", "doctor"]

    finance_score = sum(1 for t in finance_terms if any(t in c for c in all_cols_lower))
    ecommerce_score = sum(1 for t in ecommerce_terms if any(t in c for c in all_cols_lower))
    healthcare_score = sum(1 for t in healthcare_terms if any(t in c for c in all_cols_lower))

    max_score = max(finance_score, ecommerce_score, healthcare_score)
    if max_score >= 2:
        if finance_score == max_score:
            semantic["domain"] = "finance"
        elif ecommerce_score == max_score:
            semantic["domain"] = "e-commerce"
        elif healthcare_score == max_score:
            semantic["domain"] = "healthcare"

    # Business term mapping
    term_patterns = {
        "primary_key": ["id", "pk", "key"],
        "timestamp": ["date", "time", "created", "updated", "timestamp"],
        "monetary": ["price", "cost", "amount", "revenue", "total", "fee"],
        "identifier": ["name", "title", "label", "code", "sku"],
        "quantity": ["count", "qty", "quantity", "number"],
    }

    for col in all_cols:
        col_lower = col.lower()
        for term, patterns in term_patterns.items():
            if any(p in col_lower for p in patterns):
                if term not in semantic["business_terms"]:
                    semantic["business_terms"][term] = []
                semantic["business_terms"][term].append(col)

    return semantic


def _build_multi_dataset_context(datasets_dict: dict, session_data: dict) -> str:
    """Build comprehensive context for multiple datasets with 4-phase analysis."""
    import pandas as pd

    parts = [f"MULTI-TABLE DATASET ({len(datasets_dict)} tables loaded)"]

    # Phase 1: Profile each table
    profiles = {}
    for filename, df in datasets_dict.items():
        if df is None:
            continue
        profiles[filename] = _profile_single_table(df, filename)

    for name, p in profiles.items():
        col_preview = ", ".join(list(p["columns"].keys())[:10])
        if len(p["columns"]) > 10:
            col_preview += f"... (+{len(p['columns'])-10} more)"

        parts.append(
            f"""
📊 **{name}**
   Shape: {p['shape'][0]:,} rows × {p['shape'][1]} cols | Missing: {p['missing_pct']:.1f}%
   Columns: [{col_preview}]
   Numeric: {len(p['numeric_cols'])} | Categorical: {len(p['categorical_cols'])}
   PK Candidates: {p['pk_candidates'][:3] if p['pk_candidates'] else 'None detected'}"""
        )

        # Top statistics for numeric columns
        if p["numeric_cols"]:
            stat_preview = []
            for col in p["numeric_cols"][:3]:
                info = p["columns"].get(col, {})
                if info.get("mean") is not None:
                    stat_preview.append(f"{col}: μ={info['mean']}, range=[{info['min']}, {info['max']}]")
            if stat_preview:
                parts.append("   Stats: " + " | ".join(stat_preview))

    # Phase 2: Relationship detection
    relationships = _detect_relationships(profiles, datasets_dict)
    if relationships:
        parts.append("\n🔗 TABLE RELATIONSHIPS:")
        for rel in relationships[:5]:
            parts.append(
                f"   • {rel['pk_table']}.{rel['column']} → {rel['fk_table']}.{rel['column']} "
                f"({rel['type']}, {rel['coverage']}% coverage, {rel['orphans']} orphans)"
            )

    # Phase 3: Cross-table analysis
    cross_stats = _compute_cross_table_stats(datasets_dict, relationships)
    if cross_stats["correlations"]:
        parts.append("\n📈 CROSS-TABLE CORRELATIONS:")
        for corr in cross_stats["correlations"][:5]:
            parts.append(f"   • {corr['col1']} ↔ {corr['col2']}: {corr['correlation']} ({corr['tables']})")

    # Phase 4: Semantic layer
    semantic = _build_semantic_layer(profiles, datasets_dict)
    parts.append(f"\n🏷️ DETECTED DOMAIN: {semantic['domain'].upper()}")
    if semantic["business_terms"]:
        term_summary = []
        for term, cols in list(semantic["business_terms"].items())[:4]:
            term_summary.append(f"{term}: {cols[:2]}")
        parts.append("   Business terms: " + " | ".join(term_summary))

    # Code hints
    parts.append(
        f"""
💡 ACCESS PATTERNS:
   • All tables: datasets['table_name'] or globals()[table_name]
   • Join example: pd.merge(datasets['{list(datasets_dict.keys())[0]}'], datasets['{list(datasets_dict.keys())[-1]}'], on='shared_col')
   • Available tables: {list(datasets_dict.keys())}"""
    )

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
