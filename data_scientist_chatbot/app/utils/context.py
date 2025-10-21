"""Session context utilities"""
import logging
import sys
import os

logger = logging.getLogger(__name__)

def get_artifacts_context(session_id: str) -> str:
    """Get current session artifacts for brain agent context"""
    try:
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
        from src.api_utils.artifact_tracker import artifact_tracker

        artifacts_data = artifact_tracker.get_session_artifacts(session_id)
        artifacts = artifacts_data.get('artifacts', [])

        if not artifacts:
            return ""

        categories = artifacts_data.get('categories', {})
        artifact_lines = []

        for cat_key, cat_data in categories.items():
            cat_artifacts = cat_data.get('artifacts', [])
            if cat_artifacts:
                artifact_lines.append(f"{cat_data['icon']} {cat_data['label']} ({len(cat_artifacts)}):")
                for artifact in cat_artifacts[:10]:
                    artifact_id = artifact.get('artifact_id', '')
                    filename = artifact.get('filename', '')
                    file_path = artifact.get('file_path', '')
                    description = artifact.get('description', '')

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

def get_data_context(session_id: str) -> str:
    """Shared function to compute comprehensive data context from session store"""
    try:
        import builtins
        import pandas as pd
        import numpy as np

        if not (hasattr(builtins, '_session_store') and session_id in builtins._session_store):
            return "No dataset available. Please upload data first."

        session_data = builtins._session_store[session_id]
        df = session_data.get('dataframe')
        data_profile = session_data.get('data_profile')

        if df is None:
            return "No dataset available. Please upload data first."

        # Build comprehensive context
        context_parts = []

        # 1. Dataset Overview
        context_parts.append(f"""
DATASET OVERVIEW:
• Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
• Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
• Missing values: {df.isnull().sum().sum():,} total ({df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.1f}%)""")

        # 2. Column Details with Statistics
        if data_profile and hasattr(data_profile, 'ai_agent_context'):
            column_context = data_profile.ai_agent_context.get('column_details', {})
            hints = data_profile.ai_agent_context.get('code_generation_hints', {})

            context_parts.append("\nCOLUMNS:")
            for col, info in column_context.items():
                line = f"• {col}: {info['dtype']} ({info['semantic_type']})"
                if info['null_count'] > 0:
                    line += f" - {info['null_count']} nulls ({info['null_count']/df.shape[0]*100:.1f}%)"
                line += f", {info['unique_count']} unique"
                context_parts.append(line)
        else:
            # Fallback: compute basic column info
            context_parts.append("\nCOLUMNS:")
            for col in df.columns:
                null_count = df[col].isnull().sum()
                unique_count = df[col].nunique()
                dtype = str(df[col].dtype)
                line = f"• {col}: {dtype}"
                if null_count > 0:
                    line += f" - {null_count} nulls ({null_count/len(df)*100:.1f}%)"
                line += f", {unique_count} unique"
                context_parts.append(line)

        # 3. Statistical Summary for Numeric Columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            context_parts.append(f"\nNUMERIC STATISTICS ({len(numeric_cols)} columns):")
            desc = df[numeric_cols].describe()
            for col in numeric_cols[:10]:  # Show details for first 10
                stats = desc[col]
                context_parts.append(
                    f"• {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                    f"range=[{stats['min']:.2f}, {stats['max']:.2f}], "
                    f"median={stats['50%']:.2f}"
                )
            if len(numeric_cols) > 10:
                context_parts.append(f"  ... and {len(numeric_cols) - 10} more numeric columns")

        # 4. Categorical Summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            context_parts.append(f"\nCATEGORICAL COLUMNS ({len(categorical_cols)} total):")
            for col in categorical_cols[:5]:  # Top 5
                n_unique = df[col].nunique()
                top_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
                top_freq = df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
                context_parts.append(
                    f"• {col}: {n_unique} categories, "
                    f"most common='{top_val}' ({top_freq/len(df)*100:.1f}%)"
                )
            if len(categorical_cols) > 5:
                context_parts.append(f"  ... and {len(categorical_cols) - 5} more categorical columns")

        # 5. Correlations (top insights)
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            # Get top 5 absolute correlations (excluding diagonal)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            if corr_pairs:
                context_parts.append("\nTOP CORRELATIONS:")
                for col1, col2, corr_val in corr_pairs[:5]:
                    context_parts.append(f"• {col1} ↔ {col2}: {corr_val:.3f}")

        # 6. Data Quality Issues
        issues = []
        high_missing_cols = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.3]
        if high_missing_cols:
            issues.append(f"High missing values: {', '.join(high_missing_cols[:3])}")

        high_cardinality_cols = [col for col in categorical_cols
                                if df[col].nunique() / len(df) > 0.5]
        if high_cardinality_cols:
            issues.append(f"High cardinality: {', '.join(high_cardinality_cols[:3])}")

        if issues:
            context_parts.append("\nDATA QUALITY ALERTS:")
            for issue in issues:
                context_parts.append(f"⚠ {issue}")

        # 7. Suggested Analysis Approaches
        if data_profile and hasattr(data_profile, 'ai_agent_context'):
            hints = data_profile.ai_agent_context.get('code_generation_hints', {})
            if hints.get('suggested_target_columns'):
                context_parts.append(f"\nSUGGESTED TARGETS: {', '.join(hints['suggested_target_columns'][:3])}")

        return "\n".join(context_parts)

    except Exception as e:
        logger.error(f"Failed to get data context: {e}", exc_info=True)
        return "Error loading dataset context"