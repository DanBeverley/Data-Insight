"""Session context utilities"""

import logging

logger = logging.getLogger(__name__)


def get_data_context(session_id: str) -> str:
    """Shared function to compute data_context from session store"""
    try:
        import builtins
        if (hasattr(builtins, '_session_store') and
            session_id in builtins._session_store and
            'data_profile' in builtins._session_store[session_id]):

            data_profile = builtins._session_store[session_id]['data_profile']
            column_context = data_profile.ai_agent_context['column_details']
            hints = data_profile.ai_agent_context['code_generation_hints']

            return f"""
                    DATASET CONTEXT:
                    Available dataset with columns:
                    {chr(10).join(f'• {col}: {info["dtype"]} ({info["semantic_type"]}) - {info["null_count"]} nulls, {info["unique_count"]} unique values'
                                    for col, info in column_context.items())}

                    INSIGHTS:
                    • Numeric columns: {', '.join(hints['numeric_columns'])}
                    • Categorical columns: {', '.join(hints['categorical_columns'])}
                    • Suggested targets: {', '.join(hints['suggested_target_columns'])}
                    """
    except Exception as e:
        logger.warning(f"Failed to get data context: {e}")
        return "No dataset context available"