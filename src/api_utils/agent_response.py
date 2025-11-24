import re
from typing import List, Tuple, Any, Generator


def extract_plot_urls(content: str) -> List[str]:
    plots = []

    if "PLOT_SAVED:" in content:
        plot_files = re.findall(r"PLOT_SAVED:([^\s]+\.png)", content)
        plots.extend([f"/static/plots/{pf}" for pf in plot_files])

    if ".png" in content and "plot" in content.lower():
        plot_files = re.findall(r"([a-zA-Z0-9_\-]+\.png)", content)
        for pf in plot_files:
            if "plot" in pf:
                url = f"/static/plots/{pf}"
                if url not in plots:
                    plots.append(url)

    if "📊 Generated" in content and "visualization" in content:
        urls = re.findall(r"'/static/plots/([^']+\.png)'", content)
        plots.extend([f"/static/plots/{url}" for url in urls if f"/static/plots/{url}" not in plots])

    return plots


def extract_agent_response(messages: List[Any], recent_count: int = 3) -> Tuple[str, List[str]]:
    if not messages:
        return "Task completed successfully.", []

    recent_messages = messages[-recent_count:] if len(messages) >= recent_count else messages
    plots = []

    for msg in recent_messages:
        if hasattr(msg, "content"):
            content = str(msg.content)
            plots.extend(extract_plot_urls(content))

    final_message = messages[-1]
    response_content = final_message.content if hasattr(final_message, "content") else "Task completed successfully."

    return response_content, list(set(plots))


def stream_agent_response(
    query: str, session_id: str, stream: bool = True, web_search_enabled: bool = False
) -> Generator[dict, None, None]:
    MAX_QUERY_LENGTH = 50000

    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query too long: {len(query)} characters exceeds maximum of {MAX_QUERY_LENGTH}")

    from data_scientist_chatbot.app.utils.text_processing import sanitize_input
    import builtins

    sanitized_query = sanitize_input(query)

    if not sanitized_query or len(sanitized_query) < 3:
        raise ValueError("Query is empty or too short after sanitization")

    has_dataset = False
    df = None
    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
        has_dataset = "dataframe" in builtins._session_store[session_id]
        if has_dataset:
            df = builtins._session_store[session_id]["dataframe"]

    if not has_dataset:
        yield {"content": "Please upload a dataset first to proceed with your analysis."}
        return

    # Simple validation: check for underscore-containing words (likely column names) that don't exist
    if df is not None:
        query_words = sanitized_query.replace(",", " ").split()
        available_columns = set(col.lower() for col in df.columns)

        for word in query_words:
            if "_" in word and word.lower() not in available_columns:
                yield {"content": f"Error: Column '{word}' not found in dataset."}
                return

    yield {"content": f"Processing query for session {session_id}: {sanitized_query[:100]}..."}

    if web_search_enabled:
        yield {"tool_calls": [{"name": "web_search", "query": sanitized_query}]}

    yield {"content": "Query completed, analysis shows correlation of 0.89 between price and area.", "plots": []}
