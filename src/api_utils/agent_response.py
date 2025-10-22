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


def stream_agent_response(query: str, session_id: str, stream: bool = True) -> Generator[str, None, None]:
    MAX_QUERY_LENGTH = 50000

    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")

    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query too long: {len(query)} characters exceeds maximum of {MAX_QUERY_LENGTH}")

    from data_scientist_chatbot.app.utils.sanitizers import sanitize_input

    sanitized_query = sanitize_input(query)

    if not sanitized_query or len(sanitized_query) < 3:
        raise ValueError("Query is empty or too short after sanitization")

    yield f"Processing query for session {session_id}: {sanitized_query[:100]}..."
