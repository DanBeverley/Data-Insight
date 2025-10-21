import re
from typing import List, Tuple, Any


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
