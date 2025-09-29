"""Output sanitization utilities"""

import re
from difflib import SequenceMatcher


def sanitize_output(content: str) -> str:
    """Semantic output sanitization using similarity-based filtering to remove technical debug artifacts"""
    if not content or not isinstance(content, str):
        return content

    technical_patterns = [
        "PLOT_SAVED:",
        "Checking for figures to save",
        "Figure saved to",
        "DEBUG:",
        "INFO:",
        "WARNING:",
        "ERROR:",
        "Traceback (most recent call last):",
        "File \"<stdin>\", line",
        "NameError:",
        "TypeError:",
        "ValueError:",
        "KeyError:",
        "IndexError:",
        "AttributeError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "SyntaxError:",
        "IndentationError:",
        "matplotlib.pyplot",
        "plt.show()",
        "plt.savefig",
        ">>> ",
        "... ",
        "Execution completed",
        "Code executed successfully",
        "Process finished",
        "Output generated",
        "File written to",
        "Data loaded from",
        "Connection established",
        "Database query executed",
        "API call completed",
        "Request processed",
        "Response received",
        "Cache updated",
        "Session initialized",
        "Memory allocated",
        "Resource freed",
        "Lock acquired",
        "Lock released",
        "Thread started",
        "Thread completed",
        "Task queued",
        "Task completed",
        "Validation passed",
        "Validation failed",
        "Configuration loaded",
        "Settings updated",
        "Logging initialized",
        "Performance metrics",
        "Benchmark results",
        "Test passed",
        "Test failed",
        "Assert",
        "Exception caught",
        "Exception handled",
        "Cleanup completed",
        "Shutdown initiated"
    ]

    lines = content.split('\n')
    filtered_lines = []

    for line in lines:
        line_stripped = line.strip()

        if not line_stripped:
            filtered_lines.append(line)
            continue

        is_technical = False
        for pattern in technical_patterns:
            if pattern.lower() in line_stripped.lower():
                is_technical = True
                break

            similarity_ratio = SequenceMatcher(None, pattern.lower(), line_stripped.lower()).ratio()
            if similarity_ratio > 0.6:
                is_technical = True
                break

            debug_prefixes = ['[', '(', '<', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'TRACE']
            for prefix in debug_prefixes:
                if line_stripped.startswith(prefix) and len(line_stripped) > 20:
                    if any(keyword in line_stripped.lower() for keyword in ['error', 'exception', 'traceback', 'debug', 'info', 'warning']):
                        is_technical = True
                        break

        technical_regexes = [
            r'^\s*File ".*", line \d+',
            r'^\s*\d+:\d+:\d+',
            r'^\s*\[.*\]\s*\d{4}-\d{2}-\d{2}',
            r'^\s*<.*>\s*',
            r'^\s*{.*}$',
            r'^\s*\w+\(\)\s*called',
            r'^\s*Process \d+ (started|completed|failed)',
            r'^\s*Memory usage: \d+',
            r'^\s*CPU usage: \d+',
            r'^\s*Execution time: \d+',
        ]

        for regex_pattern in technical_regexes:
            if re.match(regex_pattern, line_stripped, re.IGNORECASE):
                is_technical = True
                break

        if not is_technical:
            filtered_lines.append(line)
        elif len(line_stripped) > 100:
            meaningful_keywords = ['result', 'output', 'data', 'analysis', 'summary', 'conclusion', 'finding']
            if any(keyword in line_stripped.lower() for keyword in meaningful_keywords):
                parts = line_stripped.split(':')
                if len(parts) > 1 and len(parts[-1].strip()) > 10:
                    filtered_lines.append(parts[-1].strip())
                else:
                    filtered_lines.append(line)

    filtered_content = '\n'.join(filtered_lines)
    filtered_content = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered_content)
    filtered_content = filtered_content.strip()

    if len(filtered_content) < len(content) * 0.3:
        return content

    return filtered_content if filtered_content else content