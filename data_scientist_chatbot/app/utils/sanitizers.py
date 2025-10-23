"""Input and output sanitization utilities"""

import re
from pathlib import Path
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
        'File "<stdin>", line',
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
        "Shutdown initiated",
    ]

    lines = content.split("\n")
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
            debug_prefixes = ["[", "(", "<", "DEBUG", "INFO", "WARN", "ERROR", "TRACE"]
            for prefix in debug_prefixes:
                if line_stripped.startswith(prefix) and len(line_stripped) > 20:
                    if any(
                        keyword in line_stripped.lower()
                        for keyword in ["error", "exception", "traceback", "debug", "info", "warning"]
                    ):
                        is_technical = True
                        break
        technical_regexes = [
            r'^\s*File ".*", line \d+',
            r"^\s*\d+:\d+:\d+",
            r"^\s*\[.*\]\s*\d{4}-\d{2}-\d{2}",
            r"^\s*<.*>\s*",
            r"^\s*{.*}$",
            r"^\s*\w+\(\)\s*called",
            r"^\s*Process \d+ (started|completed|failed)",
            r"^\s*Memory usage: \d+",
            r"^\s*CPU usage: \d+",
            r"^\s*Execution time: \d+",
        ]

        for regex_pattern in technical_regexes:
            if re.match(regex_pattern, line_stripped, re.IGNORECASE):
                is_technical = True
                break

        if not is_technical:
            filtered_lines.append(line)
        elif len(line_stripped) > 100:
            meaningful_keywords = ["result", "output", "data", "analysis", "summary", "conclusion", "finding"]
            if any(keyword in line_stripped.lower() for keyword in meaningful_keywords):
                parts = line_stripped.split(":")
                if len(parts) > 1 and len(parts[-1].strip()) > 10:
                    filtered_lines.append(parts[-1].strip())
                else:
                    filtered_lines.append(line)
    filtered_content = "\n".join(filtered_lines)
    filtered_content = re.sub(r"\n\s*\n\s*\n", "\n\n", filtered_content)
    filtered_content = filtered_content.strip()
    if len(filtered_content) < len(content) * 0.3:
        return content
    return filtered_content if filtered_content else content


def sanitize_input(user_input: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not user_input or not isinstance(user_input, str):
        return user_input

    sanitized = user_input

    sql_patterns = [
        r"('\s*(OR|AND)\s*'?\d*'?\s*=\s*'?\d*)",
        r"(--)",
        r"(;.*DROP)",
        r"(UNION\s+SELECT)",
        r"(/\*.*\*/)",
    ]
    for pattern in sql_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

    xss_patterns = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"onerror\s*=",
        r"onload\s*=",
        r"<iframe",
        r"<embed",
        r"<object",
    ]
    for pattern in xss_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

    cmd_patterns = [r"`", r"\$\(", r"\|\s*\w+"]
    for pattern in cmd_patterns:
        sanitized = re.sub(pattern, "", sanitized)

    code_patterns = [r"exec\s*\(", r"eval\s*\(", r"__import__"]
    for pattern in code_patterns:
        sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)

    return sanitized


def sanitize_file_path(file_path: str) -> str:
    """Validate and sanitize file paths to prevent path traversal"""
    if not file_path:
        raise ValueError("File path cannot be empty")

    dangerous_patterns = ["..", "~", "/etc/", "\\windows\\", "\\system32\\", "C:\\", "/root/"]

    for pattern in dangerous_patterns:
        if pattern.lower() in file_path.lower():
            raise ValueError(f"Invalid file path: contains forbidden pattern '{pattern}'")

    try:
        path = Path(file_path)
        if path.is_absolute():
            raise ValueError("Absolute paths are not allowed")

        resolved = path.resolve()
        if ".." in str(resolved):
            raise ValueError("Path traversal detected")

    except Exception as e:
        raise ValueError(f"Invalid file path: {str(e)}")

    return file_path
