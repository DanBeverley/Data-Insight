"""Consolidated text processing utilities: parsing, sanitization, and format extraction"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional


def parse_message_to_tool_call(message, tool_id_prefix="call"):
    """Extract tool calls from message content"""
    if hasattr(message, "tool_calls") and message.tool_calls:
        return True
    content_str = str(message.content).strip()
    try:
        json_str = content_str
        if "```json" in json_str:
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
        if not json_str.startswith("{") and "{" in json_str:
            json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
        tool_data = json.loads(json_str)
        if "name" in tool_data and "arguments" in tool_data:
            tool_name = tool_data["name"]
            tool_args = tool_data["arguments"]

            message.tool_calls = [{"name": tool_name, "args": tool_args, "id": f"{tool_id_prefix}_{tool_name}"}]
            message.content = ""
            return True
    except Exception:
        pass
    return False


def parse_deepseek_xml_tools(content: str) -> list:
    tool_calls = []
    try:
        invoke_pattern = r"<｜DSML｜invoke name=\"(.*?)\">(.*?)</｜DSML｜invoke>"
        invokes = re.findall(invoke_pattern, content, re.DOTALL)

        for name, params_block in invokes:
            args = {}
            param_pattern = r"<｜DSML｜parameter name=\"(.*?)\"(?: string=\"(.*?)\")?>(.*?)</｜DSML｜parameter>"
            params = re.findall(param_pattern, params_block, re.DOTALL)

            for param_name, is_string, param_value in params:
                if is_string == "false":
                    try:
                        args[param_name] = json.loads(param_value)
                    except:
                        args[param_name] = param_value
                else:
                    args[param_name] = param_value

            tool_calls.append({"name": name, "arguments": args})
    except Exception:
        pass
    return tool_calls


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
        "PROFILING_INSIGHTS_START",
        "PROFILING_INSIGHTS_END",
        "REPORT_SUMMARY_START",
        "REPORT_SUMMARY_END",
    ]

    # Strip structured blocks first
    block_patterns = [
        r"(?:\*\*PROFILING_INSIGHTS_START\*\*(.*?)\*\*PROFILING_INSIGHTS_END\*\*)",
        r"(?:PROFILING_INSIGHTS_START(.*?)PROFILING_INSIGHTS_END)",
        r"(?:\*\*REPORT_SUMMARY_START\*\*(.*?)\*\*REPORT_SUMMARY_END\*\*)",
        r"(?:REPORT_SUMMARY_START(.*?)REPORT_SUMMARY_END)",
    ]
    for pattern in block_patterns:
        content = re.sub(pattern, "", content, flags=re.DOTALL)

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


class FormatParser:
    """
    Extract model save format preferences from natural language
    Non-rigid pattern matching for user intent
    """

    def __init__(self):
        self.format_patterns = {
            "onnx": [
                r"\bonnx\b",
                r"save\s+as\s+onnx",
                r"export\s+to\s+onnx",
                r"in\s+onnx\s+format",
            ],
            "joblib": [
                r"\bjoblib\b",
                r"save\s+with\s+joblib",
                r"\.joblib\b",
            ],
            "pickle": [
                r"\bpickle\b",
                r"\bpkl\b",
                r"save\s+as\s+pickle",
                r"\.pkl\b",
            ],
            "pytorch": [
                r"\bpytorch\b",
                r"\btorch\b",
                r"\.pt\b",
                r"\.pth\b",
                r"pytorch\s+format",
            ],
            "h5": [
                r"\bh5\b",
                r"\bhdf5\b",
                r"\.h5\b",
                r"keras\s+format",
            ],
            "savedmodel": [
                r"\bsavedmodel\b",
                r"saved_model",
                r"tensorflow\s+savedmodel",
                r"save\s+as\s+savedmodel",
            ],
            "json": [
                r"xgboost.*json",
                r"save\s+as\s+json",
                r"\.json\b",
            ],
        }

    def extract_format(self, user_message: str) -> Optional[str]:
        """
        Extract format specification from user message

        Args:
            user_message: Natural language request from user

        Returns:
            Format name if detected, None otherwise

        Examples:
            "train model and save as ONNX" → "onnx"
            "export to joblib format" → "joblib"
            "save model in PyTorch format" → "pytorch"
            "train XGBoost model" → None (no format specified)
        """
        if not user_message:
            return None

        message_lower = user_message.lower()

        for format_name, patterns in self.format_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    return format_name

        return None

    def extract_with_context(self, user_message: str, code: Optional[str] = None) -> Optional[str]:
        """
        Extract format with additional context from code

        Args:
            user_message: User's natural language request
            code: Optional training code to analyze

        Returns:
            Format name or None
        """
        format_from_message = self.extract_format(user_message)
        if format_from_message:
            return format_from_message

        if code:
            format_from_code = self._extract_from_code_comments(code)
            if format_from_code:
                return format_from_code

        return None

    def _extract_from_code_comments(self, code: str) -> Optional[str]:
        """
        Extract format hints from code comments

        Examples:
            # save_format: onnx
            # Save as joblib
        """
        comment_lines = [line for line in code.split("\n") if line.strip().startswith("#")]

        for line in comment_lines:
            line_lower = line.lower()

            if "format" in line_lower or "save" in line_lower:
                for format_name, patterns in self.format_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, line_lower):
                            return format_name

        return None


# Global instance for easy import
format_parser = FormatParser()


def extract_format_from_request(user_message: str, code: Optional[str] = None) -> Optional[str]:
    """
    Convenience function for format extraction

    Args:
        user_message: User's natural language request
        code: Optional training code

    Returns:
        Format name or None
    """
    return format_parser.extract_with_context(user_message, code)
