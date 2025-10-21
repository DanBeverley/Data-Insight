"""Natural language parser for model save format extraction"""
import re
from typing import Optional


class FormatParser:
    """
    Extract model save format preferences from natural language
    Non-rigid pattern matching for user intent
    """

    def __init__(self):
        self.format_patterns = {
            'onnx': [
                r'\bonnx\b',
                r'save\s+as\s+onnx',
                r'export\s+to\s+onnx',
                r'in\s+onnx\s+format',
            ],
            'joblib': [
                r'\bjoblib\b',
                r'save\s+with\s+joblib',
                r'\.joblib\b',
            ],
            'pickle': [
                r'\bpickle\b',
                r'\bpkl\b',
                r'save\s+as\s+pickle',
                r'\.pkl\b',
            ],
            'pytorch': [
                r'\bpytorch\b',
                r'\btorch\b',
                r'\.pt\b',
                r'\.pth\b',
                r'pytorch\s+format',
            ],
            'h5': [
                r'\bh5\b',
                r'\bhdf5\b',
                r'\.h5\b',
                r'keras\s+format',
            ],
            'savedmodel': [
                r'\bsavedmodel\b',
                r'saved_model',
                r'tensorflow\s+savedmodel',
                r'save\s+as\s+savedmodel',
            ],
            'json': [
                r'xgboost.*json',
                r'save\s+as\s+json',
                r'\.json\b',
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
        # First check explicit user request
        format_from_message = self.extract_format(user_message)
        if format_from_message:
            return format_from_message

        # Check code comments for format hints
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
        comment_lines = [line for line in code.split('\n') if line.strip().startswith('#')]

        for line in comment_lines:
            line_lower = line.lower()

            if 'format' in line_lower or 'save' in line_lower:
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
