"""Custom exception classes for Data Insight application"""


class DataInsightError(Exception):
    """Base exception class for the application."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class LLMGenerationError(DataInsightError):
    """Raised when Ollama/OpenAI fails to generate a response or returns malformed JSON."""

    pass


class CodeExecutionError(DataInsightError):
    """Raised when the Sandbox (E2B) fails or code times out."""

    pass


class ResourceQuotaError(DataInsightError):
    """Raised when AWS/Azure quotas are exhausted."""

    pass


class StateManagementError(DataInsightError):
    """Raised when LangGraph state is corrupted (like the NoneType retry_count issue)."""

    pass


class ToolExecutionError(DataInsightError):
    """Raised when tool execution fails."""

    pass


class SessionNotFoundError(DataInsightError):
    """Raised when session ID is invalid or not found."""

    pass


class DataValidationError(DataInsightError):
    """Raised when data validation fails."""

    pass


class ConfigurationError(DataInsightError):
    """Raised when configuration is invalid or missing."""

    pass


class SandboxError(CodeExecutionError):
    """Raised when E2B sandbox operations fail."""

    pass


class GPUExecutionError(CodeExecutionError):
    """Raised when GPU training/execution fails."""

    pass


class ModelNotFoundError(DataInsightError):
    """Raised when trained model cannot be loaded."""

    pass


class StorageError(DataInsightError):
    """Raised when cloud storage operations fail."""

    pass
