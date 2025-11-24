"""LangSmith tracing configuration and decorators"""

import os
from typing import Optional
from langsmith import Client, traceable

client: Optional[Client] = None


def initialize_langsmith() -> Optional[Client]:
    """
    Initialize LangSmith client with environment variables.

    Returns:
        Client instance if enabled, None otherwise
    """
    global client

    if not os.getenv("LANGSMITH_TRACING", "false").lower() == "true":
        return None

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return None

    try:
        client = Client(api_key=api_key, api_url=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"))
        return client
    except Exception as e:
        print(f"Failed to initialize LangSmith: {e}")
        return None


def should_trace() -> bool:
    """
    Determine if current request should be traced.

    Development: 100% tracing
    Production: 10% sampling + 100% of errors
    """
    env = os.getenv("ENVIRONMENT", "development")

    if env == "development":
        return True

    import random

    return random.random() < 0.1


def get_trace_metadata(session_id: str = None, **kwargs) -> dict:
    """
    Build standard metadata for traces.

    Args:
        session_id: User session identifier
        **kwargs: Additional metadata fields

    Returns:
        Metadata dictionary
    """
    metadata = {
        "environment": os.getenv("ENVIRONMENT", "development"),
        "project": os.getenv("LANGSMITH_PROJECT", "data-insight-production"),
    }

    if session_id:
        metadata["session_id"] = session_id

    metadata.update(kwargs)
    return metadata


client = initialize_langsmith()
