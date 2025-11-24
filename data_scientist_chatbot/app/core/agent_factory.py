"""Agent creation and subgraph factory functions"""

import json
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

try:
    from .logger import logger
except ImportError:
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from logger import logger

try:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.model_manager import ModelManager
except ImportError as e:
    raise ImportError(f"Import error in agent_factory.py: {e}")

model_manager = ModelManager()


def create_router_agent():
    """Create Router agent for fast binary routing decisions"""
    config = model_manager.get_ollama_config("router")
    return ChatOllama(**config)


def create_brain_agent():
    """Create Brain agent for business reasoning and planning"""
    config = model_manager.get_ollama_config("brain")
    return ChatOllama(**config)


def create_hands_agent():
    """Create Hands agent for code execution"""
    config = model_manager.get_ollama_config("hands")
    return ChatOllama(**config)


def create_status_agent():
    """Create Status agent for real-time progress updates"""
    config = model_manager.get_ollama_config("status")
    return ChatOllama(**config)
