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


def create_brain_agent(mode: str = "chat"):
    config = model_manager.get_ollama_config("brain")
    if mode == "report":
        config["reasoning"] = True
        config["temperature"] = 0.35
        if "options" not in config:
            config["options"] = {}
        config["options"]["num_predict"] = 8192
    else:
        config["reasoning"] = False
    return ChatOllama(**config)


def create_hands_agent():
    """Create Hands agent for code execution"""
    config = model_manager.get_ollama_config("hands")
    return ChatOllama(**config)


def create_status_agent():
    """Create Status agent for real-time progress updates"""
    config = model_manager.get_ollama_config("status")
    return ChatOllama(**config)


def create_vision_agent():
    """Create Vision agent for image analysis"""
    config = model_manager.get_ollama_config("vision")
    return ChatOllama(**config)


def create_verifier_agent():
    """Create Verifier agent for task validation"""
    config = model_manager.get_ollama_config("verifier")
    return ChatOllama(**config)
