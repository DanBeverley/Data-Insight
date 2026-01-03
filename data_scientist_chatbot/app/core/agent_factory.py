"""Agent creation and subgraph factory functions"""

import os
import json
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

try:
    from .logger import logger
except ImportError:
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from logger import logger

try:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from core.model_manager import ModelManager
except ImportError as e:
    raise ImportError(f"Import error in agent_factory.py: {e}")

model_manager = ModelManager()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

GEMINI_MODELS = {
    "brain": "gemini-3-pro",
    "hands": "gemini-3-pro",
    "status": "gemini-3-pro",
    "vision": "gemini-3-pro",
    "verifier": "gemini-3-pro",
}


def _create_gemini_agent(agent_type: str, thinking_budget: int = -1, **kwargs):
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = GEMINI_MODELS.get(agent_type, "gemini-3-pro")
    temperature = kwargs.get("temperature", 0.7)

    gen_config = {}
    if thinking_budget != -1:
        gen_config["thinkingConfig"] = {"thinkingBudget": thinking_budget}

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
        generation_config=gen_config if gen_config else None,
    )


def _create_ollama_agent(agent_type: str, **kwargs):
    config = model_manager.get_ollama_config(agent_type)
    config.update(kwargs)
    return ChatOllama(**config)


def create_brain_agent(mode: str = "chat"):
    temperature = 0.35 if mode == "report" else 0.7
    thinking_budget = 8192 if mode == "report" else -1

    if LLM_PROVIDER == "gemini" and GOOGLE_API_KEY:
        return _create_gemini_agent("brain", thinking_budget=thinking_budget, temperature=temperature)

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
    if LLM_PROVIDER == "gemini" and GOOGLE_API_KEY:
        return _create_gemini_agent("hands", temperature=0.2)
    config = model_manager.get_ollama_config("hands")
    return ChatOllama(**config)


def create_status_agent():
    if LLM_PROVIDER == "gemini" and GOOGLE_API_KEY:
        return _create_gemini_agent("status", temperature=0.5)
    config = model_manager.get_ollama_config("status")
    return ChatOllama(**config)


def create_vision_agent():
    if LLM_PROVIDER == "gemini" and GOOGLE_API_KEY:
        return _create_gemini_agent("vision", temperature=0.3)
    config = model_manager.get_ollama_config("vision")
    return ChatOllama(**config)


def create_verifier_agent():
    if LLM_PROVIDER == "gemini" and GOOGLE_API_KEY:
        return _create_gemini_agent("verifier", temperature=0.1)
    config = model_manager.get_ollama_config("verifier")
    return ChatOllama(**config)


logger.info(f"[AGENT_FACTORY] LLM Provider: {LLM_PROVIDER.upper()}")
