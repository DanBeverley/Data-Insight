"""Model Arena section builder - trains and compares multiple models"""

from typing import Dict, Any, List
from langsmith import traceable


@traceable(name="train_baseline_models", tags=["ml", "model-arena"])
async def train_baseline_models(dataset_path: str, target_col: str, session_id: str) -> List[Dict[str, Any]]:
    """
    Train baseline models using the hands agent.

    This function delegates all training logic to the existing hands agent,
    avoiding code duplication.

    Args:
        dataset_path: Path to the dataset CSV
        target_col: Target column for prediction
        session_id: User session identifier

    Returns:
        List of model results with metrics
    """
    from data_scientist_chatbot.app.agent import run_hands_agent
    from langchain_core.messages import HumanMessage
    import json
    import re

    prompt = f"""
Train these 3 models on the dataset at {dataset_path} to predict '{target_col}':
1. XGBoost (max_depth=6, n_estimators=100)
2. RandomForest (n_estimators=100)
3. LogisticRegression (max_iter=1000)

For each model, return JSON in this exact format:
{{
    "model_name": "XGBoost",
    "accuracy": 0.89,
    "f1_score": 0.84,
    "training_time_seconds": 42,
    "model_file_path": "/path/to/model.pkl"
}}

Return a JSON array with all three model results.
"""

    state = {"messages": [HumanMessage(content=prompt)], "session_id": session_id, "plan": None, "scratchpad": ""}

    result = run_hands_agent(state)
    messages = result.get("messages", [])

    if messages and hasattr(messages[-1], "content"):
        content = messages[-1].content

        try:
            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                models_data = json.loads(json_match.group(0))
                return models_data
        except Exception as e:
            pass

    return [
        {
            "model_name": "XGBoost",
            "accuracy": 0.0,
            "f1_score": 0.0,
            "training_time_seconds": 0,
            "error": "Training failed",
        }
    ]
