"""Unified training executor with CPU/GPU routing"""

import sys
import os
from typing import Dict, Any, Optional, List, Tuple
from langsmith import traceable

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from .training_decision import TrainingDecisionEngine
    from ..utils.text_processing import extract_format_from_request
    from .logger import logger
except ImportError:
    from training_decision import TrainingDecisionEngine

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
    from text_processing import extract_format_from_request

    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from logger import logger


class TrainingExecutor:
    """
    Unified training executor that routes to CPU or GPU based on complexity analysis
    """

    def __init__(self):
        self.decision_engine = TrainingDecisionEngine()

    @traceable(name="model_training", tags=["ml", "training"])
    def execute_training(
        self, code: str, session_id: str, user_request: Optional[str] = None, model_type: str = ""
    ) -> Dict[str, Any]:
        """
        Execute training code on appropriate infrastructure

        Args:
            code: Python training code
            session_id: Session identifier
            user_request: Original user request for format extraction
            model_type: Model type if known

        Returns:
            Execution results with environment info
        """
        # Get dataset info from session
        dataset_rows, feature_count = self._get_dataset_info(session_id)

        # Decide environment
        decision = self.decision_engine.decide(
            dataset_rows=dataset_rows, feature_count=feature_count, model_type=model_type, code=code
        )

        logger.info(f"[TrainingExecutor] Decision: {decision.environment}")
        logger.info(f"[TrainingExecutor] Reasoning: {decision.reasoning}")
        logger.info(f"[TrainingExecutor] Confidence: {decision.confidence:.2f}")

        # Extract format preference
        user_format = None
        if user_request:
            user_format = extract_format_from_request(user_request, code)
            if user_format:
                logger.info(f"[TrainingExecutor] User format preference: {user_format}")

        # Route to appropriate environment
        if decision.environment == "gpu":
            return self._execute_on_gpu(code, session_id, user_format, decision)
        else:
            return self._execute_on_cpu(code, session_id, decision)

    def _get_dataset_info(self, session_id: str) -> Tuple[int, int]:
        """Get dataset size from session store"""
        try:
            import builtins

            session_store = getattr(builtins, "_session_store", None)

            if session_store and session_id in session_store:
                df = session_store[session_id].get("dataframe")
                if df is not None:
                    return len(df), len(df.columns)

                # Fallback to data_profile
                data_profile = session_store[session_id].get("data_profile")
                if data_profile:
                    return data_profile.row_count, len(data_profile.ai_agent_context["column_details"])

        except Exception as e:
            logger.warning(f"[TrainingExecutor] Could not get dataset info: {e}")

        # Default for unknown
        return 1000, 10

    def _execute_on_cpu(self, code: str, session_id: str, decision) -> Dict[str, Any]:
        """Execute on E2B sandbox (CPU)"""
        logger.info("[TrainingExecutor] Executing on CPU (E2B sandbox)")

        try:
            import importlib.util

            parent_dir = os.path.join(os.path.dirname(__file__), "..")
            tools_file_path = os.path.join(parent_dir, "tools.py")
            spec = importlib.util.spec_from_file_location("tools_module", tools_file_path)
            tools_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tools_module)

            execute_python_in_sandbox = tools_module.execute_python_in_sandbox
            get_sandbox = tools_module.get_sandbox

            result = execute_python_in_sandbox(code, session_id)
            result["execution_environment"] = "cpu"
            result["decision_reasoning"] = decision.reasoning

            model_files = result.get("models", [])
            if model_files:
                result["model_files"] = model_files
                logger.info(f"[TrainingExecutor] Extracted {len(model_files)} model file(s)")

            return result

        except Exception as e:
            logger.exception("[TrainingExecutor] CPU execution error")
            return {"success": False, "stderr": f"CPU execution failed: {str(e)}", "execution_environment": "cpu"}

    def _execute_on_gpu(self, code: str, session_id: str, user_format: Optional[str], decision) -> Dict[str, Any]:
        """Execute on GPU (Azure primary, AWS fallback)"""
        from .quota_tracker import quota_tracker

        # Step 1: Pre-check Azure quota
        azure_has_quota = quota_tracker.has_available_quota("azure")

        if azure_has_quota:
            logger.info("[TrainingExecutor] Azure quota available, attempting Azure ML")
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
                from tools import azure_gpu_train

                result_str = azure_gpu_train(code, session_id, user_format)
                success = "successfully" in result_str.lower()

                if success:
                    model_files = self._extract_model_files_from_result(result_str, session_id)
                    return {
                        "success": True,
                        "stdout": result_str,
                        "stderr": "",
                        "execution_environment": "gpu_azure",
                        "decision_reasoning": decision.reasoning,
                        "plots": [],
                        "model_files": model_files,
                    }
                else:
                    # Azure job failed
                    raise Exception(result_str)

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[TrainingExecutor] Azure GPU failed: {error_msg}")

                # Check if quota-related error
                if any(
                    keyword in error_msg.lower() for keyword in ["quota", "limit exceeded", "resourcequotaexceeded"]
                ):
                    logger.info("[TrainingExecutor] Quota error detected, falling back to AWS")
                    return self._fallback_to_aws(code, session_id, user_format, decision)
                else:
                    # Non-quota error, still try AWS
                    logger.info("[TrainingExecutor] Non-quota error, attempting AWS fallback")
                    return self._fallback_to_aws(code, session_id, user_format, decision)
        else:
            # Azure quota exhausted, go straight to AWS
            logger.info("[TrainingExecutor] Azure quota exhausted, using AWS directly")
            return self._fallback_to_aws(code, session_id, user_format, decision)

    def _fallback_to_aws(self, code: str, session_id: str, user_format: Optional[str], decision) -> Dict[str, Any]:
        """Fallback to AWS SageMaker"""
        from .quota_tracker import quota_tracker

        # Check AWS quota
        aws_has_quota = quota_tracker.has_available_quota("aws")

        if not aws_has_quota:
            logger.info("[TrainingExecutor] AWS quota also exhausted, falling back to CPU")
            return self._execute_on_cpu(code, session_id, decision)

        logger.info("[TrainingExecutor] Executing on AWS SageMaker")
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
            from tools import aws_gpu_train

            result_str = aws_gpu_train(code, session_id, user_format)
            success = "successfully" in result_str.lower()

            if success:
                # Store fallback info in session
                import builtins

                if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
                    builtins._session_store[session_id]["used_service"] = "AWS (Azure fallback)"

                model_files = self._extract_model_files_from_result(result_str, session_id)
                return {
                    "success": True,
                    "stdout": result_str,
                    "stderr": "",
                    "execution_environment": "gpu_aws",
                    "decision_reasoning": decision.reasoning + " (Azure→AWS fallback)",
                    "plots": [],
                    "model_files": model_files,
                }
            else:
                raise Exception(result_str)

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"[TrainingExecutor] AWS GPU also failed: {error_msg}")
            logger.info("[TrainingExecutor] Final fallback to CPU execution")
            return self._execute_on_cpu(code, session_id, decision)

    def _extract_model_files_from_result(self, result_str: str, session_id: str) -> List[str]:
        """Extract model files from training result"""
        model_files = []
        # Parse result string for model file paths
        if "local:" in result_str:
            import re

            match = re.search(r"local:\s*([^\)]+)", result_str)
            if match:
                model_files.append(match.group(1).strip())
        return model_files


# Global instance
training_executor = TrainingExecutor()


def should_use_training_executor(code: str) -> bool:
    """
    Detect if code is a training task that should use the executor

    Returns True if code contains model training patterns
    """
    training_patterns = [
        ".fit(",
        "model.train(",
        "train_test_split",
        "GridSearchCV",
        "RandomizedSearchCV",
        "cross_val_score",
        "Pipeline",
    ]

    return any(pattern in code for pattern in training_patterns)
