"""Hands agent - Executes code and generates artifacts."""

import re
import json
from typing import Dict, Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import traceable

from data_scientist_chatbot.app.core.state import GlobalState
from data_scientist_chatbot.app.core.agent_factory import create_hands_agent
from data_scientist_chatbot.app.core.logger import logger
from data_scientist_chatbot.app.core.training_decision import TrainingDecisionEngine
from data_scientist_chatbot.app.prompts import get_hands_prompt
from data_scientist_chatbot.app.utils.context import get_data_context
from data_scientist_chatbot.app.utils.text_processing import parse_message_to_tool_call, extract_format_from_request
from data_scientist_chatbot.app.utils.semantic_matcher import get_semantic_matcher
from data_scientist_chatbot.app.context_manager import get_session_memory, record_execution
from data_scientist_chatbot.app.tools import execute_python_in_sandbox
from data_scientist_chatbot.app.tools.tool_definitions import python_code_interpreter
from src.api_utils.session_management import session_data_manager
from src.api_utils.artifact_tracker import get_artifact_tracker
from src.learning.adaptive_system import AdaptiveLearningSystem


def _get_task_description(state: GlobalState) -> str:
    """Extract task description from state."""
    delegated_task = state.get("current_task_description")
    if delegated_task:
        logger.info(f"[HANDS] FULL DELEGATED TASK: {delegated_task}")
        return delegated_task

    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content"):
            return last_message.content

    return "Perform data analysis task"


def _build_retry_context(
    feedback_data: Dict, previous_execution: Dict, existing_artifacts: List, agent_insights: List
) -> Optional[str]:
    """Build targeted task description for retry attempts."""
    if feedback_data.get("approved", True):
        return None

    feedback_msg = feedback_data.get("feedback", "Unknown feedback")
    logger.info(f"[HANDS] Detected rejection feedback: {feedback_msg}")

    execution_context_parts = []

    prev_stdout = previous_execution.get("stdout", "")
    if prev_stdout:
        stdout_preview = prev_stdout[:4000] if len(prev_stdout) > 4000 else prev_stdout
        execution_context_parts.append(f"**YOUR PREVIOUS EXECUTION OUTPUT:**\n{stdout_preview}")

    if existing_artifacts:
        artifact_details = []
        for artifact in existing_artifacts:
            if isinstance(artifact, dict):
                fname = artifact.get("filename", "unknown")
                category = artifact.get("category", "unknown")
                artifact_details.append(f"- {fname} ({category})")
            else:
                artifact_details.append(f"- {getattr(artifact, 'filename', str(artifact))}")
        execution_context_parts.append(
            f"**ARTIFACTS CREATED ({len(existing_artifacts)} total):**\n" + "\n".join(artifact_details)
        )
    else:
        execution_context_parts.append("**ARTIFACTS CREATED:** None")

    if agent_insights:
        insights_summary = []
        for insight in agent_insights[:10]:
            if isinstance(insight, dict):
                label = insight.get("label", "")
                value = str(insight.get("value", ""))[:200]
                insights_summary.append(f"- {label}: {value}")
            else:
                insights_summary.append(f"- {str(insight)[:200]}")
        execution_context_parts.append(f"**YOUR PREVIOUS INSIGHTS/FINDINGS:**\n" + "\n".join(insights_summary))

    missing_items = feedback_data.get("missing_items", [])
    execution_context_parts.append(f"**VERIFIER FEEDBACK:**\n{feedback_msg}")
    execution_context = "\n\n".join(execution_context_parts)

    is_execution_error = "execution_errors" in missing_items or "Code execution had errors" in feedback_msg

    if is_execution_error:
        return f"""**YOUR PREVIOUS CODE FAILED WITH AN ERROR.**

{execution_context}

**YOUR TASK:**
1. Analyze the error traceback above - identify the exact line, operation, and root cause
2. Understand WHY it failed (data issue, wrong column name, type mismatch, missing value, etc.)
3. Write CORRECTED code that handles this error case properly
4. Use defensive coding: add appropriate error handling, null checks, or type conversions as needed
5. The {len(existing_artifacts)} existing artifacts are complete - only fix the failing code section

Write the corrected code:"""
    else:
        return f"""Retrying...
{execution_context}

{len(existing_artifacts)} artifacts already exist and are COMPLETE - DO NOT regenerate them.
Only output what's missing:
- If "df_info" is missing: Run `df.info()`, `df.describe()`, and print the shape
- If "insights" is missing: Print the PROFILING_INSIGHTS JSON block
- If a specific plot is missing: Generate ONLY that plot

Write minimal code that ONLY fixes the missing item(s)."""


def _get_data_schema(session_id: str) -> str:
    """Extract data schema from session."""
    try:
        session_data = session_data_manager.get_session(session_id)
        if session_data:
            df = session_data.get("dataframe")
            if df is not None:
                schema_parts = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    sample_vals = df[col].dropna().head(3).tolist()
                    sample_str = ", ".join([str(v)[:20] for v in sample_vals])
                    schema_parts.append(f"- {col} ({dtype}): e.g. {sample_str}")
                return "Columns:\n" + "\n".join(schema_parts)
    except Exception as e:
        logger.debug(f"[HANDS] Schema extraction skipped: {e}")
    return "Schema not available"


def _parse_artifacts_from_output(stdout: str) -> List[Dict[str, Any]]:
    """Parse artifact references from execution output."""
    artifacts = []
    for line in stdout.split("\n"):
        if "PLOT_SAVED:" in line:
            fname = line.split(":", 1)[1].strip()
            # Clean path prefixes to extract just the filename
            if "/static/plots/" in fname:
                fname = fname.split("/static/plots/")[-1]
            if "static/plots/" in fname:
                fname = fname.split("static/plots/")[-1]
            if "/" in fname:
                fname = fname.split("/")[-1]
            artifacts.append({"filename": fname, "category": "visualization", "local_path": f"/static/plots/{fname}"})
        elif "MODEL_SAVED:" in line:
            fname = line.split(":", 1)[1].strip()
            if "/" in fname:
                fname = fname.split("/")[-1]
            artifacts.append({"filename": fname, "category": "model", "local_path": f"/static/models/{fname}"})
    return artifacts


def _sync_artifacts_from_tracker(session_id: str, existing_artifacts: List[Dict]) -> List[Dict]:
    """Sync artifacts from tracker to ensure completeness."""
    try:
        tracker = get_artifact_tracker()
        tracker_result = tracker.get_session_artifacts(session_id)
        tracked = tracker_result.get("artifacts", []) if isinstance(tracker_result, dict) else []
        for artifact in tracked:
            filename = artifact.get("filename", "")
            if "/" in filename:
                filename = filename.split("/")[-1]
            if filename and not any(a.get("filename") == filename for a in existing_artifacts):
                existing_artifacts.append(
                    {
                        "filename": filename,
                        "category": artifact.get("category", "visualization"),
                        "local_path": f"/static/plots/{filename}",
                    }
                )
    except Exception as e:
        logger.warning(f"[HANDS] Could not sync artifacts from tracker: {e}")
    return existing_artifacts


@traceable(name="hands_execution", tags=["agent", "code"])
def run_hands_agent(state: GlobalState) -> Dict[str, Any]:
    """Execute Hands agent to run code and generate artifacts."""
    session_id = state.get("session_id")

    enhanced_state = state.copy()
    last_sequence = enhanced_state.get("last_agent_sequence") or []
    enhanced_state["last_agent_sequence"] = last_sequence + ["hands"]

    retry_count = enhanced_state.get("retry_count") or 0
    if last_sequence and len(last_sequence) >= 4:
        recent_sequence = last_sequence[-4:]
        if recent_sequence == ["brain", "hands", "brain", "hands"]:
            enhanced_state["retry_count"] = retry_count + 1

    task_description = _get_task_description(state)
    existing_artifacts = list(state.get("artifacts") or [])
    previous_execution = state.get("execution_result") or {}
    agent_insights = list(state.get("agent_insights") or [])

    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        content = str(last_msg.content)
        if "approved" in content and "feedback" in content:
            try:
                clean_content = content.replace("```json", "").replace("```", "").strip()
                feedback_data = json.loads(clean_content)
                retry_task = _build_retry_context(feedback_data, previous_execution, existing_artifacts, agent_insights)
                if retry_task:
                    task_description = retry_task
            except Exception as e:
                logger.warning(f"[HANDS] Failed to parse feedback JSON: {e}")

    data_context = get_data_context(session_id, query=task_description)

    try:
        session_data = session_data_manager.get_session(session_id)
        if session_data:
            from data_scientist_chatbot.app.tools import refresh_sandbox_data

            # Sync ALL datasets if available (multi-dataset support)
            datasets_dict = session_data.get("datasets", {})
            if datasets_dict:
                synced_count = 0
                for filename, df in datasets_dict.items():
                    if df is not None:
                        refresh_sandbox_data(session_id, df, filename)
                        synced_count += 1
                if synced_count > 0:
                    logger.info(f"[HANDS] Synced {synced_count} datasets to sandbox")
            elif session_data.get("dataframe") is not None:
                # Fallback: single dataframe (backward compat)
                expected_df = session_data["dataframe"]
                filename = session_data.get("filename")
                refresh_sandbox_data(session_id, expected_df, filename)
                logger.info(f"[HANDS] Sandbox refreshed with {expected_df.shape[0]} rows, {expected_df.shape[1]} cols")
    except Exception as e:
        logger.warning(f"[HANDS] Sandbox refresh check failed: {e}")

    logger.info(f"[HANDS] Executing: {task_description[:100]}...")

    is_training_task = any(
        keyword in task_description.lower()
        for keyword in ["train", "model", "fit", "predict", "classify", "regression", "cluster"]
    )

    dataset_rows = 0
    feature_count = 0
    try:
        session_data = session_data_manager.get_session(session_id)
        if session_data:
            df = session_data.get("dataframe")
            if df is not None:
                dataset_rows = len(df)
                feature_count = len(df.columns)
            elif session_data.get("data_profile"):
                insights = session_data["data_profile"].dataset_insights
                dataset_rows = insights.total_records
                feature_count = insights.total_features
    except Exception as e:
        logger.warning(f"[HANDS] Could not retrieve dataset stats: {e}")

    environment_context = ""
    if is_training_task:
        decision_engine = TrainingDecisionEngine()
        decision = decision_engine.decide(
            dataset_rows=dataset_rows, feature_count=feature_count, model_type="", code=None
        )
        logger.info(f"[HANDS] Training decision: {decision.environment.upper()} ({decision.reasoning})")
        if decision.environment == "gpu":
            environment_context = f"\n**EXECUTION ENVIRONMENT: GPU (Azure ML / AWS SageMaker)**\nDecision reasoning: {decision.reasoning}\n"
        else:
            environment_context = f"\n**EXECUTION ENVIRONMENT: CPU (E2B Sandbox)**\nDataset is pre-loaded as `df` variable - use it directly.\nDecision reasoning: {decision.reasoning}\n"

    detected_format = extract_format_from_request(task_description)
    format_context = ""
    if detected_format:
        format_hints = {
            "onnx": "Use torch.onnx.export() or skl2onnx for ONNX format",
            "joblib": "Use joblib.dump() for saving",
            "pickle": "Use pickle.dump() for saving",
        }
        format_hint = format_hints.get(detected_format, f"Save in {detected_format} format")
        format_context = f"\n\n**USER REQUESTED FORMAT:** {detected_format.upper()}\n{format_hint}"

    enhanced_data_context = data_context
    if environment_context:
        enhanced_data_context = f"{data_context}\n{environment_context}"
    if format_context:
        enhanced_data_context = f"{enhanced_data_context}{format_context}"

    try:
        pattern_context = ""
        try:
            matcher = get_semantic_matcher()
            adaptive_system = AdaptiveLearningSystem()
            execution_history = adaptive_system.get_execution_history(success_only=True)
            if execution_history and task_description:
                pattern_context = matcher.find_relevant_patterns(task_description, execution_history, top_k=3)
        except Exception as e:
            logger.debug(f"[HANDS] Pattern retrieval skipped: {e}")

        learning_context = ""
        try:
            memory = get_session_memory(session_id)
            learning_context = memory.get_learning_context()
        except Exception as e:
            logger.debug(f"[HANDS] Learning context skipped: {e}")

        llm = create_hands_agent()
        hands_tools = [python_code_interpreter]
        llm_with_tools = llm.bind_tools(hands_tools)

        prompt = get_hands_prompt()
        agent_runnable = prompt | llm_with_tools

        data_schema = _get_data_schema(session_id)

        hands_state = {
            "messages": [HumanMessage(content=task_description)],
            "data_context": enhanced_data_context,
            "pattern_context": pattern_context,
            "learning_context": learning_context,
            "plan_context": "",
            "data_schema": data_schema,
        }

        task_lower = task_description.lower()
        if any(kw in task_lower for kw in ["train", "model", "fit", "predict", "regression", "classify"]):
            max_turns = 8
        elif any(kw in task_lower for kw in ["comprehensive", "full analysis", "eda", "exploratory"]):
            max_turns = 6
        else:
            max_turns = 5

        logger.info(f"[HANDS] Max turns set to {max_turns} based on task complexity")

        turn_count = 0
        final_response = None
        loop_messages = hands_state["messages"]
        artifacts = list(existing_artifacts)
        execution_summary = ""

        while turn_count < max_turns:
            logger.info(f"[HANDS] Starting Turn {turn_count + 1}/{max_turns}")
            hands_state["messages"] = loop_messages

            llm_response = agent_runnable.invoke(hands_state)
            final_response = llm_response

            parse_message_to_tool_call(llm_response, "hands_direct")

            response_preview = str(llm_response.content)[:500] if llm_response.content else "EMPTY"
            logger.info(f"[HANDS] LLM Response Preview: {response_preview}")

            if not (hasattr(llm_response, "tool_calls") and llm_response.tool_calls):
                logger.info("[HANDS] No tool calls - Agent has completed its thought process.")
                break

            tool_call = llm_response.tool_calls[0]
            if isinstance(tool_call, dict):
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call_{tool_name}")
            else:
                tool_name = tool_call.name
                tool_args = tool_call.args
                tool_id = tool_call.id

            if tool_name == "python_code_interpreter":
                code = tool_args.get("code", "")
                logger.info(f"[HANDS] Executing code ({len(code)} chars)...")
                logger.info(f"[HANDS] CODE PREVIEW (first 2000 chars):\n{code[:2000]}")

                result = execute_python_in_sandbox(code, session_id)

                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")

                new_artifacts = _parse_artifacts_from_output(stdout)
                new_artifact_names = [a["filename"] for a in new_artifacts]

                tool_output_content = ""
                if new_artifact_names:
                    tool_output_content = (
                        f"SUCCESS. Generated {len(new_artifact_names)} artifacts: {', '.join(new_artifact_names)}.\n\n"
                    )
                tool_output_content += f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

                loop_messages.append(llm_response)
                loop_messages.append(ToolMessage(content=tool_output_content, tool_call_id=tool_id))

                artifacts.extend(new_artifacts)
                artifacts = _sync_artifacts_from_tracker(session_id, artifacts)

                execution_summary = tool_output_content

                if "PROFILING_INSIGHTS_END" in stdout:
                    logger.info("[HANDS] Deep discovery completed (termination token found).")
                    break

                record_execution(
                    session_id,
                    {
                        "user_request": f"Turn {turn_count+1}: {task_description[:50]}...",
                        "code": result.get("final_code", code),
                        "success": result.get("success", False),
                        "output": stdout[:500],
                        "error": stderr,
                        "artifacts": result.get("plots", []),
                        "self_corrected": result.get("self_corrected", False),
                        "attempts": result.get("attempts", 1),
                    },
                )

                execution_summary = f"--- Turn {turn_count + 1} ---\n{stdout}\n{stderr}".strip()

                if "PROFILING_INSIGHTS_START" in stdout:
                    logger.info("[HANDS] Final insights detected. Completing Deep Discovery.")
                    break

            turn_count += 1

        logger.info(f"[HANDS] Deep Discovery Loop finished after {turn_count} turns.")

        if execution_summary:
            insights_pattern = r"PROFILING_INSIGHTS_START\s*(.*?)\s*PROFILING_INSIGHTS_END"
            insights_match = re.search(insights_pattern, execution_summary, re.DOTALL)
            if insights_match:
                try:
                    agent_insights = json.loads(insights_match.group(1))
                    logger.info(f"[HANDS] Extracted {len(agent_insights)} insights from final analysis")
                except Exception:
                    pass

        if not agent_insights and artifacts:

            def get_cat(a):
                return a.get("category") if isinstance(a, dict) else getattr(a, "category", None)

            viz_count = len([a for a in artifacts if get_cat(a) == "visualization"])
            model_count = len([a for a in artifacts if get_cat(a) == "model"])
            report_count = len([a for a in artifacts if get_cat(a) == "report"])

            if viz_count > 0:
                agent_insights.append(
                    {
                        "label": "Visualizations Generated",
                        "value": f"{viz_count} charts and plots created for data exploration",
                        "type": "success",
                    }
                )
            if model_count > 0:
                agent_insights.append(
                    {
                        "label": "Models Trained",
                        "value": f"{model_count} predictive models built and saved",
                        "type": "success",
                    }
                )
            if report_count > 0:
                agent_insights.append(
                    {
                        "label": "Interactive Reports",
                        "value": f"{report_count} interactive HTML reports generated",
                        "type": "info",
                    }
                )
            if "correlation" in execution_summary.lower():
                agent_insights.append(
                    {
                        "label": "Correlation Analysis",
                        "value": "Feature correlations analyzed and visualized",
                        "type": "pattern",
                    }
                )
            if "r2" in execution_summary.lower() or "r-squared" in execution_summary.lower():
                agent_insights.append(
                    {
                        "label": "Model Evaluation",
                        "value": "Models evaluated with performance metrics",
                        "type": "metric",
                    }
                )
            logger.info(f"[HANDS] Auto-generated {len(agent_insights)} insights from artifacts")

        final_content = (
            final_response.content if final_response and hasattr(final_response, "content") else "Analysis completed."
        )

        if loop_messages and isinstance(loop_messages[-1], ToolMessage):
            summary_response = AIMessage(
                content=f"Analysis complete. Generated {len(artifacts)} visualizations.",
                additional_kwargs={"internal": True},
            )
        else:
            summary_response = AIMessage(content=final_content, additional_kwargs={"internal": True})

        return {
            "messages": [summary_response],
            "current_agent": "hands",
            "last_agent_sequence": enhanced_state.get("last_agent_sequence") or [],
            "retry_count": enhanced_state.get("retry_count") or 0,
            "artifacts": artifacts,
            "agent_insights": agent_insights,
            "execution_result": {"stdout": execution_summary, "success": True},
            "workflow_stage": "execution_complete",
        }

    except Exception as e:
        logger.error(f"[HANDS] Execution FAILED: {e}")
        import traceback

        logger.error(f"[HANDS] Traceback:\n{traceback.format_exc()}")

        return {
            "messages": [AIMessage(content=f"Hands execution failed: {e}")],
            "current_agent": "hands",
            "last_agent_sequence": enhanced_state.get("last_agent_sequence") or [],
            "retry_count": enhanced_state.get("retry_count") or 0,
            "artifacts": existing_artifacts,
            "agent_insights": agent_insights,
            "execution_result": {"stdout": "", "success": False, "error": str(e)},
            "workflow_stage": "execution_failed",
        }
