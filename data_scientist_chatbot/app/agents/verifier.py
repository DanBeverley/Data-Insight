from typing import Dict, Any, List, Optional
import json
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from data_scientist_chatbot.app.core.state import GlobalState
from data_scientist_chatbot.app.core.agent_factory import create_verifier_agent
from data_scientist_chatbot.app.prompts import get_verifier_prompt
from data_scientist_chatbot.app.core.logger import logger


class VerifierDecision(BaseModel):
    approved: bool = Field(description="Whether the task was completed successfully")
    feedback: str = Field(description="Brief explanation of the decision")
    missing_items: List[str] = Field(default=[], description="List of missing deliverables")
    existing_items: List[str] = Field(default=[], description="List of delivered items")


def _save_analysis_to_rag(state: GlobalState, task: Dict, insights: List, artifacts: List):
    session_id = state.get("session_id")
    if not session_id:
        return
    try:
        from data_scientist_chatbot.app.utils.knowledge_store import get_knowledge_store

        store = get_knowledge_store(session_id)

        summary_parts = [f"## Analysis: {task.get('description', 'Unknown')[:100]}"]
        if insights:
            summary_parts.append("\n### Key Findings")
            for i, insight in enumerate(insights[:10], 1):
                label = insight.get("label", "") if isinstance(insight, dict) else str(insight)
                summary_parts.append(f"{i}. {label}")
        if artifacts:
            summary_parts.append("\n### Generated Artifacts")
            for a in artifacts[:10]:
                name = a.get("filename", "") if isinstance(a, dict) else str(a)
                summary_parts.append(f"- {name}")

        summary = "\n".join(summary_parts)
        store.add_document(summary, source="analysis", source_name=f"Analysis: {task.get('description', '')[:50]}")
        logger.info("[VERIFIER] Saved analysis summary to RAG")
    except Exception as e:
        logger.warning(f"[VERIFIER] Failed to save to RAG: {e}")


def run_verifier_agent(state: GlobalState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("[VERIFIER] Validating task execution...")

    artifacts = state.get("artifacts") or []
    agent_insights = state.get("agent_insights") or []
    logger.info(f"[VERIFIER] Input: {len(artifacts)} artifacts, {len(agent_insights)} insights")

    current_task_desc = state.get("current_task_description")
    if current_task_desc:
        current_task = {
            "id": "delegated_task",
            "description": current_task_desc,
            "assigned_to": "hands",
            "status": "in_progress",
        }
    else:
        messages = state.get("messages", [])
        last_instruction = "Unknown Task"
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                last_instruction = msg.content
                break
        current_task = {
            "id": "dynamic_task",
            "description": last_instruction,
            "assigned_to": "hands",
            "status": "in_progress",
        }

    logger.info(f"[VERIFIER] Task: {current_task['description'][:100]}...")

    execution_result = state.get("execution_result") or {}
    execution_output = execution_result.get("stdout", "")

    from .verification_engine import build_structured_verification_input, stage1_programmatic_check

    structured_input = build_structured_verification_input(
        task_description=current_task["description"],
        execution_output=execution_output,
        artifacts=artifacts,
        agent_insights=agent_insights,
    )

    stage1_result = stage1_programmatic_check(structured_input)
    logger.info(
        f"[VERIFIER] Stage 1: {'PASSED' if stage1_result['stage1_passed'] else 'FAILED'} | {stage1_result['feedback']}"
    )

    if not stage1_result["stage1_passed"]:
        approved = False
        feedback = stage1_result["feedback"]
        missing_items = [f["check"] for f in stage1_result["failures"]]
        existing_items = stage1_result["passes"]
    else:
        verifier = create_verifier_agent()
        prompt = get_verifier_prompt()
        chain = prompt | verifier

        from langchain_core.messages import HumanMessage

        structured_summary = f"""VERIFIED FACTS (programmatic - TRUE):
- df.info() present: {structured_input['execution']['df_info_present']}
- Artifacts: {structured_input['artifacts']['count']} ({', '.join(structured_input['artifacts']['names'][:5])})
- Insights: {structured_input['insights']['count']} ({', '.join(structured_input['insights']['labels'][:5])})
- Errors: {structured_input['execution']['has_errors']}

REQUIREMENTS:
- Visualization needed: {structured_input['task_requirements']['requires_visualization']}
- Model needed: {structured_input['task_requirements']['requires_model']}"""

        invoke_params = {
            "task_description": current_task["description"],
            "execution_output": structured_summary,
            "artifacts": json.dumps(structured_input["artifacts"]),
            "agent_insights": json.dumps(structured_input["insights"]),
            "messages": [HumanMessage(content="Verify task completion. Output ONLY JSON.")],
        }

        logger.info("[VERIFIER] Stage 2: LLM verification")

        try:
            result = chain.invoke(invoke_params)
            content = result.content if hasattr(result, "content") and result.content else str(result)
            logger.info(f"[VERIFIER] LLM Response: {content[:200]}")

            json_str = _extract_json_object(content)
            if json_str and '"approved"' in json_str:
                try:
                    decision = VerifierDecision.parse_raw(json_str)
                    approved = decision.approved
                    feedback = decision.feedback
                    missing_items = decision.missing_items
                    existing_items = decision.existing_items
                except Exception:
                    raw_data = json.loads(json_str)
                    approved = raw_data.get("approved", False)
                    feedback = raw_data.get("feedback", "Parse error - defaulting to reject")
                    missing_items = raw_data.get("missing_items", ["unknown"])
                    existing_items = raw_data.get("existing_items", stage1_result["passes"])
            else:
                logger.warning("[VERIFIER] No valid JSON in LLM response - rejecting")
                approved = False
                feedback = "LLM response invalid - cannot verify completion"
                missing_items = ["valid_verification"]
                existing_items = stage1_result["passes"]

            if approved:
                requires_viz = structured_input["task_requirements"]["requires_visualization"]
                requires_model = structured_input["task_requirements"]["requires_model"]
                has_viz = structured_input["artifacts"]["has_visualizations"]
                has_model = structured_input["artifacts"]["has_models"]
                artifact_count = structured_input["artifacts"]["count"]

                if requires_viz and not has_viz:
                    logger.warning("[VERIFIER] Post-check override: Viz required but none found")
                    approved = False
                    feedback = "Task requires visualizations but none were generated"
                    missing_items = ["visualizations"]
                elif requires_model and not has_model:
                    logger.warning("[VERIFIER] Post-check override: Model required but none saved")
                    approved = False
                    feedback = "Task requires model but none was saved"
                    missing_items = ["model"]
                elif requires_viz and artifact_count < 3:
                    logger.warning(f"[VERIFIER] Post-check override: Only {artifact_count} artifacts for viz task")
                    approved = False
                    feedback = f"Task needs multiple visualizations but only {artifact_count} artifacts generated"
                    missing_items = ["more_visualizations"]

        except Exception as e:
            logger.error(f"[VERIFIER] Stage 2 error: {e}")
            approved = False
            feedback = f"Verification error: {str(e)[:50]}"
            missing_items = ["verification_error"]
            existing_items = stage1_result["passes"]

    logger.info(f"[VERIFIER] Decision: {'APPROVED' if approved else 'REJECTED'}. {feedback}")

    from langchain_core.messages import AIMessage

    decision_json = json.dumps(
        {"approved": approved, "feedback": feedback, "missing_items": missing_items, "existing_items": existing_items}
    )

    if approved:
        _save_analysis_to_rag(state, current_task, agent_insights, artifacts)

        # Sync insights to session store for API access (Key Insights Summary panel)
        session_id = state.get("session_id")
        if session_id and agent_insights:
            import builtins

            if not hasattr(builtins, "_session_store"):
                builtins._session_store = {}
            if session_id not in builtins._session_store:
                builtins._session_store[session_id] = {}
            builtins._session_store[session_id]["agent_insights"] = agent_insights
            logger.info(f"[VERIFIER] Synced {len(agent_insights)} insights to session store")

            # Persist insights to disk for survival across restarts
            from src.api_utils.artifact_tracker import get_artifact_tracker

            tracker = get_artifact_tracker()
            tracker.save_insights(session_id, agent_insights)
            logger.info(f"[VERIFIER] Persisted {len(agent_insights)} insights to disk")

        # Create synthesis ToolMessage - gives Brain a proper turn to respond
        from langchain_core.messages import HumanMessage

        def get_cat(a):
            return a.get("category") if isinstance(a, dict) else getattr(a, "category", None)

        def get_fn(a):
            return a.get("filename") if isinstance(a, dict) else getattr(a, "filename", "")

        viz_artifacts = [a for a in artifacts if get_cat(a) == "visualization"]
        model_artifacts = [a for a in artifacts if get_cat(a) == "model"]

        artifact_list = []
        for a in viz_artifacts[:10]:
            fn = get_fn(a)
            if fn:
                artifact_list.append(f"- {fn}")

        synthesis_content = f"""Present the analysis results now. 

Available visualizations ({len(viz_artifacts)}):
{chr(10).join(artifact_list) if artifact_list else 'None'}

Embed each visualization using: ![Chart Title](/static/plots/filename.png)
Provide insights and interpretation for each chart."""

        synthesis_msg = HumanMessage(content=synthesis_content, additional_kwargs={"synthesis_request": True})

        return {
            "messages": [synthesis_msg],
            "current_agent": "verifier",
            "agent_insights": agent_insights,
            "artifacts": artifacts,
            "retry_count": state.get("retry_count", 0),
            "workflow_stage": "verification_passed",
        }
    else:
        return {
            "messages": [AIMessage(content=decision_json, additional_kwargs={"internal": True})],
            "current_agent": "verifier",
            "agent_insights": agent_insights,
            "artifacts": artifacts,
            "retry_count": state.get("retry_count", 0) + 1,
            "workflow_stage": "verification_failed",
        }


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
