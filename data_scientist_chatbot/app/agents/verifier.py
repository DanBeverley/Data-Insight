from typing import Dict, Any, List, Optional
import json
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from data_scientist_chatbot.app.core.state import GlobalState, Task
from data_scientist_chatbot.app.core.agent_factory import create_verifier_agent
from data_scientist_chatbot.app.prompts import get_verifier_prompt
from data_scientist_chatbot.app.core.logger import logger


class VerifierDecision(BaseModel):
    approved: bool = Field(description="Whether the task was completed successfully")
    feedback: str = Field(description="Brief explanation of the decision")
    missing_items: List[str] = Field(default=[], description="List of missing deliverables")
    existing_items: List[str] = Field(default=[], description="List of delivered items")


def run_verifier_agent(state: GlobalState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("[VERIFIER] Validating task execution...")

    artifacts = state.get("artifacts") or []
    agent_insights = state.get("agent_insights", [])
    logger.info(f"[VERIFIER] Input: {len(artifacts)} artifacts, {len(agent_insights)} insights")

    plan = state.get("plan", [])
    current_index = state.get("current_task_index") or 0

    current_task_desc = state.get("current_task_description")

    if current_task_desc:
        current_task = {
            "id": "delegated_task",
            "description": current_task_desc,
            "assigned_to": "hands",
            "status": "in_progress",
        }
    elif not plan or current_index >= len(plan):
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
    else:
        current_task = plan[current_index]

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
                    approved = raw_data.get("approved", True)
                    feedback = raw_data.get("feedback", "Approved")
                    missing_items = raw_data.get("missing_items", [])
                    existing_items = raw_data.get("existing_items", stage1_result["passes"])
            else:
                approved = True
                feedback = "Stage 1 passed"
                missing_items = []
                existing_items = stage1_result["passes"]

            if (
                approved
                and structured_input["artifacts"]["count"] == 0
                and structured_input["task_requirements"]["requires_visualization"]
            ):
                logger.warning("[VERIFIER] Post-check override: No artifacts but viz required")
                approved = False
                feedback = "No visualization artifacts generated"
                missing_items = ["artifacts"]

        except Exception as e:
            logger.error(f"[VERIFIER] Stage 2 error: {e}")
            approved = True
            feedback = "Stage 1 passed, Stage 2 error"
            missing_items = []
            existing_items = stage1_result["passes"]

    logger.info(f"[VERIFIER] Decision: {'APPROVED' if approved else 'REJECTED'}. {feedback}")

    from langchain_core.messages import AIMessage

    decision_json = json.dumps(
        {"approved": approved, "feedback": feedback, "missing_items": missing_items, "existing_items": existing_items}
    )

    if approved:
        if plan and 0 <= current_index < len(plan):
            plan[current_index]["status"] = "completed"
            plan[current_index]["result"] = "Verified"

        return {
            "plan": plan,
            "current_task_index": current_index + 1,
            "messages": [AIMessage(content=decision_json, additional_kwargs={"internal": True})],
            "agent_insights": agent_insights,
        }
    else:
        if plan and 0 <= current_index < len(plan):
            plan[current_index]["status"] = "failed"
            plan[current_index]["error"] = feedback

        correction_task: Task = {
            "id": f"{current_task.get('id', 'task')}_fix",
            "description": f"Fix: {feedback}. Original: {current_task.get('description', '')[:100]}",
            "assigned_to": "hands",
            "status": "pending",
            "result": None,
            "artifacts": [],
            "error": None,
        }

        if plan and 0 <= current_index < len(plan):
            plan.insert(current_index + 1, correction_task)
            new_index = current_index + 1
        else:
            if not plan:
                plan = []
            plan.append(correction_task)
            new_index = len(plan) - 1

        return {
            "plan": plan,
            "current_task_index": new_index,
            "retry_count": state.get("retry_count", 0) + 1,
            "messages": [AIMessage(content=decision_json, additional_kwargs={"internal": True})],
            "agent_insights": agent_insights,
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
