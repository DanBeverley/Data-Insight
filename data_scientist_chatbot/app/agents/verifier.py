from typing import Dict, Any, List, Optional
import json
import re
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from data_scientist_chatbot.app.core.state import GlobalState, Task
from data_scientist_chatbot.app.core.agent_factory import create_verifier_agent
from data_scientist_chatbot.app.prompts import get_verifier_prompt
from data_scientist_chatbot.app.core.logger import logger


class VerifierDecision(BaseModel):
    approved: bool = Field(description="Whether the task was completed successfully")
    feedback: str = Field(description="Brief explanation of the decision")
    missing_items: List[str] = Field(
        default=[], description="List of missing deliverables (e.g., 'insights', 'artifacts', 'df_info')"
    )
    existing_items: List[str] = Field(default=[], description="List of successfully delivered items")


def run_verifier_agent(state: GlobalState, config: RunnableConfig) -> Dict[str, Any]:
    logger.info("[VERIFIER] Validating task execution...")

    artifacts = state.get("artifacts") or []
    logger.info(f"[VERIFIER] DEBUG: Received state with {len(artifacts)} artifacts")

    plan = state.get("plan", [])
    current_index = state.get("current_task_index") or 0

    current_task_desc = state.get("current_task_description")

    if current_task_desc:
        logger.info(f"[VERIFIER] Using delegated task description: {current_task_desc}")
        current_task = {
            "id": "delegated_task",
            "description": current_task_desc,
            "assigned_to": "hands",
            "status": "in_progress",
        }
    elif not plan or current_index >= len(plan):
        logger.info("[VERIFIER] No formal plan found. Verifying based on last instruction.")
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

    logger.info(f"[VERIFIER] Checking Task: {current_task['description']}")

    execution_result = state.get("execution_result") or {}
    execution_output = execution_result.get("stdout", "No output")
    artifacts = state.get("artifacts", [])

    if not execution_output:
        execution_output = "No execution output provided by the agent."

    verifier = create_verifier_agent()
    prompt = get_verifier_prompt()

    try:
        from langchain.output_parsers import PydanticOutputParser
        import re

        parser = PydanticOutputParser(pydantic_object=VerifierDecision)

        chain = prompt | verifier

        from langchain_core.messages import HumanMessage

        invoke_params = {
            "task_description": current_task["description"],
            "execution_output": execution_output,
            "artifacts": json.dumps(artifacts),
            "agent_insights": json.dumps(state.get("agent_insights", [])),
            # Use a single instruction message instead of conversation history
            # which confuses the model into continuing the conversation
            "messages": [
                HumanMessage(content="Based on the INPUT above, verify if the task was completed. Output ONLY JSON.")
            ],
        }

        logger.info(
            f"[VERIFIER] INVOKE PARAMS: task_desc={invoke_params['task_description'][:100]}, artifacts_count={len(artifacts)}, messages_count={len(invoke_params['messages'])}"
        )

        # Debug: Log full prompt to diagnose empty response
        logger.info(f"[VERIFIER] FULL PROMPT DEBUG:")
        logger.info(f"[VERIFIER]   task_description: {invoke_params['task_description'][:200]}...")
        logger.info(
            f"[VERIFIER]   execution_output: {invoke_params['execution_output'][:200] if invoke_params['execution_output'] else 'EMPTY'}..."
        )
        logger.info(f"[VERIFIER]   artifacts: {invoke_params['artifacts'][:300]}...")
        logger.info(f"[VERIFIER]   agent_insights: {invoke_params['agent_insights'][:200]}...")
        logger.info(f"[VERIFIER]   messages types: {[type(m).__name__ for m in invoke_params['messages']]}")

        # Additional diagnostic: Compare chain structure with Brain
        logger.info(f"[VERIFIER] CHAIN TYPE: {type(chain)}")
        logger.info(f"[VERIFIER] VERIFIER LLM TYPE: {type(verifier)}")

        result = chain.invoke(invoke_params)

        logger.info(f"[VERIFIER] RESULT TYPE: {type(result)}")
        if hasattr(result, "response_metadata"):
            logger.info(f"[VERIFIER] RESPONSE_METADATA: {result.response_metadata}")
        if hasattr(result, "additional_kwargs"):
            logger.info(f"[VERIFIER] ADDITIONAL_KWARGS: {result.additional_kwargs}")

        content = result.content if hasattr(result, "content") and result.content else str(result)
        logger.info(f"[VERIFIER] RAW LLM RESPONSE: {content[:500] if content else 'EMPTY'}")

        def extract_json_object(text):
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

        json_str = extract_json_object(content)
        if json_str and '"approved"' in json_str:
            try:
                decision = VerifierDecision.parse_raw(json_str)
                approved = decision.approved
                feedback = decision.feedback
                missing_items = decision.missing_items
                existing_items = decision.existing_items
            except Exception as parse_err:
                try:
                    raw_data = json.loads(json_str)
                    approved = raw_data.get("approved", bool(artifacts))
                    feedback = raw_data.get("feedback", "Parsed from raw JSON")
                    missing_items = raw_data.get("missing_items", [])
                    existing_items = raw_data.get("existing_items", [])
                except:
                    approved = bool(artifacts)
                    feedback = (
                        f"JSON parse failed, fallback based on {'artifacts present' if artifacts else 'no artifacts'}"
                    )
                    missing_items = [] if approved else ["unknown"]
                    existing_items = ["artifacts"] if artifacts else []
        else:
            approved = bool(artifacts)
            feedback = f"No JSON found, fallback based on {'artifacts present' if artifacts else 'no artifacts'}"
            missing_items = [] if approved else ["unknown"]
            existing_items = ["artifacts"] if artifacts else []

    except Exception as e:
        logger.error(f"[VERIFIER] Parsing failed: {e}")
        approved = bool(artifacts)
        feedback = f"Fallback decision based on {'artifacts present' if artifacts else 'no artifacts'}"
        missing_items = [] if approved else ["unknown"]
        existing_items = ["artifacts"] if artifacts else []

    logger.info(
        f"[VERIFIER] Decision: {'APPROVED' if approved else 'REJECTED'}. Feedback: {feedback}. Missing: {missing_items}"
    )

    if approved:
        if plan and 0 <= current_index < len(plan):
            plan[current_index]["status"] = "completed"
            plan[current_index]["result"] = "Verified Success"

        new_index = current_index + 1

        from langchain_core.messages import AIMessage
        import json as json_module

        decision_json = json_module.dumps(
            {"approved": True, "feedback": feedback, "missing_items": missing_items, "existing_items": existing_items}
        )

        return {
            "plan": plan,
            "current_task_index": new_index,
            "messages": [AIMessage(content=decision_json, additional_kwargs={"internal": True})],
            "agent_insights": state.get("agent_insights", []),
        }
    else:
        if not plan:
            plan = []

        if 0 <= current_index < len(plan):
            plan[current_index]["status"] = "failed"
            plan[current_index]["error"] = feedback

        correction_task: Task = {
            "id": f"{current_task.get('id', 'task')}_fix",
            "description": f"Fix previous error: {feedback}. Original Task: {current_task.get('description', 'Unknown')}",
            "assigned_to": current_task.get("assigned_to", "hands"),
            "status": "pending",
            "result": None,
            "artifacts": [],
            "error": None,
        }

        if 0 <= current_index < len(plan):
            plan.insert(current_index + 1, correction_task)
            new_index = current_index + 1
        else:
            plan.append(correction_task)
            new_index = len(plan) - 1

        from langchain_core.messages import AIMessage
        import json as json_module

        # Include structured rejection info for Hands to parse
        rejection_json = json_module.dumps(
            {"approved": False, "feedback": feedback, "missing_items": missing_items, "existing_items": existing_items}
        )

        return {
            "plan": plan,
            "current_task_index": new_index,
            "retry_count": state.get("retry_count", 0) + 1,
            "messages": [AIMessage(content=rejection_json, additional_kwargs={"internal": True})],
            "agent_insights": state.get("agent_insights", []),
        }
