"""LLM-driven self-correction for code execution failures"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

try:
    from .logger import logger
except ImportError:
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from logger import logger


@dataclass
class ExecutionAttempt:
    attempt_num: int
    code: str
    success: bool
    output: str
    error: str
    diagnosis: str = ""


class SelfCorrectingExecutor:
    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def execute_with_learning(self, initial_code: str, session_id: str, context: str, llm_agent) -> Dict[str, Any]:
        attempts: List[ExecutionAttempt] = []

        current_code = initial_code

        for attempt_num in range(1, self.max_attempts + 1):
            import sys
            import os

            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from tools import execute_python_in_sandbox

            result = execute_python_in_sandbox(current_code, session_id)

            attempt = ExecutionAttempt(
                attempt_num=attempt_num,
                code=current_code,
                success=result.get("success", False),
                output=result.get("stdout", ""),
                error=result.get("stderr", ""),
            )
            attempts.append(attempt)

            if attempt.success:
                logger.info(f"Success on attempt {attempt_num}")
                return self._format_success(result, attempts)

            if attempt_num == self.max_attempts:
                logger.warning(f"Failed after {self.max_attempts} attempts")
                break

            logger.info(f"Attempt {attempt_num} failed, asking LLM to fix")
            logger.error(f"E2B execution failed: {attempt.error}")

            fix_prompt = self._build_fix_prompt(attempts, context)
            from langchain_core.messages import HumanMessage

            fixed_response = llm_agent.invoke([HumanMessage(content=fix_prompt)])
            logger.debug(f"LLM response received, length: {len(fixed_response.content)}")

            try:
                diagnosis, fixed_code = self._extract_fix(fixed_response.content)
                attempt.diagnosis = diagnosis
            except Exception as e:
                logger.error(f"Failed to extract fix from LLM response: {e}")
                logger.debug(f"LLM response was: {fixed_response.content[:500]}")
                break

            if not fixed_code:
                logger.warning("LLM couldn't extract fixed code from response, stopping")
                logger.debug(f"LLM response was: {fixed_response.content[:500]}")
                break

            if fixed_code == current_code:
                logger.warning("LLM returned identical code, no fix made")
                logger.debug(f"Code unchanged after correction attempt {attempt_num}")
                break

            logger.info(f"Code changed, trying corrected version (attempt {attempt_num + 1})")
            current_code = fixed_code
            logger.info(f"LLM diagnosis: {diagnosis[:100]}...")

        return self._format_failure(attempts)

    def _build_fix_prompt(self, attempts: List[ExecutionAttempt], context: str) -> str:
        last_attempt = attempts[-1]

        failure_history = ""
        if len(attempts) > 1:
            failure_history = "\n\nPREVIOUS FAILED ATTEMPTS:\n"
            for att in attempts[:-1]:
                failure_history += (
                    f"\nAttempt {att.attempt_num}:\n```python\n{att.code}\n```\nError: {att.error[:200]}\n"
                )

        return f"""Your code execution failed. Analyze the error and fix it.

{context}

FAILED CODE:
```python
{last_attempt.code}
```

ERROR OUTPUT:
{last_attempt.error}
{failure_history}

Respond ONLY with valid JSON (no additional text):
{{
  "diagnosis": "one sentence explaining the root cause",
  "fixed_code": "the complete corrected Python code"
}}
"""

    def _extract_fix(self, response: str) -> Tuple[str, str]:
        import json

        response = response.strip()

        json_start = response.find("{")
        json_end = response.rfind("}")

        if json_start == -1 or json_end == -1 or json_start >= json_end:
            logger.error("No JSON object found in LLM response")
            return "No JSON in response", ""

        json_str = response[json_start : json_end + 1]

        try:
            data = json.loads(json_str)
            diagnosis = data.get("diagnosis", "").strip()
            fixed_code = data.get("fixed_code", "").strip()

            if not diagnosis:
                diagnosis = "No diagnosis provided"

            return diagnosis, fixed_code

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.debug(f"Attempted to parse: {json_str[:200]}")
            return "Invalid JSON format", ""

    def _format_success(self, result: Dict, attempts: List[ExecutionAttempt]) -> Dict:
        retry_info = ""
        if len(attempts) > 1:
            retry_info = f"\n\n[Self-corrected after {len(attempts)} attempts]"
            for att in attempts[:-1]:
                retry_info += f"\n  Attempt {att.attempt_num}: {att.diagnosis}"

        final_result = {
            **result,
            "self_corrected": len(attempts) > 1,
            "attempts": len(attempts),
            "stdout": result.get("stdout", "") + retry_info,
            "final_code": attempts[-1].code if attempts else "",
        }

        return final_result

    def _format_failure(self, attempts: List[ExecutionAttempt]) -> Dict:
        last_attempt = attempts[-1]

        failure_summary = f"Failed after {len(attempts)} attempts:\n"
        for att in attempts:
            failure_summary += f"\nAttempt {att.attempt_num}: {att.diagnosis or 'Initial attempt'}\n{att.error[:150]}"

        return {
            "success": False,
            "stderr": failure_summary,
            "self_corrected": False,
            "attempts": len(attempts),
            "plots": [],
            "models": [],
        }
