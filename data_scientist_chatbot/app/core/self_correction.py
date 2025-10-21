"""LLM-driven self-correction for code execution failures"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


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

            fix_prompt = self._build_fix_prompt(attempts, context)
            from langchain_core.messages import HumanMessage

            fixed_response = llm_agent.invoke([HumanMessage(content=fix_prompt)])

            diagnosis, fixed_code = self._extract_fix(fixed_response.content)
            attempt.diagnosis = diagnosis

            if not fixed_code or fixed_code == current_code:
                logger.warning("LLM couldn't generate different code, stopping")
                break

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

Respond with:
1. DIAGNOSIS: One sentence explaining the root cause
2. FIXED_CODE: The corrected Python code

Format:
DIAGNOSIS: [your analysis]

FIXED_CODE:
```python
[corrected code]
```
"""

    def _extract_fix(self, response: str) -> tuple[str, str]:
        import re

        diagnosis_match = re.search(r"DIAGNOSIS:\s*(.+?)(?=\n\n|FIXED_CODE:|$)", response, re.DOTALL)
        diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else "No diagnosis"

        code_match = re.search(r"FIXED_CODE:\s*```python\s*(.+?)\s*```", response, re.DOTALL)
        if not code_match:
            code_match = re.search(r"```python\s*(.+?)\s*```", response, re.DOTALL)

        fixed_code = code_match.group(1).strip() if code_match else ""

        return diagnosis, fixed_code

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
