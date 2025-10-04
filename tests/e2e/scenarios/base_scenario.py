from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioStep:
    name: str
    action: str
    expected_outcome: str
    timeout: int = 30
    retry_count: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ScenarioResult:
    scenario_name: str
    passed: bool
    duration: float
    steps_completed: int
    total_steps: int
    errors: List[str]
    metadata: Dict[str, Any]


class BaseScenario(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[ScenarioStep] = []
        self.errors: List[str] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @abstractmethod
    def setup(self) -> None:
        """Prepare test environment and resources"""
        pass

    @abstractmethod
    def define_steps(self) -> List[ScenarioStep]:
        """Define scenario steps in execution order"""
        pass

    @abstractmethod
    def teardown(self) -> None:
        """Clean up test resources"""
        pass

    def execute(self) -> ScenarioResult:
        logger.info(f"Starting scenario: {self.name}")
        self.start_time = time.time()
        steps_completed = 0

        try:
            self.setup()
            self.steps = self.define_steps()

            for idx, step in enumerate(self.steps, 1):
                logger.info(f"Step {idx}/{len(self.steps)}: {step.name}")

                try:
                    success = self._execute_step(step)
                    if success:
                        steps_completed += 1
                        logger.info(f"✓ Step {idx} completed: {step.name}")
                    else:
                        error_msg = f"Step {idx} failed: {step.name}"
                        self.errors.append(error_msg)
                        logger.error(error_msg)

                        if step.retry_count == 0:
                            break

                except Exception as e:
                    error_msg = f"Step {idx} exception: {step.name} - {str(e)}"
                    self.errors.append(error_msg)
                    logger.error(error_msg)
                    break

        finally:
            self.teardown()
            self.end_time = time.time()

        duration = self.end_time - self.start_time
        passed = steps_completed == len(self.steps) and len(self.errors) == 0

        return ScenarioResult(
            scenario_name=self.name,
            passed=passed,
            duration=duration,
            steps_completed=steps_completed,
            total_steps=len(self.steps),
            errors=self.errors,
            metadata=self.get_metadata()
        )

    @abstractmethod
    def _execute_step(self, step: ScenarioStep) -> bool:
        """Execute a single scenario step"""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return scenario metadata for reporting"""
        return {
            "description": self.description,
            "start_time": self.start_time,
            "end_time": self.end_time
        }

    def validate_outcome(self, expected: str, actual: str) -> bool:
        """Validate step outcome against expectations"""
        return expected.lower() in actual.lower()

    def log_step_result(self, step_name: str, success: bool, details: str = ""):
        """Log step execution result"""
        status = "✓" if success else "✗"
        logger.info(f"{status} {step_name}: {details}")
