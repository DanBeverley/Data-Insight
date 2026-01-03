import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ExecutionSummary:
    df_info_present: bool = False
    df_shape: Optional[str] = None
    df_describe_present: bool = False
    insights_present: bool = False
    insights_count: int = 0
    insights_labels: List[str] = field(default_factory=list)
    artifacts_count: int = 0
    artifact_names: List[str] = field(default_factory=list)
    artifact_types: List[str] = field(default_factory=list)
    has_errors: bool = False
    error_messages: List[str] = field(default_factory=list)
    execution_successful: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_execution_output(stdout: str) -> ExecutionSummary:
    summary = ExecutionSummary()

    if not stdout:
        summary.execution_successful = False
        return summary

    if "df.shape" in stdout.lower() or re.search(r"Shape:\s*\(\d+,\s*\d+\)", stdout):
        summary.df_info_present = True
        shape_match = re.search(r"\((\d+),\s*(\d+)\)", stdout)
        if shape_match:
            summary.df_shape = f"({shape_match.group(1)}, {shape_match.group(2)})"

    if "<class 'pandas" in stdout or "RangeIndex:" in stdout or "dtypes:" in stdout:
        summary.df_info_present = True

    if "count" in stdout.lower() and "mean" in stdout.lower() and "std" in stdout.lower():
        summary.df_describe_present = True

    insights_match = re.search(r"PROFILING_INSIGHTS_START\s*(.*?)\s*PROFILING_INSIGHTS_END", stdout, re.DOTALL)
    if insights_match:
        summary.insights_present = True
        try:
            insights_json = json.loads(insights_match.group(1).strip())
            if isinstance(insights_json, list):
                summary.insights_count = len(insights_json)
                summary.insights_labels = [i.get("label", "Unknown") for i in insights_json[:10] if isinstance(i, dict)]
        except json.JSONDecodeError:
            pass

    error_patterns = [
        r"Traceback \(most recent call last\)",
        r"Error:",
        r"Exception:",
        r"ModuleNotFoundError:",
        r"NameError:",
        r"TypeError:",
        r"ValueError:",
        r"KeyError:",
        r"AttributeError:",
    ]
    for pattern in error_patterns:
        if re.search(pattern, stdout, re.IGNORECASE):
            summary.has_errors = True
            error_match = re.search(rf"{pattern}.*?(?=\n\n|\Z)", stdout, re.DOTALL | re.IGNORECASE)
            if error_match:
                summary.error_messages.append(error_match.group(0))

    return summary


def parse_artifacts(artifacts: List[Any]) -> ExecutionSummary:
    summary = ExecutionSummary()

    if not artifacts:
        return summary

    summary.artifacts_count = len(artifacts)

    for artifact in artifacts:
        if isinstance(artifact, dict):
            filename = artifact.get("filename", "unknown")
            category = artifact.get("category", "unknown")
        else:
            filename = getattr(artifact, "filename", str(artifact))
            category = getattr(artifact, "category", "unknown")

        summary.artifact_names.append(filename)

        if filename.endswith(".html"):
            summary.artifact_types.append("interactive_chart")
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            summary.artifact_types.append("image")
        elif filename.endswith((".pkl", ".joblib", ".onnx")):
            summary.artifact_types.append("model")
        else:
            summary.artifact_types.append(category)

    return summary


def build_structured_verification_input(
    task_description: str, execution_output: str, artifacts: List[Any], agent_insights: List[Any]
) -> Dict[str, Any]:
    exec_summary = parse_execution_output(execution_output)
    artifact_summary = parse_artifacts(artifacts)

    insights_summary = {
        "count": len(agent_insights) if agent_insights else 0,
        "labels": [],
        "has_insights": bool(agent_insights and len(agent_insights) > 0),
    }
    if agent_insights:
        for insight in agent_insights[:10]:
            if isinstance(insight, dict):
                insights_summary["labels"].append(insight.get("label", "Unknown"))

    task_requirements = {
        "requires_visualization": any(
            kw in task_description.lower()
            for kw in [
                "plot",
                "chart",
                "graph",
                "visualiz",
                "histogram",
                "scatter",
                "bar",
                "line",
                "heatmap",
                "distribution",
            ]
        ),
        "requires_model": any(
            kw in task_description.lower()
            for kw in ["train", "model", "predict", "classify", "regression", "cluster", "fit"]
        ),
        "requires_analysis": any(
            kw in task_description.lower()
            for kw in ["analyz", "analys", "eda", "explor", "profile", "statistic", "correlation"]
        ),
        "requires_insights": True,
    }

    return {
        "task_description": task_description,
        "task_requirements": task_requirements,
        "execution": {
            "df_info_present": exec_summary.df_info_present,
            "df_shape": exec_summary.df_shape,
            "df_describe_present": exec_summary.df_describe_present,
            "insights_in_stdout": exec_summary.insights_present,
            "insights_count_stdout": exec_summary.insights_count,
            "has_errors": exec_summary.has_errors,
            "error_messages": exec_summary.error_messages[:3],
        },
        "artifacts": {
            "count": artifact_summary.artifacts_count,
            "names": artifact_summary.artifact_names,
            "types": artifact_summary.artifact_types,
            "has_visualizations": any(t in ["interactive_chart", "image"] for t in artifact_summary.artifact_types),
            "has_models": any(t == "model" for t in artifact_summary.artifact_types),
        },
        "insights": insights_summary,
    }


def stage1_programmatic_check(structured_input: Dict[str, Any]) -> Dict[str, Any]:
    requirements = structured_input["task_requirements"]
    execution = structured_input["execution"]
    artifacts = structured_input["artifacts"]
    insights = structured_input["insights"]

    failures = []
    passes = []

    if execution["has_errors"]:
        error_detail = execution["error_messages"][0] if execution["error_messages"] else "Unknown error"
        failures.append(
            {
                "check": "execution_errors",
                "reason": f"Code execution had errors: {error_detail}",
            }
        )

    if not execution["df_info_present"]:
        failures.append(
            {"check": "df_info_missing", "reason": "Required df.info() or df.shape output not found in execution"}
        )
    else:
        passes.append("df_info")

    if requirements["requires_visualization"]:
        if artifacts["count"] == 0:
            failures.append(
                {
                    "check": "visualization_missing",
                    "reason": "Task requires visualization but no artifacts were generated",
                }
            )
        elif not artifacts["has_visualizations"]:
            failures.append(
                {
                    "check": "visualization_type_wrong",
                    "reason": f"Task requires visualization but artifacts are: {artifacts['types']}",
                }
            )
        else:
            passes.append("artifacts")

    if requirements["requires_model"]:
        if not artifacts["has_models"]:
            passes.append("model_training")

    if not insights["has_insights"] and not execution["insights_in_stdout"]:
        failures.append(
            {"check": "insights_missing", "reason": "No insights provided. PROFILING_INSIGHTS block is required."}
        )
    else:
        passes.append("insights")

    stage1_passed = len(failures) == 0

    return {
        "stage1_passed": stage1_passed,
        "failures": failures,
        "passes": passes,
        "feedback": "; ".join([f["reason"] for f in failures]) if failures else "All programmatic checks passed",
    }


def parse_expected_outputs(stdout: str) -> Dict[str, Any]:
    match = re.search(r"EXPECTED_OUTPUTS:\s*(\{[^}]+\})", stdout, re.DOTALL)
    if not match:
        return {"artifacts": [], "insights": [], "df_info": True}

    try:
        raw = match.group(1).replace("'", '"')
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"artifacts": [], "insights": [], "df_info": True}


def validate_contract(
    expected: Dict[str, Any], actual_artifacts: List[str], actual_insights: List[str], df_info_present: bool
) -> Dict[str, Any]:
    mismatches = []

    expected_artifacts = expected.get("artifacts", [])
    if expected_artifacts and len(actual_artifacts) < len(expected_artifacts):
        missing = set(expected_artifacts) - set(actual_artifacts)
        mismatches.append(
            {
                "type": "artifacts",
                "expected": len(expected_artifacts),
                "actual": len(actual_artifacts),
                "missing": list(missing)[:5],
            }
        )

    expected_insights = expected.get("insights", [])
    if expected_insights and len(actual_insights) < len(expected_insights):
        missing = set(expected_insights) - set(actual_insights)
        mismatches.append(
            {
                "type": "insights",
                "expected": len(expected_insights),
                "actual": len(actual_insights),
                "missing": list(missing)[:5],
            }
        )

    if expected.get("df_info", True) and not df_info_present:
        mismatches.append({"type": "df_info", "expected": True, "actual": False, "missing": ["df.info() output"]})

    return {"contract_valid": len(mismatches) == 0, "mismatches": mismatches}


def build_reflection_feedback(contract_result: Dict[str, Any], structured_input: Dict[str, Any]) -> str:
    if contract_result["contract_valid"]:
        return ""

    lines = ["**CONTRACT VIOLATION - REFLECTION REQUIRED:**", ""]
    lines.append("Your declared EXPECTED_OUTPUTS did not match actual results:")

    for mismatch in contract_result["mismatches"]:
        mtype = mismatch["type"]
        expected = mismatch["expected"]
        actual = mismatch["actual"]
        missing = mismatch.get("missing", [])

        lines.append(f"- {mtype.upper()}: Expected {expected}, got {actual}")
        if missing:
            lines.append(f"  Missing: {', '.join(str(m) for m in missing[:3])}")

    lines.append("")
    lines.append("**REFLECT:** What went wrong? Did you:")

    if any(m["type"] == "artifacts" for m in contract_result["mismatches"]):
        lines.append("- Forget to call fig.write_html() or plt.savefig()?")
        lines.append("- Use wrong filename?")

    if any(m["type"] == "insights" for m in contract_result["mismatches"]):
        lines.append("- Forget to print PROFILING_INSIGHTS block?")

    if any(m["type"] == "df_info" for m in contract_result["mismatches"]):
        lines.append("- Forget to run df.info() and df.describe()?")

    lines.append("")
    lines.append("Fix ONLY the missing items. Do NOT regenerate existing artifacts.")

    return "\n".join(lines)
