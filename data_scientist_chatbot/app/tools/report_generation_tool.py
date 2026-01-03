"""Tool for generating comprehensive reports dynamically"""

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import List, Optional


class ReportGenerationInput(BaseModel):
    report_type: str = Field(description="Type of report: 'eda', 'model_results', or 'general_analysis'")
    analysis_focus: Optional[str] = Field(
        default=None,
        description="Specific aspects to emphasize (e.g., 'missing values', 'correlations', 'feature importance')",
    )
    artifact_ids: Optional[List[str]] = Field(
        default=None, description="List of artifact IDs to reference and explain in the report"
    )
    image_paths: Optional[List[str]] = Field(
        default=None, description="List of paths to uploaded images to analyze and include in the report"
    )


@tool
def generate_comprehensive_report(
    report_type: str,
    analysis_focus: str = "",
    artifact_ids: str = "",
    image_paths: str = "",
) -> str:
    """
    Generate a comprehensive, interactive HTML dashboard and report.

    CRITICAL: Call this tool ONLY when user EXPLICITLY asks for "report", "dashboard", or "full UI view".

    Args:
        report_type: Type of report - 'eda' (exploratory), 'model_results' (ML), or 'general_analysis'
        analysis_focus: Specific focus (e.g., 'sales trends', 'missing data')
        artifact_ids: Comma-separated artifact IDs to include
        image_paths: Comma-separated image paths to analyze

    Returns:
        Confirmation string. The actual report is streamed to the frontend UI.
    """
    try:
        # Import here to avoid circular imports
        import sys
        import os
        from pathlib import Path

        # Ensure src is in path
        src_path = str(Path(__file__).parent.parent.parent.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from src.reporting.unified_report_generator import UnifiedReportGenerator
        from src.api_utils.session_management import session_data_manager

        # Get session ID from context (this assumes the tool is running in a context where session_id is available)
        # In the current architecture, we might need to pass session_id explicitly or retrieve it from a global context
        # For now, we'll try to get it from the active session if possible, or default to a placeholder
        # NOTE: The agent usually doesn't pass session_id to tools directly.
        # We might need to rely on the context manager or pass it as an argument if the schema allows.
        # However, looking at the tool definition, it doesn't take session_id.
        # We will assume the session_id is injected or available via context.

        # For this refactor, we'll instantiate the generator.
        # The actual execution might need the session_id which is tricky here without changing the signature.
        # But the tool definition in tools.py usually handles the session_id injection.

        # Let's return a message that triggers the actual generation in the agent loop
        # or if we can, run it here.

        # Actually, the agent.py calls this tool.
        # Let's look at how other tools get the session_id.
        # They usually get it from the `execute_python_in_sandbox` which has it.

        # Since we can't easily get session_id here without changing the signature,
        # and the user wants to consolidate logic,
        # we will return a structured signal that the Agent can interpret to run the generator,
        # OR we just return the text as before but ensure the Agent *actually* calls the generator code.

        # Wait, the previous implementation just returned a string:
        # return "Report generation initiated. The Frontend Architect is designing the dashboard now..."

        # The actual work was done in `run_architect_node` in `agent.py`.
        # So this tool is just a "trigger" for the architect node.

        # So, strictly speaking, this tool file doesn't need to *run* the code,
        # it just needs to exist so the LLM can call it.
        # BUT, the `run_architect_node` in `agent.py` MUST use `UnifiedReportGenerator`.

        # Let's verify `agent.py` uses `UnifiedReportGenerator`.
        return "Report generation initiated. The Frontend Architect is designing the dashboard now..."

    except Exception as e:
        return f"Error initiating report generation: {str(e)}"
