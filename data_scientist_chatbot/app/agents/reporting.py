"""Reporting nodes - Analyst, Architect, and Presenter."""

import os
import re
import base64
from typing import Dict, Any, List

from langchain_core.messages import AIMessage
from langsmith import traceable

from data_scientist_chatbot.app.core.state import GlobalState
from data_scientist_chatbot.app.core.logger import logger
from src.api_utils.artifact_tracker import get_artifact_tracker
from src.api_utils.session_management import session_data_manager
from src.reporting.unified_report_generator import UnifiedReportGenerator


@traceable(name="analyst_node", tags=["reporting", "analyst"])
def run_analyst_node(state: GlobalState) -> Dict[str, Any]:
    """Pass artifacts to architect for dashboard generation."""
    logger.info("[ANALYST] Passing to Architect...")
    artifacts = state.get("artifacts") or []
    logger.info(f"[ANALYST] {len(artifacts)} artifacts available for dashboard.")
    return {
        "messages": [
            AIMessage(
                content=f"Analyst: {len(artifacts)} artifacts ready for dashboard.",
                additional_kwargs={"internal": True},
            )
        ],
        "current_agent": "analyst",
        "artifacts": artifacts,
        "agent_insights": state.get("agent_insights") or [],
        "retry_count": state.get("retry_count", 0),
        "workflow_stage": "reporting",
    }


def _embed_artifact(filename: str, artifacts: List[Dict], static_plots_dir: str) -> str:
    """Embed a single artifact as HTML."""
    artifact = next(
        (
            a
            for a in artifacts
            if a.get("filename") == filename
            or a.get("filename", "").endswith(filename)
            or filename.endswith(a.get("filename", ""))
        ),
        None,
    )

    if not artifact:
        static_path = os.path.join(static_plots_dir, filename)
        if os.path.exists(static_path):
            artifact = {"filename": filename, "local_path": static_path}

    if artifact:
        path = artifact.get("file_path") or artifact.get("local_path")
        if not path:
            logger.warning(f"[ARCHITECT] Artifact {filename} has no path")
            return f'<div class="chart-error">Artifact {filename} has no path</div>'
        if not os.path.exists(path):
            logger.warning(f"[ARCHITECT] Artifact file not found: {path}")
            return f'<div class="chart-error">Chart file not found: {filename}</div>'

        try:
            if filename.endswith(".html"):
                chart_url = f"/static/plots/{filename}"
                return f'<div class="chart-container"><iframe src="{chart_url}" style="width: 100%; height: 500px; border: none; border-radius: 12px; background: transparent;"></iframe></div>'
            elif filename.endswith(".png"):
                with open(path, "rb") as f:
                    b64_img = base64.b64encode(f.read()).decode("utf-8")
                return f'<div class="chart-container"><img src="data:image/png;base64,{b64_img}" alt="{filename}" style="max-width: 100%; border-radius: 12px;"/></div>'
            else:
                return f'<div class="chart-error">Unsupported format: {filename}</div>'
        except Exception as e:
            logger.error(f"[ARCHITECT] Failed to embed {filename}: {e}")
            return f'<div class="chart-error">Could not load {filename}</div>'

    logger.warning(f"[ARCHITECT] Artifact not found: {filename}")
    return f'<div class="chart-error">Artifact {filename} not found</div>'


@traceable(name="architect_node", tags=["reporting", "architect"])
async def run_architect_node(state: GlobalState) -> Dict[str, Any]:
    """Generate UI Dashboard using UnifiedReportGenerator."""
    logger.info(f"[ARCHITECT] Starting Architect Node for session {state.get('session_id')}...")
    session_id = state.get("session_id")

    tracker = get_artifact_tracker()
    artifact_data = tracker.get_session_artifacts(session_id)
    artifacts = artifact_data.get("artifacts", [])

    dataset_artifacts = [a for a in artifacts if a.get("category") == "dataset"]
    dataset_path = None
    session_data = None

    if dataset_artifacts:
        dataset_path = dataset_artifacts[0].get("local_path")

    if not dataset_path:
        session_data = session_data_manager.get_session(session_id)
        dataset_path = session_data.get("dataset_path") if session_data else None

    logger.info(f"[ARCHITECT] Found {len(artifacts)} artifacts, dataset_path: {dataset_path}")

    if not dataset_path:
        if session_data is None:
            session_data = session_data_manager.get_session(session_id)
        logger.error(f"[ARCHITECT] DEBUG: Session data keys: {list(session_data.keys()) if session_data else 'None'}")

    report_content = None
    import builtins

    if hasattr(builtins, "_session_store") and session_id in builtins._session_store:
        report_content = builtins._session_store[session_id].get("report_summary")
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and "Dataset Overview" in str(msg.content):
                report_content = msg.content
                break

    generator = UnifiedReportGenerator()
    static_plots_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "static", "plots")

    final_html = ""
    try:
        async for chunk in generator.generate(
            session_id=session_id,
            dataset_path=dataset_path,
            artifacts=artifacts,
            report_type="general_analysis",
            report_content=report_content,
        ):
            if chunk["section"] == "executive_dashboard":
                final_html = chunk["html"]

        def replace_placeholder(match):
            filename = match.group(1)
            return _embed_artifact(filename, artifacts, static_plots_dir)

        final_html = re.sub(
            r'<div[^>]*data-filename=["\']([^"\']+)["\'][^>]*>.*?</div>',
            replace_placeholder,
            final_html,
            flags=re.DOTALL,
        )

        logger.info(f"[ARCHITECT] Dashboard generated ({len(final_html)} chars).")

        report_path = generator.save_standalone_report(final_html, session_id)
        filename = os.path.basename(report_path)
        report_url = f"/reports/{filename}"

        logger.info(f"[ARCHITECT] Dashboard generated and saved to {report_path} (URL: {report_url})")

        return {
            "messages": [
                AIMessage(
                    content=f"**Report Generated Successfully**\n\nYour comprehensive data analysis report is ready. It includes:\n- Executive summary with key findings\n- Interactive visualizations\n- Statistical insights and patterns\n- Actionable recommendations\n\n📊 [**View Interactive Dashboard**](report:{report_url})",
                    additional_kwargs={"report_url": report_url, "report_path": report_path},
                )
            ],
            "current_agent": "architect",
            "artifacts": artifacts,
            "agent_insights": state.get("agent_insights") or [],
            "retry_count": state.get("retry_count", 0),
            "report_url": report_url,
            "report_path": report_path,
            "workflow_stage": "report_generated",
        }
    except Exception as e:
        logger.error(f"[ARCHITECT] Failed to generate dashboard: {e}")
        return {
            "messages": [AIMessage(content=f"Failed to generate dashboard: {e}")],
            "current_agent": "architect",
            "artifacts": artifacts,
            "agent_insights": state.get("agent_insights") or [],
            "retry_count": state.get("retry_count", 0),
            "workflow_stage": "report_failed",
        }


@traceable(name="presenter_node", tags=["reporting", "presenter"])
async def run_presenter_node(state: GlobalState) -> Dict[str, Any]:
    """Deliver final response to user."""
    return {
        "messages": [AIMessage(content="Analysis completed, but report generation failed.")],
        "current_agent": "presenter",
        "artifacts": state.get("artifacts") or [],
        "agent_insights": state.get("agent_insights") or [],
        "retry_count": state.get("retry_count", 0),
        "workflow_stage": "completed",
    }
