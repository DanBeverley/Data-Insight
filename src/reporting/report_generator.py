"""Report generation orchestrator"""

import uuid
from typing import Dict, Any, AsyncGenerator, Optional
from datetime import datetime
from langsmith import traceable


class ReportGenerator:
    """Orchestrates progressive report generation for dataset analysis"""

    def __init__(self):
        from src.intelligence.hybrid_data_profiler import HybridDataProfiler

        self.profiler = HybridDataProfiler()

    @traceable(name="generate_report", tags=["reporting"])
    async def generate(
        self, session_id: str, dataset_path: str, dataset_name: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate report sections progressively.

        Args:
            session_id: User session identifier
            dataset_path: Path to dataset file
            dataset_name: Display name of dataset

        Yields:
            Report sections as they complete
        """
        from data_scientist_chatbot.app.agent import run_brain_agent
        from data_scientist_chatbot.app.core.logger import logger

        logger.info(f"[REPORT] Starting generation for {dataset_name}")

        try:
            profile = await self._profile_dataset(dataset_path)
            yield {
                "section": "data_profile",
                "html": self._render_profile(profile),
                "timestamp": datetime.utcnow().isoformat(),
            }
            logger.info(f"[REPORT] Data profile section completed")

            summary = await self._generate_summary(profile, session_id)
            yield {
                "section": "executive_summary",
                "html": self._render_summary(summary),
                "timestamp": datetime.utcnow().isoformat(),
            }
            logger.info(f"[REPORT] Executive summary section completed")

            recommendations = await self._generate_recommendations(profile, session_id)
            yield {
                "section": "recommendations",
                "html": self._render_recommendations(recommendations),
                "timestamp": datetime.utcnow().isoformat(),
            }
            logger.info(f"[REPORT] Recommendations section completed")

            yield {
                "section": "model_arena",
                "html": self._render_model_arena_placeholder(),
                "timestamp": datetime.utcnow().isoformat(),
            }
            logger.info(f"[REPORT] Model arena placeholder created")

        except Exception as e:
            logger.error(f"[REPORT] Generation failed: {e}")
            yield {
                "section": "error",
                "html": f'<div class="error">Report generation failed: {str(e)}</div>',
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _profile_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Run hybrid data profiler"""
        import pandas as pd
        import asyncio
        from functools import partial

        try:
            df = pd.read_csv(dataset_path)

            loop = asyncio.get_event_loop()
            profile_summary = await loop.run_in_executor(
                None, partial(self.profiler.generate_comprehensive_profile, df)
            )

            return {
                "n_rows": profile_summary.basic_info.get("n_rows", len(df)),
                "n_columns": profile_summary.basic_info.get("n_columns", len(df.columns)),
                "missing_pct": profile_summary.data_quality.get("overall_completeness", 100.0),
                "profile_summary": profile_summary,
            }
        except Exception as e:
            return {"error": str(e), "n_rows": 0, "n_columns": 0, "missing_pct": 0.0}

    async def _generate_summary(self, profile: Dict[str, Any], session_id: str) -> str:
        """Use brain agent to generate executive summary"""
        from data_scientist_chatbot.app.agent import run_brain_agent
        from langchain_core.messages import HumanMessage

        prompt = f"""
Analyze this dataset profile and generate a concise executive summary (3-5 bullet points):

Dataset Statistics:
- Rows: {profile.get('n_rows', 'Unknown')}
- Columns: {profile.get('n_columns', 'Unknown')}
- Missing Data: {profile.get('missing_pct', 0)}%

Generate:
1. Key findings about data quality
2. Notable patterns or characteristics
3. Immediate insights for decision making

Format as HTML bullet list.
"""

        state = {"messages": [HumanMessage(content=prompt)], "session_id": session_id, "plan": None, "scratchpad": ""}

        result = run_brain_agent(state)
        messages = result.get("messages", [])

        if messages and hasattr(messages[-1], "content"):
            return messages[-1].content
        return "Unable to generate summary at this time."

    async def _generate_recommendations(self, profile: Dict[str, Any], session_id: str) -> str:
        """Use brain agent to generate iterative recommendations"""
        from data_scientist_chatbot.app.agent import run_brain_agent
        from langchain_core.messages import HumanMessage

        prompt = f"""
Based on this dataset profile, suggest 3-5 iterative improvements:

Dataset Characteristics:
- Size: {profile.get('n_rows', 0)} rows × {profile.get('n_columns', 0)} columns
- Data Quality: {100 - profile.get('missing_pct', 0):.1f}%

Provide actionable recommendations in these categories:
1. Data Quality Improvements
2. Feature Engineering Opportunities
3. Model Selection Suggestions

Format as HTML ordered list with clear action items.
"""

        state = {"messages": [HumanMessage(content=prompt)], "session_id": session_id, "plan": None, "scratchpad": ""}

        result = run_brain_agent(state)
        messages = result.get("messages", [])

        if messages and hasattr(messages[-1], "content"):
            return messages[-1].content
        return "Unable to generate recommendations."

    def _render_profile(self, profile: Dict[str, Any]) -> str:
        """Convert profile dict to HTML"""
        n_rows = profile.get("n_rows", 0)
        n_cols = profile.get("n_columns", 0)
        missing = profile.get("missing_pct", 0.0)
        completeness = 100 - missing

        quality_icon = "✓" if completeness >= 95 else "⚠" if completeness >= 80 else "✗"

        return f"""
<div class="report-section data-profile">
    <h3>Data Profile</h3>
    <div class="stats-grid">
        <div class="stat-card">
            <span class="stat-label">Rows</span>
            <span class="stat-value">{n_rows:,}</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Columns</span>
            <span class="stat-value">{n_cols}</span>
        </div>
        <div class="stat-card">
            <span class="stat-label">Completeness</span>
            <span class="stat-value">{completeness:.1f}% {quality_icon}</span>
        </div>
    </div>
</div>
"""

    def _render_summary(self, summary: str) -> str:
        """Wrap summary in HTML"""
        return f"""
<div class="report-section executive-summary">
    <h3>Executive Summary</h3>
    <div class="summary-content">
        {summary}
    </div>
</div>
"""

    def _render_recommendations(self, recommendations: str) -> str:
        """Wrap recommendations in HTML"""
        return f"""
<div class="report-section recommendations">
    <h3>Recommended Next Steps</h3>
    <div class="recommendations-content">
        {recommendations}
    </div>
</div>
"""

    def _render_model_arena_placeholder(self) -> str:
        """Render model arena with training button"""
        return """
<div class="report-section model-arena">
    <h3>Model Arena</h3>
    <div class="arena-placeholder">
        <p>Train multiple models to compare performance on this dataset.</p>
        <button id="run-models-btn" class="primary-btn" onclick="window.startModelTraining()">
            ▶ Run Model Comparison
        </button>
        <p class="help-text">Trains XGBoost, RandomForest, and LogisticRegression (~3-5 minutes)</p>
    </div>
</div>
"""
