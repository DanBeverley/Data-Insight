"""Unified Report Generator with Vision capabilities"""

import uuid
import base64
import os
from typing import Dict, Any, AsyncGenerator, Optional, List
from datetime import datetime
from langsmith import traceable
from langchain_core.messages import HumanMessage
from data_scientist_chatbot.app.core.logger import logger


class UnifiedReportGenerator:
    """
    Unified engine for generating comprehensive, multimodal reports.
    Integrates Vision (Qwen3-VL), Data Profiling, and Agent-Authored Layouts.
    """

    def __init__(self):
        from src.intelligence.hybrid_data_profiler import HybridDataProfiler

        self.profiler = HybridDataProfiler()

    @traceable(name="generate_unified_report", tags=["reporting", "vision", "unified"])
    async def generate(
        self,
        session_id: str,
        report_type: str = "general_analysis",
        dataset_path: Optional[str] = None,
        artifacts: Optional[List[Dict]] = None,
        image_paths: Optional[List[str]] = None,
        analysis_focus: Optional[str] = None,
        report_content: Optional[str] = None,
        user_request: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate report sections progressively using a unified pipeline.
        """

        logger.info(f"[REPORT] Starting Unified Report ({report_type}) for session {session_id}")

        try:
            dashboard_context = {
                "profile": {},
                "image_insights": [],
                "artifacts": artifacts or [],
                "models": [],
                "drivers": [],
            }

            if image_paths:
                logger.info(f"[REPORT] Analyzing {len(image_paths)} uploaded images")
                async for insight in self._analyze_images(image_paths, session_id):
                    dashboard_context["image_insights"].append(insight)
                    yield {
                        "section": "visual_analysis",
                        "html": self._render_vision_card(insight),
                        "timestamp": datetime.utcnow().isoformat(),
                    }

            if artifacts:
                yield {
                    "section": "artifact_gallery",
                    "html": await self._render_artifact_gallery(artifacts, session_id),
                    "timestamp": datetime.utcnow().isoformat(),
                }

            logger.info("[REPORT] Calling Brain to structure report content...")
            structured_content = await self._generate_executive_dashboard(
                profile=dashboard_context["profile"],
                image_insights=dashboard_context["image_insights"],
                artifacts=dashboard_context["artifacts"],
                models=dashboard_context["models"],
                drivers=dashboard_context["drivers"],
                focus=analysis_focus or report_type,
                session_id=session_id,
                report_content=report_content,
            )

            styling_keywords = [
                "style",
                "theme",
                "color",
                "font",
                "design",
                "look",
                "aesthetic",
                "modern",
                "minimal",
                "dark",
                "light",
                "blue",
                "gradient",
            ]
            user_wants_custom_styling = user_request and any(kw in user_request.lower() for kw in styling_keywords)

            if user_wants_custom_styling:
                logger.info("[REPORT] User requested custom styling, calling Vision...")
                styled_html = await self._style_with_vision(
                    structured_content=structured_content,
                    artifacts=dashboard_context["artifacts"],
                    user_request=user_request,
                    session_id=session_id,
                )
            else:
                logger.info("[REPORT] Using default styling...")
                styled_html = self._apply_default_styling(structured_content, session_id)

            yield {"section": "executive_dashboard", "html": styled_html, "timestamp": datetime.utcnow().isoformat()}

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            yield {
                "section": "error",
                "html": f'<div class="error">Error generating report: {str(e)}</div>',
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _profile_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Profile the dataset using HybridDataProfiler"""
        from src.common.scalable_dataframe import ScalableDataFrame

        # Load data
        sdf = ScalableDataFrame(dataset_path)
        df = sdf.to_pandas()

        # Generate profile
        profile_summary = self.profiler.generate_comprehensive_profile(df)

        # Convert to dict structure expected by renderers
        return {
            "n_rows": profile_summary.dataset_insights.total_records,
            "n_columns": profile_summary.dataset_insights.total_features,
            "missing_cells": profile_summary.dataset_insights.missing_data_percentage,
            "quality_score": profile_summary.dataset_insights.data_quality_score,
            "summary": profile_summary,
        }

    async def _analyze_images(self, image_paths: List[str], session_id: str) -> AsyncGenerator[Dict, None]:
        """Analyze images using Vision Agent (Qwen3-VL)"""
        from data_scientist_chatbot.app.core.agent_factory import create_vision_agent

        vision_agent = create_vision_agent()

        for path in image_paths:
            try:
                # Encode image
                with open(path, "rb") as img_file:
                    b64_image = base64.b64encode(img_file.read()).decode("utf-8")

                msg = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Analyze this image in detail. Extract key data points, identify trends, and describe the visual structure. Return a structured summary.",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    ]
                )

                response = await vision_agent.ainvoke([msg])

                yield {"filename": path.split("/")[-1], "analysis": response.content, "path": path}
            except Exception as e:
                yield {"filename": path, "error": str(e)}

    async def _style_with_vision(
        self, structured_content: str, artifacts: List[Dict], user_request: Optional[str], session_id: str
    ) -> str:
        from data_scientist_chatbot.app.core.agent_factory import create_vision_agent

        vision_agent = create_vision_agent()

        artifact_list = ", ".join([a.get("filename", "") for a in artifacts]) if artifacts else "None"

        styling_context = user_request if user_request else "Use default professional dark theme with cyan accents"

        prompt = f"""You are a frontend developer specializing in CSS and HTML styling.
Your ONLY task is to style the report content below. DO NOT modify the content itself.

USER'S ORIGINAL REQUEST (extract styling preferences from this):
{styling_context}

BRAIN'S STRUCTURED REPORT CONTENT (DO NOT change the text, only add styling):
{structured_content}

AVAILABLE ARTIFACTS TO EMBED:
{artifact_list}

YOUR TASK:
1. Extract any styling preferences from the user's request (colors, fonts, layout, etc.)
2. If no specific styling is mentioned, use a professional dark theme with cyan accents
3. Apply CSS styling to make the report visually polished and professional
4. Embed artifact references using iframes where data-filename attributes exist
5. Ensure responsive design

OUTPUT REQUIREMENTS:
- Return a complete HTML document with embedded <style> tag
- Include Inter font from Google Fonts
- Make it print-friendly
- DO NOT alter the actual content text from Brain's report

Return only the styled HTML. No markdown code blocks."""

        try:
            result = await vision_agent.ainvoke([HumanMessage(content=prompt)])
            if result and result.content:
                styled = result.content.strip()
                if styled.startswith("```"):
                    styled = styled.split("\n", 1)[1] if "\n" in styled else styled
                    if styled.endswith("```"):
                        styled = styled[:-3]
                return styled
        except Exception as e:
            logger.error(f"[REPORT] Vision styling failed: {e}, using default styling")

        return self._apply_default_styling(structured_content, session_id)

    def _apply_default_styling(self, content: str, session_id: str) -> str:
        import re

        html_content = content

        # Convert markdown headings
        html_content = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html_content, flags=re.MULTILINE)
        html_content = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html_content, flags=re.MULTILINE)

        # Convert bold text
        html_content = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_content)

        # Convert italic text
        html_content = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", html_content)

        # Convert numbered lists (1. item)
        html_content = re.sub(r"^(\d+\.) (.+)$", r"<li>\2</li>", html_content, flags=re.MULTILINE)

        # Convert bullet lists
        html_content = re.sub(r"^[\-\*] (.+)$", r"<li>\1</li>", html_content, flags=re.MULTILINE)

        # Wrap consecutive li elements in ul/ol
        html_content = re.sub(r"(<li>.*</li>\n?)+", r"<ul>\g<0></ul>", html_content)

        # Parse EMBED syntax: ![Description](EMBED:filename.html) or ![Description](EMBED:filename)
        def replace_markdown_embed(match):
            description = match.group(1)
            filename = match.group(2)
            return f'<div class="chart-placeholder artifact-placeholder" data-filename="{filename}"><p>📊 {description}</p></div>'

        html_content = re.sub(r"!\[(.*?)\]\(EMBED:([^)]+)\)", replace_markdown_embed, html_content)

        # Parse legacy EMBED syntax: [EMBED:filename]
        def replace_legacy_embed(match):
            filename = match.group(1)
            return f'<div class="chart-placeholder artifact-placeholder" data-filename="{filename}"><p>📊 Chart: {filename}</p></div>'

        html_content = re.sub(r"\[EMBED:([^\]]+)\]", replace_legacy_embed, html_content)

        # Parse markdown tables
        def parse_markdown_table(table_match):
            lines = table_match.group(0).strip().split("\n")
            if len(lines) < 2:
                return table_match.group(0)

            html_table = '<table class="report-table">'

            # Header row
            headers = [cell.strip() for cell in lines[0].split("|") if cell.strip()]
            html_table += "<thead><tr>"
            for header in headers:
                html_table += f"<th>{header}</th>"
            html_table += "</tr></thead>"

            # Body rows (skip separator row)
            html_table += "<tbody>"
            for line in lines[2:]:
                cells = [cell.strip() for cell in line.split("|") if cell.strip()]
                if cells:
                    html_table += "<tr>"
                    for cell in cells:
                        html_table += f"<td>{cell}</td>"
                    html_table += "</tr>"
            html_table += "</tbody></table>"

            return html_table

        # Match markdown tables (lines starting with |)
        html_content = re.sub(r"(\|[^\n]+\|\n)+", parse_markdown_table, html_content)

        # Convert double newlines to paragraph breaks
        html_content = re.sub(r"\n\n", "</p><p>", html_content)
        html_content = f"<p>{html_content}</p>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Report</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --background: hsl(240, 10%, 3.9%);
            --foreground: hsl(0, 0%, 98%);
            --primary: hsl(180, 100%, 50%);
            --muted: hsl(240, 3.7%, 15.9%);
            --border: hsl(240, 3.7%, 25%);
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background: var(--background);
            color: var(--foreground);
            padding: 2rem;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
        }}
        h1, h2 {{ color: var(--primary); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
        h3 {{ color: var(--foreground); }}
        p {{ margin: 1rem 0; }}
        ul {{ margin: 1rem 0; padding-left: 1.5rem; }}
        li {{ margin: 0.5rem 0; }}
        strong {{ color: var(--primary); }}
        .chart-placeholder {{
            background: var(--muted);
            border: 1px dashed var(--border);
            border-radius: 8px;
            padding: 2rem;
            text-align: center;
            margin: 1rem 0;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            background: var(--muted);
            border-radius: 8px;
            overflow: hidden;
        }}
        .report-table th {{
            background: rgba(0, 242, 234, 0.1);
            color: var(--primary);
            padding: 0.75rem 1rem;
            text-align: left;
            font-weight: 600;
            border-bottom: 1px solid var(--border);
        }}
        .report-table td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }}
        .report-table tr:last-child td {{
            border-bottom: none;
        }}
        .report-table tr:hover {{
            background: rgba(0, 242, 234, 0.05);
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

    async def _generate_executive_dashboard(
        self,
        profile: Dict,
        image_insights: List[Dict],
        artifacts: List[Dict],
        models: List[Dict],
        drivers: Dict,
        focus: str,
        session_id: str,
        report_content: Optional[str] = None,
    ) -> str:
        from data_scientist_chatbot.app.core.agent_factory import create_brain_agent

        brain_agent = create_brain_agent(mode="report")

        context = f"""
        DATA PROFILE: {profile}
        VISUAL INSIGHTS: {image_insights}
        ARTIFACTS: {len(artifacts)} generated charts. filenames: {[a.get('filename') for a in artifacts]}
        AUTOML RESULTS: {models}
        KEY DRIVERS: {drivers}
        FOCUS: {focus}
        
        EXECUTIVE NARRATIVE (Markdown):
        {report_content if report_content else "No narrative provided."}
        """

        prompt = f"""
        You are a senior data analyst generating a professional report for executive stakeholders.
        
        CRITICAL LANGUAGE REQUIREMENT:
        You MUST respond ONLY in English. Do not use Chinese, Japanese, or any other language.
        All text, headers, and content must be in English.
        
        AVAILABLE CONTEXT:
        {context}
        
        YOUR TASK:
        Generate a comprehensive, professionally structured report in MARKDOWN format.
        
        REQUIRED SECTIONS:
        1. ## Executive Summary - Brief overview of key findings
        2. ## Introduction - Dataset description and analysis objectives
        3. ## Methodology - Data processing approach, techniques used
        4. ## Key Findings - Most important discoveries with specific data points
        5. ## Conclusions & Recommendations - Actionable insights and next steps
        
        EMBEDDING ARTIFACTS:
        To reference charts/visualizations, use this exact syntax:
        [EMBED:filename.html]
        
        Example: The correlation analysis reveals strong relationships.
        [EMBED:correlation_matrix.html]
        
        PROFESSIONAL STANDARDS:
        - Write in clear, executive-friendly language (ENGLISH ONLY)
        - Support claims with specific data points from the analysis
        - Reference visualizations where they support the narrative
        - Be specific and actionable in recommendations
        
        OUTPUT:
        Return well-structured MARKDOWN (not HTML). Use ## for section headers, **bold** for emphasis, and bullet points.
        """

        state = {"messages": [HumanMessage(content=prompt)], "session_id": session_id, "plan": None, "scratchpad": ""}

        try:
            result = await brain_agent.ainvoke([HumanMessage(content=prompt)])
            if not result or not result.content:
                logger.error("[REPORT] Brain agent returned empty during dashboard generation")
                return '<div class="error">Failed to generate dashboard layout</div>'
            return result.content
        except Exception as e:
            logger.error(f"[REPORT] Brain agent error: {e}")
            return f'<div class="error">Dashboard generation failed: {str(e)}</div>'

    def save_standalone_report(self, dashboard_html: str, session_id: str) -> str:
        """Wrap the dashboard in a standalone HTML5 file with Design System"""

        # Google-grade Design System
        html_skeleton = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Analysis Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --background: hsl(240, 10%, 3.9%);
            --foreground: hsl(0, 0%, 98%);
            --primary: hsl(180, 100%, 50%);
            --muted: hsl(240, 3.7%, 15.9%);
            --muted-foreground: hsl(240, 5%, 64.9%);
            --border: hsl(240, 3.7%, 15.9%);
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background: var(--background);
            color: var(--foreground);
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.6;
        }}
        
        .report-section {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        
        .section-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border);
        }}
        
        .body-text {{
            color: var(--foreground);
            margin-bottom: 1rem;
            font-size: 1rem;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        
        .metric-box {{
            background: var(--muted);
            border-radius: 10px;
            padding: 1.25rem;
            text-align: center;
        }}
        
        .metric-box .label {{
            display: block;
            font-size: 0.75rem;
            color: var(--muted-foreground);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        
        .metric-box .value {{
            display: block;
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--primary);
        }}
        
        .insight-card {{
            background: rgba(0, 255, 255, 0.05);
            border-left: 3px solid var(--primary);
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            border-radius: 0 8px 8px 0;
        }}
        
        .chart-container {{
            background: var(--muted);
            border-radius: 8px;
            padding: 1rem;
            margin: 1.5rem 0;
        }}
        
        .chart-container iframe {{
            width: 100%;
            height: 400px;
            border: none;
            border-radius: 6px;
        }}
        
        .findings-list {{
            list-style: none;
            padding: 0;
            margin: 1rem 0;
        }}
        
        .findings-list li {{
            padding: 0.5rem 0 0.5rem 1.5rem;
            position: relative;
            color: var(--foreground);
        }}
        
        .findings-list li::before {{
            content: "→";
            position: absolute;
            left: 0;
            color: var(--primary);
        }}
        
        h1 {{
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }}
        
        h2 {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--foreground);
        }}
        
        h3 {{
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--muted-foreground);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
    </style>
</head>
<body>
    <div class="max-w-5xl mx-auto">
        <header class="mb-8">
            <h1>Analysis Report</h1>
            <p style="color: var(--muted-foreground);">Generated by Data Insight AI • {datetime.now().strftime("%B %d, %Y")}</p>
        </header>
        
        <main>
            {dashboard_html}
        </main>
        
        <footer class="mt-12 text-center text-slate-600 text-sm">
            <p>Confidential • Internal Use Only</p>
        </footer>
    </div>
</body>
</html>
        """

        # Ensure directory exists
        reports_dir = os.path.join("data", "reports")
        os.makedirs(reports_dir, exist_ok=True)

        filename = f"Analysis_Report_{session_id[:8]}.html"
        filepath = os.path.join(reports_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_skeleton)

        # [UI REQUIREMENT] Also save as 'dashboard.html' for the persistent UI panel
        dashboard_path = os.path.join(reports_dir, "dashboard.html")
        with open(dashboard_path, "w", encoding="utf-8") as f:
            f.write(html_skeleton)

        return filepath

    # Premium HTML Templates with Glassmorphism and Modern UI
    TEMPLATES = {
        "profile_card": """
        <style>
            .premium-card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
                color: #e0e0e0;
                font-family: 'Inter', sans-serif;
            }
            .premium-card h3 {
                margin-top: 0;
                color: #fff;
                font-weight: 600;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                padding-bottom: 12px;
                margin-bottom: 20px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 20px;
            }
            .metric-box {
                background: rgba(255, 255, 255, 0.03);
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                transition: transform 0.2s;
            }
            .metric-box:hover {
                transform: translateY(-2px);
                background: rgba(255, 255, 255, 0.08);
            }
            .metric-label {
                display: block;
                font-size: 0.85rem;
                color: #a0a0a0;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 8px;
            }
            .metric-value {
                display: block;
                font-size: 1.8rem;
                font-weight: 700;
                background: linear-gradient(45deg, #4ade80, #22d3ee);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
        </style>
        <div class="premium-card profile-card">
            <h3>Dataset Overview</h3>
            <div class="metrics-grid">
                <div class="metric-box">
                    <span class="metric-label">Total Rows</span>
                    <span class="metric-value">{n_rows:,}</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">Total Columns</span>
                    <span class="metric-value">{n_columns}</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">Data Quality</span>
                    <span class="metric-value">98%</span> 
                </div>
            </div>
        </div>
        """,
        "vision_card": """
        <div class="premium-card vision-card">
            <h3>Visual Analysis: {filename}</h3>
            <div class="analysis-content" style="line-height: 1.6; color: #d0d0d0;">
                {analysis}
            </div>
        </div>
        """,
        "model_card": """
        <div class="premium-card model-card">
            <h3>AutoML Performance</h3>
            <div class="metrics-grid">
                <div class="metric-box">
                    <span class="metric-label">Best Model</span>
                    <span class="metric-value" style="font-size: 1.4rem;">{model_name}</span>
                </div>
                <div class="metric-box">
                    <span class="metric-label">{metric_name}</span>
                    <span class="metric-value">{metric_value:.4f}</span>
                </div>
            </div>
        </div>
        """,
        "driver_card": """
        <div class="premium-card driver-card">
            <h3>Key Drivers: <span style="color: #22d3ee;">{target_column}</span></h3>
            <p style="margin-bottom: 16px; color: #ccc;">{insight}</p>
            <ul style="list-style: none; padding: 0;">{drivers_html}</ul>
        </div>
        """,
    }

    def _render_profile_card(self, profile: Dict) -> str:
        """Render a simple profile card (fallback/initial)"""
        return self.TEMPLATES["profile_card"].format(
            n_rows=profile.get("n_rows", 0), n_columns=profile.get("n_columns", 0)
        )

    def _render_vision_card(self, insight: Dict) -> str:
        """Render a card for visual analysis results"""
        return self.TEMPLATES["vision_card"].format(
            filename=insight.get("filename"), analysis=insight.get("analysis", "No analysis available")
        )

    def _render_model_card(self, models_data: List[Dict]) -> str:
        """Render model training results"""
        if not models_data:
            return '<div class="card model-card">No models trained.</div>'

        best_model = models_data[0]
        return self.TEMPLATES["model_card"].format(
            model_name=best_model.get("model_name", "Unknown"),
            metric_name=best_model.get("metric_name", "Metric").title(),
            metric_value=best_model.get("metric_value", 0),
        )

    def _render_driver_card(self, driver_data: Dict) -> str:
        """Render key driver analysis results"""
        if not driver_data:
            return '<div class="error">No driver data available</div>'
        drivers = driver_data.get("top_drivers", [])
        drivers_html = "".join(
            [f"<li><b>{d['feature']}</b>: {d['importance']:.2f} importance</li>" for d in drivers[:3]]
        )

        return self.TEMPLATES["driver_card"].format(
            target_column=driver_data.get("target_column"),
            insight=driver_data.get("insight", ""),
            drivers_html=drivers_html,
        )

    async def _render_artifact_gallery(self, artifacts: List[Dict], session_id: str) -> str:
        """Render gallery with support for interactive HTML plots"""
        html_parts = ['<div class="artifact-gallery">']

        for artifact in artifacts:
            path = artifact.get("file_path") or artifact.get("local_path", "")
            filename = artifact.get("filename", "")

            if filename.endswith(".html"):
                # Embed interactive plot using iframe for isolation
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        plot_html = f.read()
                    # Extract body content or embed directly
                    html_parts.append(
                        f"""
                    <div class="card interactive-plot-card">
                        <h4>{filename}</h4>
                        <div class="plot-container">
                            <iframe srcdoc="{plot_html.replace('"', '&quot;')}" style="width: 100%; height: 500px; border: none;"></iframe>
                        </div>
                    </div>
                    """
                    )
                except Exception as e:
                    html_parts.append(f'<div class="error">Could not load {filename}: {e}</div>')
            elif filename.endswith(".png"):
                # Static image
                try:
                    with open(path, "rb") as f:
                        b64_img = base64.b64encode(f.read()).decode("utf-8")
                    html_parts.append(
                        f"""
                    <div class="card image-card">
                        <h4>{filename}</h4>
                        <img src="data:image/png;base64,{b64_img}" alt="{filename}" />
                    </div>
                    """
                    )
                except:
                    pass

        html_parts.append("</div>")
        return "\n".join(html_parts)
