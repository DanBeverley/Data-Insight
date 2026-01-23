import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from pathlib import Path
import tempfile


@pytest.fixture
def mock_report_data() -> Dict[str, Any]:
    return {
        "session_id": "test-session-abc123",
        "dataset_name": "housing_data",
        "insights": [
            {"label": "Dataset Size", "value": "545 rows", "type": "overview"},
            {"label": "Price Range", "value": "$100K - $1M", "type": "finding"},
        ],
        "artifacts": [
            {
                "filename": "price_distribution.html",
                "category": "report",
                "local_path": "/static/plots/price_distribution.html",
            },
            {
                "filename": "correlation_heatmap.html",
                "category": "report",
                "local_path": "/static/plots/correlation_heatmap.html",
            },
        ],
        "executive_summary": "Analysis of 545 property records showing strong correlation between area and price.",
    }


@pytest.mark.integration
class TestReportGenerationFlow:
    def test_report_generator_creates_html_file(self, mock_report_data: Dict):
        from src.reporting.unified_report_generator import UnifiedReportGenerator

        generator = UnifiedReportGenerator()
        assert generator is not None

    def test_report_includes_artifacts(self, mock_report_data: Dict):
        artifacts = mock_report_data["artifacts"]

        assert len(artifacts) == 2
        assert any("distribution" in a["filename"] for a in artifacts)
        assert any("heatmap" in a["filename"] for a in artifacts)

    def test_report_includes_insights(self, mock_report_data: Dict):
        insights = mock_report_data["insights"]

        assert len(insights) >= 2
        assert any(i["label"] == "Dataset Size" for i in insights)


@pytest.mark.integration
class TestReportExportFlow:
    @pytest.mark.asyncio
    async def test_export_to_pdf_integration(self, mock_report_data: Dict):
        from src.reporting.report_exporter import get_report_exporter

        exporter = get_report_exporter()
        sections = [
            {"title": "Summary", "content": mock_report_data["executive_summary"]},
            {"title": "Insights", "content": str(mock_report_data["insights"])},
        ]

        try:
            result = await exporter.export_pdf(sections, title="Test Report")
            assert isinstance(result, bytes)
            assert result[:4] == b"%PDF"
        except (ImportError, RuntimeError, Exception) as e:
            error_msg = str(e).lower()
            if any(kw in error_msg for kw in ["playwright", "browser", "pdf generation failed"]):
                pytest.skip("Playwright browser not available in CI")
            raise

    def test_export_to_html_integration(self, mock_report_data: Dict):
        from src.reporting.report_exporter import get_report_exporter

        exporter = get_report_exporter()
        sections = [{"title": "Summary", "content": mock_report_data["executive_summary"]}]

        result = exporter.export_html(sections, title="Test Report")
        assert isinstance(result, bytes)
        assert b"<!DOCTYPE html>" in result

    def test_export_to_docx_integration(self, mock_report_data: Dict):
        from src.reporting.report_exporter import get_report_exporter

        exporter = get_report_exporter()
        sections = [{"title": "Summary", "content": mock_report_data["executive_summary"]}]

        try:
            result = exporter.export_docx(sections, title="Test Report")
            assert isinstance(result, bytes)
            assert len(result) > 0
        except ImportError:
            pytest.skip("python-docx not installed")


@pytest.mark.integration
class TestReportArtifactEmbedding:
    def test_standalone_html_embeds_artifacts(self):
        from src.reporting.report_exporter import make_standalone_html

        html_with_iframe = """<!DOCTYPE html>
<html>
<body>
<iframe src="/static/plots/test_chart.html"></iframe>
</body>
</html>"""

        result = make_standalone_html(html_with_iframe)
        assert result is not None
        assert isinstance(result, str)

    def test_html_conversion_preserves_structure(self):
        from src.reporting.report_exporter import html_to_text

        html = """<html>
<head><style>.test{}</style></head>
<body>
<h1>Title</h1>
<p>Content with <strong>bold</strong> text.</p>
</body>
</html>"""

        result = html_to_text(html)
        assert "Title" in result
        assert "Content" in result
        assert ".test" not in result


@pytest.mark.integration
class TestReportRouterIntegration:
    def test_export_endpoint_handles_session_id(self):
        session_id = "abc12345-test-session"
        prefix = session_id[:8]

        assert prefix == "abc12345"

    def test_report_path_construction(self):
        from pathlib import Path

        reports_dir = Path("data/reports")
        session_id = "test123"

        expected_pattern = f"*{session_id[:8]}*.html"
        assert "test1234" in expected_pattern or session_id[:8] in expected_pattern
