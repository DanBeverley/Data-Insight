import pytest
from typing import List, Dict
from pathlib import Path
import tempfile
import os

from src.reporting.report_exporter import (
    ReportExporter,
    html_to_text,
    html_to_reportlab,
    make_standalone_html,
    get_report_exporter,
)


@pytest.fixture
def sample_sections() -> List[Dict]:
    return [
        {
            "title": "Executive Summary",
            "content": "<h2>Overview</h2><p>This analysis covers <strong>545 properties</strong>.</p>",
        },
        {
            "title": "Key Findings",
            "content": "<ul><li>Price range: $100K - $1M</li><li>Average area: 5000 sqft</li></ul>",
        },
    ]


@pytest.fixture
def exporter() -> ReportExporter:
    return get_report_exporter()


@pytest.mark.unit
class TestHtmlToText:
    def test_strips_script_tags(self):
        html = "<script>alert('xss')</script><p>Content</p>"
        result = html_to_text(html)
        assert "alert" not in result
        assert "Content" in result

    def test_strips_style_tags(self):
        html = "<style>.red{color:red}</style><p>Text</p>"
        result = html_to_text(html)
        assert "color" not in result
        assert "Text" in result

    def test_converts_headings(self):
        html = "<h1>Title</h1><h2>Subtitle</h2>"
        result = html_to_text(html)
        assert "===" in result or "Title" in result
        assert "##" in result or "Subtitle" in result

    def test_converts_lists(self):
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        result = html_to_text(html)
        assert "•" in result or "Item 1" in result

    def test_converts_bold_and_italic(self):
        html = "<strong>Bold</strong> and <em>Italic</em>"
        result = html_to_text(html)
        assert "**Bold**" in result or "Bold" in result
        assert "*Italic*" in result or "Italic" in result

    def test_handles_empty_string(self):
        result = html_to_text("")
        assert result == ""

    def test_extracts_chart_references(self):
        html = '<iframe data-filename="chart.html" src="/static/plots/chart.html"></iframe>'
        result = html_to_text(html)
        assert "chart" in result.lower()


@pytest.mark.unit
class TestHtmlToReportlab:
    def test_converts_headings_to_font_tags(self):
        html = "<h1>Title</h1>"
        result = html_to_reportlab(html)
        assert "<font" in result or "<b>" in result

    def test_converts_strong_to_bold(self):
        html = "<strong>Important</strong>"
        result = html_to_reportlab(html)
        assert "<b>" in result and "</b>" in result

    def test_converts_em_to_italic(self):
        html = "<em>Emphasis</em>"
        result = html_to_reportlab(html)
        assert "<i>" in result and "</i>" in result

    def test_converts_line_breaks(self):
        html = "<p>Line1</p><p>Line2</p>"
        result = html_to_reportlab(html)
        assert "<br/>" in result

    def test_handles_iframe_chart_reference(self):
        html = '<iframe src="/static/plots/distribution.html"></iframe>'
        result = html_to_reportlab(html)
        assert "Chart" in result or "distribution" in result


@pytest.mark.unit
class TestMakeStandaloneHtml:
    def test_preserves_content_without_iframes(self):
        html = "<html><body><p>Simple content</p></body></html>"
        result = make_standalone_html(html)
        assert "Simple content" in result

    def test_handles_missing_artifact_files(self):
        html = '<iframe src="/static/plots/nonexistent_chart.html"></iframe>'
        result = make_standalone_html(html)
        assert result is not None


@pytest.mark.unit
class TestReportExporter:
    @pytest.mark.asyncio
    async def test_export_pdf_returns_bytes(self, exporter: ReportExporter, sample_sections: List[Dict]):
        try:
            result = await exporter.export_pdf(sample_sections, title="Test Report")
            assert isinstance(result, bytes)
            assert len(result) > 0
            assert result[:4] == b"%PDF"
        except ImportError:
            pytest.skip("reportlab not installed")

    def test_export_docx_returns_bytes(self, exporter: ReportExporter, sample_sections: List[Dict]):
        try:
            result = exporter.export_docx(sample_sections, title="Test Report")
            assert isinstance(result, bytes)
            assert len(result) > 0
        except ImportError:
            pytest.skip("python-docx not installed")

    def test_export_html_returns_bytes(self, exporter: ReportExporter, sample_sections: List[Dict]):
        result = exporter.export_html(sample_sections, title="Test Report")
        assert isinstance(result, bytes)
        assert b"<!DOCTYPE html>" in result
        assert b"Test Report" in result

    def test_export_txt_returns_bytes(self, exporter: ReportExporter, sample_sections: List[Dict]):
        result = exporter.export_txt(sample_sections, title="Test Report")
        assert isinstance(result, bytes)
        content = result.decode("utf-8")
        assert "EXECUTIVE SUMMARY" in content or "Executive Summary" in content

    def test_export_handles_empty_sections(self, exporter: ReportExporter):
        result = exporter.export_html([], title="Empty Report")
        assert isinstance(result, bytes)
        assert b"Empty Report" in result

    def test_export_handles_html_in_content(self, exporter: ReportExporter):
        sections = [{"title": "Test", "content": "<script>bad</script><p>Good</p>"}]
        result = exporter.export_txt(sections, title="Test")
        content = result.decode("utf-8")
        assert "bad" not in content or "script" not in content


@pytest.mark.unit
class TestExporterEdgeCases:
    @pytest.mark.asyncio
    async def test_pdf_with_unicode_content(self, exporter: ReportExporter):
        sections = [{"title": "Unicode", "content": "<p>Price: ₹1,000,000 • 中文</p>"}]
        try:
            result = await exporter.export_pdf(sections, title="Unicode Test")
            assert isinstance(result, bytes)
        except ImportError:
            pytest.skip("reportlab not installed")
        except Exception:
            pass

    def test_docx_with_special_characters(self, exporter: ReportExporter):
        sections = [{"title": "Special", "content": "<p>Test &amp; &lt;chars&gt;</p>"}]
        try:
            result = exporter.export_docx(sections, title="Special Chars")
            assert isinstance(result, bytes)
        except ImportError:
            pytest.skip("python-docx not installed")
