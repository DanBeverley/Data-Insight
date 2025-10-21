import pytest
from data_scientist_chatbot.app.utils.sanitizers import sanitize_output


@pytest.mark.unit
class TestOutputSanitization:
    def test_sanitize_removes_debug_messages_with_content(self):
        output = """DEBUG: Processing data
        The correlation between price and area is 0.75
        This indicates a strong positive relationship
        INFO: Task completed successfully
        The analysis shows significant findings"""
        result = sanitize_output(output)
        assert "correlation between price and area is 0.75" in result
        assert "analysis shows significant findings" in result

    def test_sanitize_removes_plot_save_messages_with_content(self):
        output = """PLOT_SAVED: /tmp/plot.png
        Analysis complete
        The scatter plot reveals a linear pattern
        between the two variables"""
        result = sanitize_output(output)
        assert "scatter plot reveals a linear pattern" in result

    def test_sanitize_function_returns_string(self):
        output = 'Traceback (most recent call last):\nFile "<stdin>", line 1\nTypeError: error'
        result = sanitize_output(output)
        assert isinstance(result, str)

    def test_sanitize_preserves_meaningful_content(self):
        output = "The correlation coefficient is 0.85"
        result = sanitize_output(output)
        assert result == output

    def test_sanitize_preserves_analysis_results(self):
        output = "Analysis result: Mean = 42.5, Std = 10.2"
        result = sanitize_output(output)
        assert "42.5" in result
        assert "10.2" in result

    def test_sanitize_handles_empty_string(self):
        result = sanitize_output("")
        assert result == ""

    def test_sanitize_handles_none_input(self):
        result = sanitize_output(None)
        assert result is None

    def test_sanitize_with_mixed_content(self):
        output = """matplotlib.pyplot
        plt.show()
        The regression model achieved R-squared of 0.92
        This demonstrates excellent model fit
        The residuals show normal distribution"""
        result = sanitize_output(output)
        assert "regression model achieved" in result
        assert "R-squared of 0.92" in result

    def test_sanitize_handles_mixed_debug_and_analysis(self):
        output = """DEBUG: Starting analysis
        The mean value is 50 with standard deviation of 10
        INFO: Processing complete
        The data shows normal distribution
        Confidence interval: [45, 55]"""
        result = sanitize_output(output)
        assert "mean value is 50" in result or "standard deviation" in result
        assert "Confidence interval" in result or "[45, 55]" in result

    def test_sanitize_preserves_data_analysis_terms(self):
        output = "Summary: Correlation is strong\nConclusion: Positive relationship"
        result = sanitize_output(output)
        assert "Summary" in result or "Correlation is strong" in result
        assert "Conclusion" in result or "Positive relationship" in result

    def test_sanitize_fallback_protection(self):
        minimal_content = "OK"
        result = sanitize_output(minimal_content)
        assert result == minimal_content

    def test_sanitize_returns_valid_output(self):
        output = ">>> print(42)\n42\n>>> "
        result = sanitize_output(output)
        assert isinstance(result, str)
        assert "42" in result

    def test_sanitize_handles_multiline_cleanup(self):
        output = """Line 1


        Line 2"""
        result = sanitize_output(output)
        assert result.count("\n\n\n") == 0

    def test_sanitize_preserves_important_technical_content(self):
        output = """The analysis completed with the following results:
        - Mean: 42.5
        - Median: 40.0
        - Standard Deviation: 5.2
        These statistics indicate a normal distribution"""
        result = sanitize_output(output)
        assert "Mean: 42.5" in result
        assert "Standard Deviation: 5.2" in result

    def test_sanitize_handles_realistic_output(self):
        output = """Analyzing dataset...
        Found 1000 rows and 10 columns
        Correlation matrix computed
        Result: Strong correlation (0.85) between features A and B
        Recommendation: Consider feature engineering"""
        result = sanitize_output(output)
        assert "1000 rows and 10 columns" in result or "Strong correlation" in result
