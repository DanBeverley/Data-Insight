import pytest
import pandas as pd
import numpy as np

from src.data_quality.validator import (
    DataQualityValidator,
    ValidationReport,
    ValidationCheck
)


@pytest.mark.unit
class TestDataQualityValidator:

    @pytest.fixture
    def clean_df(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'value': [10.5, 20.3, 30.1, 40.9, 50.2],
            'category': ['A', 'B', 'C', 'D', 'E']
        })

    @pytest.fixture
    def df_with_duplicates(self):
        return pd.DataFrame({
            'id': [1, 2, 2, 3],
            'value': [10, 20, 20, 30],
            'category': ['A', 'B', 'B', 'C']
        })

    @pytest.fixture
    def df_with_high_missing(self):
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'mostly_null': [np.nan, np.nan, np.nan, np.nan, 1.0],
            'good_col': [1, 2, 3, 4, 5]
        })

    @pytest.fixture
    def df_with_mixed_types(self):
        return pd.DataFrame({
            'id': [1, 2, 3],
            'mixed': [1, 'two', 3.0]
        })

    def test_validator_requires_non_empty_dataframe(self):
        with pytest.raises(ValueError):
            DataQualityValidator(pd.DataFrame())

    def test_validator_initialization(self, clean_df):
        validator = DataQualityValidator(clean_df)
        assert validator.df is not None
        assert isinstance(validator.expected_schema, dict)

    def test_validate_returns_report(self, clean_df):
        validator = DataQualityValidator(clean_df)
        report = validator.validate()
        assert isinstance(report, ValidationReport)

    def test_validate_clean_data_passes(self, clean_df):
        validator = DataQualityValidator(clean_df)
        report = validator.validate()
        assert report.is_valid is True

    def test_check_duplicates_detects_duplicates(self, df_with_duplicates):
        validator = DataQualityValidator(df_with_duplicates)
        report = validator.validate()

        duplicate_checks = [c for c in report.checks if 'Duplicate' in c.name]
        assert len(duplicate_checks) > 0
        assert duplicate_checks[0].passed is False

    def test_check_duplicates_passes_on_clean_data(self, clean_df):
        validator = DataQualityValidator(clean_df)
        report = validator.validate()

        duplicate_checks = [c for c in report.checks if 'Duplicate' in c.name]
        assert len(duplicate_checks) > 0
        assert duplicate_checks[0].passed is True

    def test_check_high_missing_values_detects(self, df_with_high_missing):
        validator = DataQualityValidator(df_with_high_missing)
        report = validator.validate()

        missing_checks = [c for c in report.checks if 'Missing' in c.name]
        assert len(missing_checks) > 0

    def test_check_mixed_types_detects(self, df_with_mixed_types):
        validator = DataQualityValidator(df_with_mixed_types)
        report = validator.validate()

        mixed_checks = [c for c in report.checks if 'Mixed' in c.name]
        assert len(mixed_checks) > 0
        assert mixed_checks[0].passed is False

    def test_schema_compliance_with_matching_schema(self, clean_df):
        expected_schema = {'id': 'int', 'value': 'float', 'category': 'str'}
        validator = DataQualityValidator(clean_df, expected_schema=expected_schema)
        report = validator.validate()

        schema_checks = [c for c in report.checks if 'Schema' in c.name]
        assert len(schema_checks) > 0
        assert schema_checks[0].passed is True

    def test_schema_compliance_with_missing_columns(self, clean_df):
        expected_schema = {'id': 'int', 'value': 'float', 'missing_col': 'str'}
        validator = DataQualityValidator(clean_df, expected_schema=expected_schema)
        report = validator.validate()

        schema_checks = [c for c in report.checks if 'Schema' in c.name]
        assert len(schema_checks) > 0
        assert schema_checks[0].passed is False

    def test_validation_check_structure(self, clean_df):
        validator = DataQualityValidator(clean_df)
        report = validator.validate()

        for check in report.checks:
            assert isinstance(check, ValidationCheck)
            assert hasattr(check, 'name')
            assert hasattr(check, 'passed')
            assert hasattr(check, 'message')

    def test_report_is_invalid_when_check_fails(self, df_with_duplicates):
        validator = DataQualityValidator(df_with_duplicates)
        report = validator.validate()

        assert report.is_valid is False

    def test_report_contains_multiple_checks(self, clean_df):
        validator = DataQualityValidator(clean_df)
        report = validator.validate()

        assert len(report.checks) >= 4

    def test_check_details_provided_on_failure(self, df_with_duplicates):
        validator = DataQualityValidator(df_with_duplicates)
        report = validator.validate()

        failed_checks = [c for c in report.checks if not c.passed]
        if failed_checks:
            assert failed_checks[0].details is not None

    def test_validator_with_empty_schema(self, clean_df):
        validator = DataQualityValidator(clean_df, expected_schema={})
        report = validator.validate()

        schema_checks = [c for c in report.checks if 'Schema' in c.name]
        assert schema_checks[0].passed is True
