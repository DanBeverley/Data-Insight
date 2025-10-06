import pytest
import pandas as pd
from src.intelligence.data_profiler import IntelligentDataProfiler, SemanticType


@pytest.mark.unit
class TestIntelligentDataProfiler:

    def test_profiler_initialization(self):
        profiler = IntelligentDataProfiler()
        assert profiler is not None
        assert hasattr(profiler, 'patterns')
        assert hasattr(profiler, 'domain_keywords')

    def test_profile_dataset_structure(self, housing_dataset):
        profiler = IntelligentDataProfiler()
        result = profiler.profile_dataset(housing_dataset)

        assert 'column_profiles' in result
        assert 'domain_analysis' in result
        assert 'relationship_analysis' in result
        assert 'overall_recommendations' in result

    def test_column_profiles_created(self, housing_dataset):
        profiler = IntelligentDataProfiler()
        result = profiler.profile_dataset(housing_dataset)

        column_profiles = result['column_profiles']
        assert len(column_profiles) == len(housing_dataset.columns)

        for col in housing_dataset.columns:
            assert col in column_profiles
            profile = column_profiles[col]
            assert profile.column == col
            assert isinstance(profile.semantic_type, SemanticType)
            assert 0 <= profile.confidence <= 1

    def test_detect_numeric_columns(self, housing_dataset):
        profiler = IntelligentDataProfiler()
        result = profiler.profile_dataset(housing_dataset)

        price_profile = result['column_profiles']['price']
        assert price_profile.semantic_type == SemanticType.CURRENCY

    def test_detect_categorical_columns(self, housing_dataset):
        profiler = IntelligentDataProfiler()
        result = profiler.profile_dataset(housing_dataset)

        furnishing_profile = result['column_profiles']['furnishingstatus']
        assert furnishing_profile.column == 'furnishingstatus'
        assert furnishing_profile.evidence['unique_count'] == 3
        assert isinstance(furnishing_profile.semantic_type, SemanticType)

    def test_evidence_in_profile(self, housing_dataset):
        profiler = IntelligentDataProfiler()
        result = profiler.profile_dataset(housing_dataset)

        for col, profile in result['column_profiles'].items():
            assert 'dtype' in profile.evidence
            assert 'null_ratio' in profile.evidence
            assert 'cardinality' in profile.evidence
            assert 'unique_count' in profile.evidence

    def test_recommendations_generated(self, housing_dataset):
        profiler = IntelligentDataProfiler()
        result = profiler.profile_dataset(housing_dataset)

        for col, profile in result['column_profiles'].items():
            assert isinstance(profile.recommendations, list)

    def test_domain_analysis_structure(self, housing_dataset):
        profiler = IntelligentDataProfiler()
        result = profiler.profile_dataset(housing_dataset)

        domain_analysis = result['domain_analysis']
        assert 'detected_domains' in domain_analysis
        assert 'recommendations' in domain_analysis

    def test_relationship_analysis_structure(self, housing_dataset):
        profiler = IntelligentDataProfiler()
        result = profiler.profile_dataset(housing_dataset)

        relationship_analysis = result['relationship_analysis']
        assert 'relationships' in relationship_analysis
        assert 'relationship_graph' in relationship_analysis
        assert 'recommendations' in relationship_analysis

    def test_profile_time_series_data(self, coffee_sales_dataset):
        profiler = IntelligentDataProfiler()
        result = profiler.profile_dataset(coffee_sales_dataset)

        date_profile = result['column_profiles']['date']
        assert date_profile.semantic_type in [
            SemanticType.DATETIME_DATE,
            SemanticType.DATETIME_TIMESTAMP,
            SemanticType.DATETIME_PARTIAL,
            SemanticType.NATURAL_KEY
        ] or date_profile.evidence.get('dtype') == 'datetime64[ns]'
