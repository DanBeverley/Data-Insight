"""Context-Aware Data Quality Assessment System for DataInsight AI"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from datetime import datetime

from .anomaly_detector import MultiLayerAnomalyDetector, AnomalyResult
from .drift_monitor import ComprehensiveDriftMonitor, DriftResult
from scipy import stats

class QualityDimension(Enum):
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy" 
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"

@dataclass
class QualityScore:
    dimension: QualityDimension
    score: float  # 0-100 quality score
    weight: float  # Importance weight for overall score
    issues: List[str]  # Specific quality issues found
    recommendations: List[str]  # Actionable recommendations
    metadata: Dict[str, Any]

@dataclass
class DataQualityReport:
    overall_score: float  # 0-100 weighted overall quality score
    dimension_scores: Dict[QualityDimension, QualityScore]
    sample_size: int
    features_analyzed: int
    anomaly_count: int
    drift_detected: bool
    critical_issues: List[str]
    recommendations: List[str]
    assessment_timestamp: datetime
    metadata: Dict[str, Any]

class ContextAwareQualityAssessor:
    """Comprehensive data quality assessment with contextual intelligence."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.anomaly_detector = MultiLayerAnomalyDetector()
        self.drift_monitor = ComprehensiveDriftMonitor()
        self.quality_history: List[DataQualityReport] = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for quality assessment."""
        return {
            'dimension_weights': {
                'completeness': 0.25,
                'accuracy': 0.20,
                'consistency': 0.15,
                'validity': 0.15,
                'timeliness': 0.15,
                'uniqueness': 0.10
            },
            'completeness_thresholds': {
                'excellent': 0.95,
                'good': 0.85,
                'acceptable': 0.70,
                'poor': 0.50
            },
            'accuracy_outlier_threshold': 0.05,  # 5% outlier tolerance
            'consistency_correlation_threshold': 0.3,
            'validity_pattern_match_threshold': 0.8,
            'uniqueness_duplicate_threshold': 0.02,  # 2% duplicate tolerance
            'min_samples_for_assessment': 10,
            'critical_score_threshold': 30,  # Below 30 = critical issues
            'anomaly_impact_weight': 0.3,  # How much anomalies impact score
            'drift_impact_weight': 0.2   # How much drift impacts score
        }
    
    def assess_quality(self, df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None,
                      context: Optional[Dict[str, Any]] = None) -> DataQualityReport:
        """
        Comprehensive data quality assessment with contextual awareness.
        
        Args:
            df: DataFrame to assess
            reference_df: Optional reference DataFrame for comparison
            context: Optional context information (domain, data type, etc.)
            
        Returns:
            Comprehensive data quality report
        """
        logging.info(f"Starting data quality assessment on {len(df)} samples, {len(df.columns)} features")
        
        if len(df) < self.config['min_samples_for_assessment']:
            raise ValueError(f"Insufficient samples for assessment. Need at least {self.config['min_samples_for_assessment']}")
        
        timestamp = datetime.now()
        context = context or {}
        
        # Assess each quality dimension
        dimension_scores = {}
        
        # 1. Completeness Assessment
        dimension_scores[QualityDimension.COMPLETENESS] = self._assess_completeness(df, context)
        
        # 2. Accuracy Assessment (outliers, anomalies)
        dimension_scores[QualityDimension.ACCURACY] = self._assess_accuracy(df, reference_df, context)
        
        # 3. Consistency Assessment
        dimension_scores[QualityDimension.CONSISTENCY] = self._assess_consistency(df, context)
        
        # 4. Validity Assessment
        dimension_scores[QualityDimension.VALIDITY] = self._assess_validity(df, context)
        
        # 5. Timeliness Assessment
        dimension_scores[QualityDimension.TIMELINESS] = self._assess_timeliness(df, context)
        
        # 6. Uniqueness Assessment
        dimension_scores[QualityDimension.UNIQUENESS] = self._assess_uniqueness(df, context)
        
        # Calculate overall weighted score
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # Detect anomalies and drift
        anomaly_results, anomaly_summary = [], {}
        drift_results, drift_detected = [], False
        
        try:
            anomaly_results = self.anomaly_detector.detect_anomalies(df, reference_df)
            anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        except Exception as e:
            logging.warning(f"Anomaly detection failed: {e}")
        
        try:
            if reference_df is not None:
                self.drift_monitor.fit_reference(reference_df)
                drift_results = self.drift_monitor.detect_drift(df)
                drift_detected = len(drift_results) > 0
        except Exception as e:
            logging.warning(f"Drift detection failed: {e}")
        
        # Apply anomaly and drift penalties to overall score
        anomaly_penalty = self._calculate_anomaly_penalty(anomaly_summary)
        drift_penalty = self._calculate_drift_penalty(drift_results)
        
        adjusted_overall_score = max(0, overall_score - anomaly_penalty - drift_penalty)
        
        # Generate critical issues and recommendations
        critical_issues = self._identify_critical_issues(dimension_scores, anomaly_results, drift_results)
        recommendations = self._generate_recommendations(dimension_scores, critical_issues, context)
        
        # Create comprehensive report
        report = DataQualityReport(
            overall_score=round(adjusted_overall_score, 2),
            dimension_scores=dimension_scores,
            sample_size=len(df),
            features_analyzed=len(df.columns),
            anomaly_count=anomaly_summary.get('total_anomalies', 0),
            drift_detected=drift_detected,
            critical_issues=critical_issues,
            recommendations=recommendations,
            assessment_timestamp=timestamp,
            metadata={
                'context': context,
                'anomaly_summary': anomaly_summary,
                'drift_summary': {'drift_count': len(drift_results), 'drift_results': [d.description for d in drift_results]},
                'anomaly_penalty': anomaly_penalty,
                'drift_penalty': drift_penalty,
                'config': self.config
            }
        )
        
        # Store in history
        self.quality_history.append(report)
        
        logging.info(f"Quality assessment complete. Overall score: {adjusted_overall_score:.2f}/100")
        return report
    
    def _assess_completeness(self, df: pd.DataFrame, context: Dict[str, Any]) -> QualityScore:
        """Assess data completeness."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells)
        
        # Calculate per-column completeness
        column_completeness = {}
        critical_columns = []
        
        for column in df.columns:
            col_completeness = 1 - (df[column].isnull().sum() / len(df))
            column_completeness[column] = col_completeness
            
            # Identify critical columns with high missing rates
            if col_completeness < 0.5:
                critical_columns.append(column)
        
        # Score calculation (0-100)
        score = completeness_ratio * 100
        
        # Issues identification
        issues = []
        if completeness_ratio < self.config['completeness_thresholds']['poor']:
            issues.append(f"Critical: {missing_cells:,} missing values ({(1-completeness_ratio)*100:.1f}%)")
        
        if critical_columns:
            issues.append(f"High missing rates in columns: {', '.join(critical_columns)}")
        
        # Missing pattern analysis
        missing_patterns = self._analyze_missing_patterns(df)
        if missing_patterns:
            issues.extend(missing_patterns)
        
        # Recommendations
        recommendations = []
        if completeness_ratio < 0.9:
            recommendations.append("Implement data validation at source to prevent missing values")
            recommendations.append("Consider imputation strategies for missing values")
        
        if critical_columns:
            recommendations.append(f"Prioritize data collection for critical columns: {', '.join(critical_columns[:3])}")
        
        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=round(score, 2),
            weight=self.config['dimension_weights']['completeness'],
            issues=issues,
            recommendations=recommendations,
            metadata={
                'completeness_ratio': completeness_ratio,
                'missing_cells': missing_cells,
                'total_cells': total_cells,
                'column_completeness': column_completeness,
                'critical_columns': critical_columns
            }
        )
    
    def _assess_accuracy(self, df: pd.DataFrame, reference_df: Optional[pd.DataFrame], 
                        context: Dict[str, Any]) -> QualityScore:
        """Assess data accuracy through outlier and anomaly detection."""
        issues = []
        recommendations = []
        
        # Basic outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        outlier_details = {}
        
        for column in numeric_cols:
            series = df[column].dropna()
            if len(series) < 10:
                continue
                
            # IQR method
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                outlier_mask = (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)
                column_outliers = outlier_mask.sum()
                total_outliers += column_outliers
                
                if column_outliers > 0:
                    outlier_details[column] = {
                        'count': column_outliers,
                        'percentage': (column_outliers / len(series)) * 100
                    }
        
        # Accuracy score based on outlier proportion
        total_numeric_values = len(df[numeric_cols].stack().dropna()) if len(numeric_cols) > 0 else 1
        outlier_ratio = total_outliers / total_numeric_values
        
        # Score: higher outlier ratio = lower accuracy
        base_score = max(0, 100 - (outlier_ratio * 500))  # Scale outlier impact
        
        # Adjust based on anomaly detection results if available
        anomaly_adjustment = 0
        try:
            anomaly_results = self.anomaly_detector.detect_anomalies(df, reference_df)
            if anomaly_results:
                high_severity_anomalies = [r for r in anomaly_results if r.severity in ['high', 'critical']]
                anomaly_adjustment = len(high_severity_anomalies) * 5  # -5 points per high severity anomaly
        except Exception as e:
            logging.warning(f"Anomaly detection failed during accuracy assessment: {e}")
        
        final_score = max(0, base_score - anomaly_adjustment)
        
        # Generate issues
        if outlier_ratio > self.config['accuracy_outlier_threshold']:
            issues.append(f"High outlier rate: {outlier_ratio*100:.2f}% of numeric values")
            
        if outlier_details:
            worst_columns = sorted(outlier_details.items(), 
                                 key=lambda x: x[1]['percentage'], reverse=True)[:3]
            issues.append(f"Highest outlier columns: {', '.join([f'{col}({info[\"percentage\"]:.1f}%)' for col, info in worst_columns])}")
        
        # Generate recommendations
        if outlier_ratio > 0.1:
            recommendations.append("Implement outlier detection and handling in data preprocessing")
            recommendations.append("Review data collection processes for accuracy issues")
            
        if outlier_details:
            recommendations.append("Consider robust statistical methods that handle outliers")
        
        return QualityScore(
            dimension=QualityDimension.ACCURACY,
            score=round(final_score, 2),
            weight=self.config['dimension_weights']['accuracy'],
            issues=issues,
            recommendations=recommendations,
            metadata={
                'outlier_ratio': outlier_ratio,
                'total_outliers': total_outliers,
                'outlier_details': outlier_details,
                'anomaly_adjustment': anomaly_adjustment
            }
        )
    
    def _assess_consistency(self, df: pd.DataFrame, context: Dict[str, Any]) -> QualityScore:
        """Assess data consistency through correlation and pattern analysis."""
        issues = []
        recommendations = []
        
        # Numeric correlation consistency
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inconsistency_score = 0
        correlation_issues = []
        
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = df[numeric_cols].corr()
                
                # Look for unexpectedly low correlations between related features
                # This is domain-specific, but we can check for basic patterns
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        corr_value = abs(corr_matrix.loc[col1, col2])
                        
                        # If column names suggest they should be related
                        if self._are_columns_related(col1, col2) and corr_value < self.config['consistency_correlation_threshold']:
                            correlation_issues.append(f"Low correlation between {col1} and {col2}: {corr_value:.3f}")
                            inconsistency_score += 10
                
            except Exception as e:
                logging.warning(f"Correlation analysis failed: {e}")
        
        # Categorical consistency
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_issues = []
        
        for column in categorical_cols:
            series = df[column].dropna()
            if len(series) < 10:
                continue
                
            # Check for similar values that might indicate inconsistent formatting
            value_counts = series.value_counts()
            similar_values = self._find_similar_categorical_values(value_counts.index.tolist())
            
            if similar_values:
                categorical_issues.extend([f"Inconsistent formatting in {column}: {', '.join(group)}" 
                                        for group in similar_values[:3]])  # Limit to first 3
                inconsistency_score += len(similar_values) * 5
        
        # Data type consistency
        dtype_issues = self._check_dtype_consistency(df)
        if dtype_issues:
            inconsistency_score += len(dtype_issues) * 5
        
        # Calculate consistency score
        base_score = 100
        final_score = max(0, base_score - inconsistency_score)
        
        # Compile issues
        issues.extend(correlation_issues)
        issues.extend(categorical_issues)
        issues.extend(dtype_issues)
        
        # Generate recommendations
        if correlation_issues:
            recommendations.append("Review data relationships and investigate unexpected correlations")
            
        if categorical_issues:
            recommendations.append("Implement standardized categorical value formatting")
            recommendations.append("Use data validation rules for categorical inputs")
            
        if dtype_issues:
            recommendations.append("Enforce consistent data types across similar fields")
        
        return QualityScore(
            dimension=QualityDimension.CONSISTENCY,
            score=round(final_score, 2),
            weight=self.config['dimension_weights']['consistency'],
            issues=issues,
            recommendations=recommendations,
            metadata={
                'inconsistency_score': inconsistency_score,
                'correlation_issues_count': len(correlation_issues),
                'categorical_issues_count': len(categorical_issues),
                'dtype_issues_count': len(dtype_issues)
            }
        )
    
    def _assess_validity(self, df: pd.DataFrame, context: Dict[str, Any]) -> QualityScore:
        """Assess data validity through format and range checks."""
        issues = []
        recommendations = []
        validity_violations = 0
        
        # Numeric range validation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        range_issues = []
        
        for column in numeric_cols:
            series = df[column].dropna()
            if len(series) == 0:
                continue
                
            # Basic range validation (negative values where they shouldn't be)
            if self._should_be_positive(column):
                negative_count = (series < 0).sum()
                if negative_count > 0:
                    range_issues.append(f"{column}: {negative_count} negative values")
                    validity_violations += negative_count
            
            # Check for impossible values (e.g., percentages > 100)
            if self._is_percentage_column(column):
                invalid_percentage = ((series < 0) | (series > 100)).sum()
                if invalid_percentage > 0:
                    range_issues.append(f"{column}: {invalid_percentage} values outside 0-100%")
                    validity_violations += invalid_percentage
        
        # Text pattern validation
        text_cols = df.select_dtypes(include=['object']).columns
        pattern_issues = []
        
        for column in text_cols:
            series = df[column].dropna().astype(str)
            if len(series) == 0:
                continue
                
            # Email validation
            if self._is_email_column(column):
                invalid_emails = self._validate_email_pattern(series)
                if invalid_emails > 0:
                    pattern_issues.append(f"{column}: {invalid_emails} invalid email formats")
                    validity_violations += invalid_emails
            
            # Phone number validation
            elif self._is_phone_column(column):
                invalid_phones = self._validate_phone_pattern(series)
                if invalid_phones > 0:
                    pattern_issues.append(f"{column}: {invalid_phones} invalid phone formats")
                    validity_violations += invalid_phones
            
            # Date format validation
            elif self._is_date_column(column):
                invalid_dates = self._validate_date_format(series)
                if invalid_dates > 0:
                    pattern_issues.append(f"{column}: {invalid_dates} invalid date formats")
                    validity_violations += invalid_dates
        
        # Calculate validity score
        total_values = len(df.stack().dropna())
        validity_ratio = 1 - (validity_violations / total_values) if total_values > 0 else 1
        score = validity_ratio * 100
        
        # Compile issues
        issues.extend(range_issues)
        issues.extend(pattern_issues)
        
        # Generate recommendations
        if range_issues:
            recommendations.append("Implement range validation rules for numeric fields")
            recommendations.append("Add data entry constraints to prevent invalid ranges")
            
        if pattern_issues:
            recommendations.append("Implement format validation for structured text fields")
            recommendations.append("Use input masks and validation in data entry forms")
            
        if validity_violations > 0:
            recommendations.append("Establish data quality rules and validation pipelines")
        
        return QualityScore(
            dimension=QualityDimension.VALIDITY,
            score=round(score, 2),
            weight=self.config['dimension_weights']['validity'],
            issues=issues,
            recommendations=recommendations,
            metadata={
                'validity_violations': validity_violations,
                'total_values': total_values,
                'validity_ratio': validity_ratio,
                'range_issues_count': len(range_issues),
                'pattern_issues_count': len(pattern_issues)
            }
        )
    
    def _assess_timeliness(self, df: pd.DataFrame, context: Dict[str, Any]) -> QualityScore:
        """Assess data timeliness."""
        issues = []
        recommendations = []
        
        # Look for timestamp/date columns
        date_cols = self._identify_date_columns(df)
        
        if not date_cols:
            # No date columns found, assume data is reasonably timely
            return QualityScore(
                dimension=QualityDimension.TIMELINESS,
                score=80.0,  # Neutral score when no date info available
                weight=self.config['dimension_weights']['timeliness'],
                issues=["No timestamp columns found for timeliness analysis"],
                recommendations=["Consider adding timestamp columns to track data freshness"],
                metadata={'date_columns_found': 0}
            )
        
        timeliness_issues = []
        outdated_records = 0
        
        current_time = datetime.now()
        
        for column in date_cols:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[column]):
                    date_series = pd.to_datetime(df[column], errors='coerce')
                else:
                    date_series = df[column]
                
                date_series = date_series.dropna()
                if len(date_series) == 0:
                    continue
                
                # Check for future dates (data integrity issue)
                future_dates = (date_series > current_time).sum()
                if future_dates > 0:
                    timeliness_issues.append(f"{column}: {future_dates} future dates detected")
                
                # Check for very old data (context dependent)
                days_threshold = context.get('max_age_days', 365)  # Default 1 year
                age_threshold = current_time - pd.Timedelta(days=days_threshold)
                old_records = (date_series < age_threshold).sum()
                outdated_records += old_records
                
                if old_records > 0:
                    timeliness_issues.append(f"{column}: {old_records} records older than {days_threshold} days")
                
            except Exception as e:
                logging.warning(f"Failed to process date column {column}: {e}")
                continue
        
        # Calculate timeliness score
        total_records = len(df)
        freshness_ratio = 1 - (outdated_records / total_records) if total_records > 0 else 1
        score = freshness_ratio * 100
        
        # Apply penalty for future dates
        future_penalty = min(20, len([issue for issue in timeliness_issues if 'future dates' in issue]) * 5)
        final_score = max(0, score - future_penalty)
        
        issues.extend(timeliness_issues)
        
        # Generate recommendations
        if outdated_records > 0:
            recommendations.append("Implement data refresh processes to maintain currency")
            recommendations.append("Establish data retention policies")
            
        if future_penalty > 0:
            recommendations.append("Add validation to prevent future dates in historical data")
        
        return QualityScore(
            dimension=QualityDimension.TIMELINESS,
            score=round(final_score, 2),
            weight=self.config['dimension_weights']['timeliness'],
            issues=issues,
            recommendations=recommendations,
            metadata={
                'date_columns_found': len(date_cols),
                'outdated_records': outdated_records,
                'freshness_ratio': freshness_ratio,
                'future_penalty': future_penalty
            }
        )
    
    def _assess_uniqueness(self, df: pd.DataFrame, context: Dict[str, Any]) -> QualityScore:
        """Assess data uniqueness and identify duplicates."""
        issues = []
        recommendations = []
        
        # Overall duplicate detection
        total_records = len(df)
        
        # Exact duplicates
        exact_duplicates = df.duplicated().sum()
        
        # Near-duplicates (for text fields)
        near_duplicates = self._detect_near_duplicates(df)
        
        total_duplicates = exact_duplicates + near_duplicates
        uniqueness_ratio = 1 - (total_duplicates / total_records)
        
        # Calculate uniqueness score
        score = uniqueness_ratio * 100
        
        # Per-column uniqueness analysis
        column_uniqueness = {}
        uniqueness_issues = []
        
        for column in df.columns:
            series = df[column].dropna()
            if len(series) == 0:
                continue
                
            unique_ratio = len(series.unique()) / len(series)
            column_uniqueness[column] = unique_ratio
            
            # Flag columns with unexpectedly low uniqueness
            if self._should_be_unique(column) and unique_ratio < 0.9:
                uniqueness_issues.append(f"{column}: Only {unique_ratio*100:.1f}% unique values")
        
        # Issues compilation
        if exact_duplicates > 0:
            issues.append(f"{exact_duplicates} exact duplicate records ({exact_duplicates/total_records*100:.1f}%)")
            
        if near_duplicates > 0:
            issues.append(f"{near_duplicates} near-duplicate records detected")
            
        issues.extend(uniqueness_issues)
        
        # Recommendations
        if total_duplicates > total_records * self.config['uniqueness_duplicate_threshold']:
            recommendations.append("Implement duplicate detection and removal processes")
            recommendations.append("Add unique constraints to prevent duplicate entries")
            
        if uniqueness_issues:
            recommendations.append("Review data collection for fields that should be unique")
        
        return QualityScore(
            dimension=QualityDimension.UNIQUENESS,
            score=round(score, 2),
            weight=self.config['dimension_weights']['uniqueness'],
            issues=issues,
            recommendations=recommendations,
            metadata={
                'exact_duplicates': exact_duplicates,
                'near_duplicates': near_duplicates,
                'total_duplicates': total_duplicates,
                'uniqueness_ratio': uniqueness_ratio,
                'column_uniqueness': column_uniqueness
            }
        )
    
    def _calculate_overall_score(self, dimension_scores: Dict[QualityDimension, QualityScore]) -> float:
        """Calculate weighted overall quality score."""
        weighted_sum = 0
        total_weight = 0
        
        for dimension, score_obj in dimension_scores.items():
            weighted_sum += score_obj.score * score_obj.weight
            total_weight += score_obj.weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _calculate_anomaly_penalty(self, anomaly_summary: Dict[str, Any]) -> float:
        """Calculate penalty based on anomaly detection results."""
        if not anomaly_summary or anomaly_summary.get('total_anomalies', 0) == 0:
            return 0
        
        total_anomalies = anomaly_summary['total_anomalies']
        by_severity = anomaly_summary.get('by_severity', {})
        
        penalty = 0
        penalty += by_severity.get('critical', 0) * 10  # -10 points per critical
        penalty += by_severity.get('high', 0) * 5       # -5 points per high
        penalty += by_severity.get('medium', 0) * 2     # -2 points per medium
        penalty += by_severity.get('low', 0) * 0.5      # -0.5 points per low
        
        # Cap penalty impact
        return min(penalty * self.config['anomaly_impact_weight'], 30)
    
    def _calculate_drift_penalty(self, drift_results: List[DriftResult]) -> float:
        """Calculate penalty based on drift detection results."""
        if not drift_results:
            return 0
        
        penalty = 0
        for drift in drift_results:
            if drift.severity == 'critical':
                penalty += 10
            elif drift.severity == 'high':
                penalty += 5
            elif drift.severity == 'medium':
                penalty += 2
            else:  # low
                penalty += 0.5
        
        # Cap penalty impact
        return min(penalty * self.config['drift_impact_weight'], 20)
    
    def _identify_critical_issues(self, dimension_scores: Dict[QualityDimension, QualityScore],
                                 anomaly_results: List[AnomalyResult], 
                                 drift_results: List[DriftResult]) -> List[str]:
        """Identify critical quality issues."""
        critical_issues = []
        
        # Check dimension scores
        for dimension, score_obj in dimension_scores.items():
            if score_obj.score < self.config['critical_score_threshold']:
                critical_issues.append(f"Critical {dimension.value} issues (score: {score_obj.score:.1f})")
        
        # Check anomaly severity
        critical_anomalies = [a for a in anomaly_results if a.severity == 'critical']
        if critical_anomalies:
            critical_issues.append(f"{len(critical_anomalies)} critical anomalies detected")
        
        # Check drift severity
        critical_drifts = [d for d in drift_results if d.severity == 'critical']
        if critical_drifts:
            critical_issues.append(f"{len(critical_drifts)} critical data drifts detected")
        
        return critical_issues
    
    def _generate_recommendations(self, dimension_scores: Dict[QualityDimension, QualityScore],
                                 critical_issues: List[str], context: Dict[str, Any]) -> List[str]:
        """Generate actionable quality improvement recommendations."""
        recommendations = []
        
        # Collect all dimension recommendations
        for score_obj in dimension_scores.values():
            recommendations.extend(score_obj.recommendations)
        
        # Add priority recommendations for critical issues
        if critical_issues:
            recommendations.insert(0, "URGENT: Address critical quality issues before production use")
        
        # Context-specific recommendations
        domain = context.get('domain', '').lower()
        if 'finance' in domain or 'financial' in domain:
            recommendations.append("Implement financial data validation rules (currency, precision)")
        elif 'healthcare' in domain or 'medical' in domain:
            recommendations.append("Ensure compliance with healthcare data quality standards")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations[:10]  # Limit to top 10 recommendations
    
    # Helper methods for quality assessment
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> List[str]:
        """Analyze patterns in missing data."""
        patterns = []
        
        # Check for columns with identical missing patterns
        missing_matrix = df.isnull()
        
        for col1 in missing_matrix.columns:
            for col2 in missing_matrix.columns:
                if col1 >= col2:  # Avoid duplicate comparisons
                    continue
                    
                if missing_matrix[col1].equals(missing_matrix[col2]):
                    patterns.append(f"Identical missing patterns: {col1} and {col2}")
        
        return patterns[:3]  # Limit to first 3 patterns
    
    def _are_columns_related(self, col1: str, col2: str) -> bool:
        """Simple heuristic to check if columns might be related."""
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # Check for common related patterns
        related_pairs = [
            ('price', 'cost'), ('amount', 'value'), ('start', 'end'),
            ('min', 'max'), ('first', 'last'), ('begin', 'finish')
        ]
        
        for word1, word2 in related_pairs:
            if (word1 in col1_lower and word2 in col2_lower) or (word2 in col1_lower and word1 in col2_lower):
                return True
        
        return False
    
    def _find_similar_categorical_values(self, values: List[str]) -> List[List[str]]:
        """Find groups of similar categorical values (basic string matching)."""
        similar_groups = []
        processed = set()
        
        for i, value1 in enumerate(values):
            if value1 in processed:
                continue
                
            similar = [value1]
            for value2 in values[i+1:]:
                if value2 in processed:
                    continue
                    
                # Simple similarity check (case-insensitive, ignoring spaces)
                v1_clean = value1.lower().replace(' ', '')
                v2_clean = value2.lower().replace(' ', '')
                
                if v1_clean == v2_clean and value1 != value2:
                    similar.append(value2)
                    processed.add(value2)
            
            if len(similar) > 1:
                similar_groups.append(similar)
                processed.update(similar)
        
        return similar_groups
    
    def _check_dtype_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check for data type consistency issues."""
        issues = []
        
        # Look for columns that might have inconsistent types
        for column in df.columns:
            series = df[column].dropna()
            if len(series) == 0:
                continue
            
            # For object columns, check if they could be numeric
            if series.dtype == 'object':
                try:
                    pd.to_numeric(series, errors='raise')
                    issues.append(f"{column}: Object type but contains only numeric values")
                except (ValueError, TypeError):
                    pass
        
        return issues
    
    def _should_be_positive(self, column_name: str) -> bool:
        """Heuristic to determine if a column should contain only positive values."""
        positive_indicators = ['count', 'quantity', 'amount', 'price', 'cost', 'age', 'size', 'length', 'weight']
        column_lower = column_name.lower()
        return any(indicator in column_lower for indicator in positive_indicators)
    
    def _is_percentage_column(self, column_name: str) -> bool:
        """Heuristic to identify percentage columns."""
        percentage_indicators = ['percent', 'pct', 'percentage', 'rate', 'ratio']
        column_lower = column_name.lower()
        return any(indicator in column_lower for indicator in percentage_indicators)
    
    def _is_email_column(self, column_name: str) -> bool:
        """Heuristic to identify email columns."""
        return 'email' in column_name.lower()
    
    def _is_phone_column(self, column_name: str) -> bool:
        """Heuristic to identify phone columns."""
        phone_indicators = ['phone', 'mobile', 'cell', 'telephone']
        column_lower = column_name.lower()
        return any(indicator in column_lower for indicator in phone_indicators)
    
    def _is_date_column(self, column_name: str) -> bool:
        """Heuristic to identify date columns."""
        date_indicators = ['date', 'time', 'created', 'updated', 'modified']
        column_lower = column_name.lower()
        return any(indicator in column_lower for indicator in date_indicators)
    
    def _validate_email_pattern(self, series: pd.Series) -> int:
        """Validate email patterns."""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        invalid_count = (~series.str.match(email_pattern, na=False)).sum()
        return invalid_count
    
    def _validate_phone_pattern(self, series: pd.Series) -> int:
        """Validate phone number patterns."""
        # Simple pattern for various phone formats
        phone_pattern = r'^\+?[\d\s\-\(\)]{10,}$'
        invalid_count = (~series.str.match(phone_pattern, na=False)).sum()
        return invalid_count
    
    def _validate_date_format(self, series: pd.Series) -> int:
        """Validate date formats."""
        invalid_count = 0
        for value in series:
            try:
                pd.to_datetime(value)
            except:
                invalid_count += 1
        return invalid_count
    
    def _identify_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that contain date/time information."""
        date_columns = []
        
        for column in df.columns:
            # Check dtype
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                date_columns.append(column)
                continue
            
            # Check column name
            if self._is_date_column(column):
                date_columns.append(column)
                continue
            
            # Sample-based detection for object columns
            if df[column].dtype == 'object':
                sample = df[column].dropna().head(10)
                if len(sample) > 0:
                    date_like_count = 0
                    for value in sample:
                        try:
                            pd.to_datetime(value)
                            date_like_count += 1
                        except:
                            pass
                    
                    if date_like_count / len(sample) > 0.7:  # 70% look like dates
                        date_columns.append(column)
        
        return date_columns
    
    def _detect_near_duplicates(self, df: pd.DataFrame) -> int:
        """Simple near-duplicate detection for text fields."""
        near_duplicates = 0
        
        text_cols = df.select_dtypes(include=['object']).columns
        
        for column in text_cols:
            series = df[column].dropna().astype(str)
            if len(series) < 2:
                continue
            
            # Simple approach: check for very similar strings
            values = series.unique()
            if len(values) < 2:
                continue
            
            for i, val1 in enumerate(values):
                for val2 in values[i+1:]:
                    # Simple similarity: same length and >90% character overlap
                    if len(val1) == len(val2) and len(val1) > 5:
                        matches = sum(c1 == c2 for c1, c2 in zip(val1, val2))
                        if matches / len(val1) > 0.9:
                            near_duplicates += 1
                            break
        
        return near_duplicates
    
    def _should_be_unique(self, column_name: str) -> bool:
        """Heuristic to determine if column should have unique values."""
        unique_indicators = ['id', 'key', 'identifier', 'uuid', 'email']
        column_lower = column_name.lower()
        return any(indicator in column_lower for indicator in unique_indicators)
    
    def get_quality_trend(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """Get quality trend analysis from historical assessments."""
        if len(self.quality_history) < 2:
            return {'message': 'Insufficient historical data for trend analysis'}
        
        recent_reports = self.quality_history
        if time_window:
            recent_reports = recent_reports[-time_window:]
        
        if len(recent_reports) < 2:
            return {'message': 'Insufficient data in specified time window'}
        
        # Calculate trends
        scores = [report.overall_score for report in recent_reports]
        timestamps = [report.assessment_timestamp for report in recent_reports]
        
        # Simple trend calculation
        if len(scores) >= 2:
            trend = scores[-1] - scores[0]
            avg_score = np.mean(scores)
        else:
            trend = 0
            avg_score = scores[0] if scores else 0
        
        return {
            'trend_direction': 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable',
            'trend_magnitude': abs(trend),
            'average_score': round(avg_score, 2),
            'latest_score': scores[-1] if scores else 0,
            'assessments_count': len(recent_reports),
            'time_span': str(timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 'N/A'
        }

def assess_data_quality(df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None,
                       context: Optional[Dict[str, Any]] = None,
                       config: Optional[Dict[str, Any]] = None) -> DataQualityReport:
    """
    Convenience function for comprehensive data quality assessment.
    
    Args:
        df: DataFrame to assess
        reference_df: Optional reference DataFrame for comparison
        context: Optional context information
        config: Optional configuration overrides
        
    Returns:
        Comprehensive data quality report
    """
    assessor = ContextAwareQualityAssessor(config)
    return assessor.assess_quality(df, reference_df, context)