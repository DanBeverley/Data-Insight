"""Multi-Layer Anomaly Detection System for DataInsight AI"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import mahalanobis

class AnomalyType(Enum):
    STATISTICAL = "statistical"
    DISTRIBUTION = "distribution"
    PATTERN = "pattern"
    MULTIVARIATE = "multivariate"

@dataclass
class AnomalyResult:
    anomaly_type: AnomalyType
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float  # 0-1 confidence score
    affected_features: List[str]
    anomaly_indices: List[int]
    description: str
    metadata: Dict[str, Any]

class MultiLayerAnomalyDetector:
    """Comprehensive anomaly detection with multiple detection strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.detection_results: List[AnomalyResult] = []
        self.feature_stats: Dict[str, Dict] = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for anomaly detection."""
        return {
            'statistical_threshold': 3.0,  # Modified Z-score threshold
            'iqr_multiplier': 1.5,  # IQR multiplier for outliers
            'isolation_contamination': 0.1,  # Expected outlier ratio
            'lof_n_neighbors': 20,  # LOF neighbors
            'min_samples_for_multivariate': 100,  # Minimum samples for multivariate analysis
            'confidence_threshold': 0.7,  # Minimum confidence to report
            'severity_thresholds': {
                'low': 0.7,
                'medium': 0.8, 
                'high': 0.9,
                'critical': 0.95
            }
        }
    
    def detect_anomalies(self, df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None) -> List[AnomalyResult]:
        """
        Comprehensive anomaly detection across multiple layers.
        
        Args:
            df: DataFrame to analyze for anomalies
            reference_df: Optional reference DataFrame for comparison
            
        Returns:
            List of detected anomalies with details
        """
        logging.info(f"Starting multi-layer anomaly detection on {len(df)} samples")
        
        self.detection_results = []
        self._compute_feature_statistics(df)
        
        # Layer 1: Statistical outlier detection
        self._detect_statistical_outliers(df)
        
        # Layer 2: Distribution-based detection
        if reference_df is not None:
            self._detect_distribution_anomalies(df, reference_df)
        
        # Layer 3: Pattern-based detection
        self._detect_pattern_anomalies(df)
        
        # Layer 4: Multivariate anomaly detection
        if len(df) >= self.config['min_samples_for_multivariate']:
            self._detect_multivariate_anomalies(df)
        
        # Filter by confidence and merge overlapping results
        filtered_results = self._filter_and_merge_results()
        
        logging.info(f"Detected {len(filtered_results)} anomalies across {len(self.detection_results)} initial detections")
        return filtered_results
    
    def _compute_feature_statistics(self, df: pd.DataFrame):
        """Compute and store feature statistics for analysis."""
        self.feature_stats = {}
        
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                series = df[column].dropna()
                if len(series) == 0:
                    continue
                    
                self.feature_stats[column] = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'median': series.median(),
                    'mad': stats.median_abs_deviation(series),  # Median Absolute Deviation
                    'q1': series.quantile(0.25),
                    'q3': series.quantile(0.75),
                    'iqr': series.quantile(0.75) - series.quantile(0.25),
                    'skewness': stats.skew(series),
                    'kurtosis': stats.kurtosis(series),
                    'min': series.min(),
                    'max': series.max()
                }
    
    def _detect_statistical_outliers(self, df: pd.DataFrame):
        """Detect outliers using statistical methods."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_cols:
            series = df[column].dropna()
            if len(series) < 10:  # Need minimum samples
                continue
                
            stats_data = self.feature_stats.get(column, {})
            if not stats_data:
                continue
            
            # Modified Z-score (more robust than standard Z-score)
            median = stats_data['median']
            mad = stats_data['mad']
            
            if mad == 0:  # Handle case where MAD is 0
                mad = stats_data['std'] * 0.6745  # Approximate MAD from std
                
            if mad > 0:
                modified_z_scores = 0.6745 * (series - median) / mad
                outlier_mask = np.abs(modified_z_scores) > self.config['statistical_threshold']
                
                if outlier_mask.any():
                    outlier_indices = series[outlier_mask].index.tolist()
                    severity = self._calculate_severity(modified_z_scores.max())
                    confidence = min(0.95, np.abs(modified_z_scores.max()) / 5.0)
                    
                    self.detection_results.append(AnomalyResult(
                        anomaly_type=AnomalyType.STATISTICAL,
                        severity=severity,
                        confidence=confidence,
                        affected_features=[column],
                        anomaly_indices=outlier_indices,
                        description=f"Statistical outliers in {column} using modified Z-score",
                        metadata={
                            'method': 'modified_z_score',
                            'threshold': self.config['statistical_threshold'],
                            'max_z_score': float(modified_z_scores.max()),
                            'outlier_count': int(outlier_mask.sum())
                        }
                    ))
            
            # IQR method
            q1, q3 = stats_data['q1'], stats_data['q3']
            iqr = stats_data['iqr']
            
            if iqr > 0:
                lower_bound = q1 - self.config['iqr_multiplier'] * iqr
                upper_bound = q3 + self.config['iqr_multiplier'] * iqr
                
                iqr_outliers = (series < lower_bound) | (series > upper_bound)
                
                if iqr_outliers.any():
                    outlier_indices = series[iqr_outliers].index.tolist()
                    # Calculate severity based on how far outside bounds
                    extreme_values = series[iqr_outliers]
                    max_deviation = max(
                        abs(extreme_values.min() - lower_bound) / iqr if extreme_values.min() < lower_bound else 0,
                        abs(extreme_values.max() - upper_bound) / iqr if extreme_values.max() > upper_bound else 0
                    )
                    severity = self._calculate_severity(max_deviation)
                    confidence = min(0.9, max_deviation / 3.0)
                    
                    self.detection_results.append(AnomalyResult(
                        anomaly_type=AnomalyType.STATISTICAL,
                        severity=severity,
                        confidence=confidence,
                        affected_features=[column],
                        anomaly_indices=outlier_indices,
                        description=f"IQR outliers in {column}",
                        metadata={
                            'method': 'iqr',
                            'multiplier': self.config['iqr_multiplier'],
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound),
                            'max_deviation': float(max_deviation),
                            'outlier_count': int(iqr_outliers.sum())
                        }
                    ))
    
    def _detect_distribution_anomalies(self, df: pd.DataFrame, reference_df: pd.DataFrame):
        """Detect anomalies by comparing distributions."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.intersection(
            reference_df.select_dtypes(include=[np.number]).columns
        )
        
        for column in numeric_cols:
            current_data = df[column].dropna()
            reference_data = reference_df[column].dropna()
            
            if len(current_data) < 30 or len(reference_data) < 30:
                continue
            
            # Kolmogorov-Smirnov test
            try:
                ks_statistic, ks_p_value = stats.ks_2samp(current_data, reference_data)
                
                if ks_p_value < 0.05:  # Significant difference
                    # Find most anomalous points using quantile comparison
                    ref_quantiles = np.percentile(reference_data, np.arange(0, 101, 5))
                    
                    anomalous_indices = []
                    for idx, value in current_data.items():
                        # Check if value falls in extreme quantiles compared to reference
                        percentile_in_ref = stats.percentileofscore(reference_data, value)
                        if percentile_in_ref < 5 or percentile_in_ref > 95:
                            anomalous_indices.append(idx)
                    
                    if anomalous_indices:
                        severity = self._calculate_severity_from_ks(ks_statistic)
                        confidence = 1.0 - ks_p_value
                        
                        self.detection_results.append(AnomalyResult(
                            anomaly_type=AnomalyType.DISTRIBUTION,
                            severity=severity,
                            confidence=confidence,
                            affected_features=[column],
                            anomaly_indices=anomalous_indices,
                            description=f"Distribution shift detected in {column}",
                            metadata={
                                'method': 'kolmogorov_smirnov',
                                'ks_statistic': float(ks_statistic),
                                'ks_p_value': float(ks_p_value),
                                'anomalous_count': len(anomalous_indices)
                            }
                        ))
            except Exception as e:
                logging.warning(f"Failed to compute KS test for {column}: {e}")
                continue
    
    def _detect_pattern_anomalies(self, df: pd.DataFrame):
        """Detect pattern-based anomalies."""
        
        # Categorical pattern anomalies
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_cols:
            series = df[column].dropna()
            if len(series) < 20:
                continue
                
            value_counts = series.value_counts()
            total_count = len(series)
            
            # Detect extremely rare categories (< 1% frequency)
            rare_categories = value_counts[value_counts / total_count < 0.01]
            
            if len(rare_categories) > 0:
                rare_indices = []
                for rare_value in rare_categories.index:
                    rare_indices.extend(series[series == rare_value].index.tolist())
                
                severity = 'low' if len(rare_categories) <= 2 else 'medium'
                confidence = 0.6  # Lower confidence for pattern anomalies
                
                self.detection_results.append(AnomalyResult(
                    anomaly_type=AnomalyType.PATTERN,
                    severity=severity,
                    confidence=confidence,
                    affected_features=[column],
                    anomaly_indices=rare_indices,
                    description=f"Rare categorical patterns in {column}",
                    metadata={
                        'method': 'rare_categories',
                        'rare_categories': rare_categories.to_dict(),
                        'total_categories': len(value_counts),
                        'rare_threshold': 0.01
                    }
                ))
        
        # Text pattern anomalies
        text_cols = [col for col in categorical_cols if self._is_text_column(df[col])]
        
        for column in text_cols:
            series = df[column].dropna().astype(str)
            if len(series) < 20:
                continue
                
            # Detect unusual text lengths
            text_lengths = series.str.len()
            length_q1, length_q3 = text_lengths.quantile(0.25), text_lengths.quantile(0.75)
            length_iqr = length_q3 - length_q1
            
            if length_iqr > 0:
                length_lower = length_q1 - 2 * length_iqr
                length_upper = length_q3 + 2 * length_iqr
                
                unusual_length_mask = (text_lengths < length_lower) | (text_lengths > length_upper)
                
                if unusual_length_mask.any():
                    unusual_indices = text_lengths[unusual_length_mask].index.tolist()
                    
                    self.detection_results.append(AnomalyResult(
                        anomaly_type=AnomalyType.PATTERN,
                        severity='low',
                        confidence=0.5,
                        affected_features=[column],
                        anomaly_indices=unusual_indices,
                        description=f"Unusual text length patterns in {column}",
                        metadata={
                            'method': 'text_length_anomaly',
                            'normal_length_range': [float(length_lower), float(length_upper)],
                            'anomaly_count': int(unusual_length_mask.sum())
                        }
                    ))
    
    def _detect_multivariate_anomalies(self, df: pd.DataFrame):
        """Detect multivariate anomalies using ensemble methods."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return
            
        # Prepare data
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        
        if len(X) < 50:  # Need sufficient samples
            return
            
        # Standardize features
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        except Exception as e:
            logging.warning(f"Failed to scale features for multivariate detection: {e}")
            return
        
        # Isolation Forest
        try:
            iso_forest = IsolationForest(
                contamination=self.config['isolation_contamination'],
                random_state=42,
                n_estimators=100
            )
            
            iso_predictions = iso_forest.fit_predict(X_scaled)
            iso_scores = iso_forest.decision_function(X_scaled)
            
            anomaly_mask = iso_predictions == -1
            
            if anomaly_mask.any():
                anomaly_indices = X.index[anomaly_mask].tolist()
                max_anomaly_score = abs(iso_scores[anomaly_mask].min())  # More negative = more anomalous
                severity = self._calculate_severity(max_anomaly_score * 2)  # Scale for severity
                confidence = min(0.85, max_anomaly_score)
                
                self.detection_results.append(AnomalyResult(
                    anomaly_type=AnomalyType.MULTIVARIATE,
                    severity=severity,
                    confidence=confidence,
                    affected_features=numeric_cols.tolist(),
                    anomaly_indices=anomaly_indices,
                    description="Multivariate anomalies detected using Isolation Forest",
                    metadata={
                        'method': 'isolation_forest',
                        'contamination': self.config['isolation_contamination'],
                        'max_anomaly_score': float(max_anomaly_score),
                        'anomaly_count': int(anomaly_mask.sum())
                    }
                ))
        except Exception as e:
            logging.warning(f"Isolation Forest failed: {e}")
        
        # Local Outlier Factor
        try:
            lof = LocalOutlierFactor(
                n_neighbors=min(self.config['lof_n_neighbors'], len(X) - 1),
                contamination=self.config['isolation_contamination']
            )
            
            lof_predictions = lof.fit_predict(X_scaled)
            lof_scores = lof.negative_outlier_factor_
            
            lof_anomaly_mask = lof_predictions == -1
            
            if lof_anomaly_mask.any():
                lof_anomaly_indices = X.index[lof_anomaly_mask].tolist()
                max_lof_score = abs(lof_scores[lof_anomaly_mask].min())
                severity = self._calculate_severity(max_lof_score)
                confidence = min(0.8, max_lof_score / 2)
                
                self.detection_results.append(AnomalyResult(
                    anomaly_type=AnomalyType.MULTIVARIATE,
                    severity=severity,
                    confidence=confidence,
                    affected_features=numeric_cols.tolist(),
                    anomaly_indices=lof_anomaly_indices,
                    description="Density-based anomalies detected using Local Outlier Factor",
                    metadata={
                        'method': 'local_outlier_factor',
                        'n_neighbors': min(self.config['lof_n_neighbors'], len(X) - 1),
                        'max_lof_score': float(max_lof_score),
                        'anomaly_count': int(lof_anomaly_mask.sum())
                    }
                ))
        except Exception as e:
            logging.warning(f"LOF failed: {e}")
    
    def _is_text_column(self, series: pd.Series) -> bool:
        """Check if column contains meaningful text data."""
        if series.dtype not in ['object', 'string']:
            return False
            
        sample_values = series.dropna().head(100)
        if len(sample_values) == 0:
            return False
            
        # Check if values look like text (average length > 10, contains spaces)
        avg_length = sample_values.astype(str).str.len().mean()
        has_spaces = sample_values.astype(str).str.contains(' ').mean()
        
        return avg_length > 10 and has_spaces > 0.3
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate severity based on anomaly score."""
        thresholds = self.config['severity_thresholds']
        
        normalized_score = min(1.0, abs(score) / 5.0)  # Normalize to 0-1
        
        if normalized_score >= thresholds['critical']:
            return 'critical'
        elif normalized_score >= thresholds['high']:
            return 'high'
        elif normalized_score >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_severity_from_ks(self, ks_statistic: float) -> str:
        """Calculate severity based on KS statistic."""
        if ks_statistic >= 0.5:
            return 'critical'
        elif ks_statistic >= 0.3:
            return 'high'
        elif ks_statistic >= 0.15:
            return 'medium'
        else:
            return 'low'
    
    def _filter_and_merge_results(self) -> List[AnomalyResult]:
        """Filter results by confidence and merge overlapping detections."""
        # Filter by confidence threshold
        filtered = [
            result for result in self.detection_results 
            if result.confidence >= self.config['confidence_threshold']
        ]
        
        # Sort by confidence (highest first)
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Simple merging: remove results with >50% overlapping indices
        merged_results = []
        
        for result in filtered:
            is_duplicate = False
            result_indices_set = set(result.anomaly_indices)
            
            for existing in merged_results:
                existing_indices_set = set(existing.anomaly_indices)
                overlap_ratio = len(result_indices_set & existing_indices_set) / len(result_indices_set | existing_indices_set)
                
                if overlap_ratio > 0.5:  # Significant overlap
                    # Keep the one with higher confidence
                    if result.confidence <= existing.confidence:
                        is_duplicate = True
                        break
                    else:
                        # Replace existing with current (higher confidence)
                        merged_results.remove(existing)
                        break
            
            if not is_duplicate:
                merged_results.append(result)
        
        return merged_results
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary statistics of detected anomalies."""
        if not self.detection_results:
            return {'total_anomalies': 0, 'by_type': {}, 'by_severity': {}}
        
        by_type = {}
        by_severity = {}
        total_affected_samples = set()
        
        for result in self.detection_results:
            # Count by type
            type_name = result.anomaly_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
            
            # Count by severity
            by_severity[result.severity] = by_severity.get(result.severity, 0) + 1
            
            # Track affected samples
            total_affected_samples.update(result.anomaly_indices)
        
        return {
            'total_anomalies': len(self.detection_results),
            'total_affected_samples': len(total_affected_samples),
            'by_type': by_type,
            'by_severity': by_severity,
            'avg_confidence': np.mean([r.confidence for r in self.detection_results]),
            'max_confidence': max([r.confidence for r in self.detection_results]) if self.detection_results else 0
        }

def detect_dataset_anomalies(df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None, 
                           config: Optional[Dict[str, Any]] = None) -> Tuple[List[AnomalyResult], Dict[str, Any]]:
    """
    Convenience function for anomaly detection on a dataset.
    
    Args:
        df: DataFrame to analyze
        reference_df: Optional reference DataFrame for comparison
        config: Optional configuration overrides
    
    Returns:
        Tuple of (anomaly_results, summary_stats)
    """
    detector = MultiLayerAnomalyDetector(config)
    results = detector.detect_anomalies(df, reference_df)
    summary = detector.get_anomaly_summary()
    
    return results, summary