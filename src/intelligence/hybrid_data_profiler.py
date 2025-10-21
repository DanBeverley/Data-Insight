"""Hybrid Data Profiling System for Dataset Awareness - DataInsight AI

Integrates all existing intelligence modules to create comprehensive dataset profiles
for AI agent awareness including anomalies, quality assessment, semantic understanding, etc.
"""

import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from intelligence.data_profiler import IntelligentDataProfiler
from intelligence.domain_detector import DomainDetector
from intelligence.relationship_discovery import RelationshipDiscovery
from data_quality.anomaly_detector import MultiLayerAnomalyDetector
from data_quality.quality_assessor import ContextAwareQualityAssessor
from data_quality.drift_monitor import ComprehensiveDriftMonitor
from data_quality.missing_value_intelligence import AdvancedMissingValueIntelligence

try:
    from security.privacy_engine import PrivacyEngine
    from security.compliance_manager import ComplianceManager
except ImportError:
    PrivacyEngine = None
    ComplianceManager = None


class ProfilerStatus(Enum):
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class DatasetInsights:
    """High-level insights about the dataset"""

    total_records: int
    total_features: int
    missing_data_percentage: float
    data_quality_score: float
    anomaly_count: int
    detected_domains: List[str]
    key_columns: List[str]
    temporal_columns: List[str]
    text_columns: List[str]
    high_cardinality_columns: List[str]
    potential_target_columns: List[str]
    data_freshness: str
    overall_health: str


@dataclass
class DataProfileSummary:
    """Comprehensive dataset profile for AI agent awareness"""

    dataset_insights: DatasetInsights
    column_profiles: Dict[str, Any]
    semantic_analysis: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    anomaly_detection: Dict[str, Any]
    relationship_analysis: Dict[str, Any]
    recommendations: List[str]
    ai_agent_context: Dict[str, Any]
    profile_metadata: Dict[str, Any]


class HybridDataProfiler:
    """Comprehensive data profiling system that integrates all intelligence modules"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # Initialize all intelligence modules
        self.intelligent_profiler = IntelligentDataProfiler()
        self.domain_detector = DomainDetector()
        self.relationship_discovery = RelationshipDiscovery()
        self.anomaly_detector = MultiLayerAnomalyDetector()
        self.quality_assessor = ContextAwareQualityAssessor()
        self.drift_monitor = ComprehensiveDriftMonitor()
        self.missing_value_intelligence = AdvancedMissingValueIntelligence()

        # Profile history for comparison
        self.profile_history: List[DataProfileSummary] = []

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for hybrid profiler"""
        return {
            "enable_anomaly_detection": True,
            "enable_quality_assessment": True,
            "enable_relationship_discovery": True,
            "enable_drift_monitoring": True,
            "enable_missing_value_analysis": True,
            "min_samples_for_full_profile": 10,
            "max_profile_time_seconds": 300,  # 5 minutes max
            "agent_context_level": "detailed",  # 'basic', 'detailed', 'comprehensive'
            "recommendation_limit": 10,
            "save_profile_history": True,
        }

    def generate_comprehensive_profile(
        self, df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None, context: Optional[Dict[str, Any]] = None
    ) -> DataProfileSummary:
        """
        Generate comprehensive dataset profile integrating all intelligence modules

        Args:
            df: DataFrame to profile
            reference_df: Optional reference DataFrame for drift/comparison analysis
            context: Optional context information (domain, purpose, etc.)

        Returns:
            Complete dataset profile with AI agent context
        """
        start_time = datetime.now()
        context = context or {}

        logging.info(
            f"Starting comprehensive profiling of dataset with {len(df)} samples and {len(df.columns)} features"
        )

        if len(df) < self.config["min_samples_for_full_profile"]:
            logging.warning(
                f"Dataset has fewer than {self.config['min_samples_for_full_profile']} samples. Profile may be limited."
            )

        try:
            # 1. Basic Intelligent Profiling
            logging.info("Phase 1: Intelligent data profiling...")
            semantic_profile = self.intelligent_profiler.profile_dataset(df)

            # 2. Data Quality Assessment
            quality_report = None
            if self.config["enable_quality_assessment"]:
                logging.info("Phase 2: Data quality assessment...")
                quality_report = self.quality_assessor.assess_quality(df, reference_df, context)

            # 3. Anomaly Detection
            anomaly_results = None
            anomaly_summary = {}
            if self.config["enable_anomaly_detection"]:
                logging.info("Phase 3: Multi-layer anomaly detection...")
                anomaly_results = self.anomaly_detector.detect_anomalies(df, reference_df)
                anomaly_summary = self.anomaly_detector.get_anomaly_summary()

            # 4. Missing Value Intelligence
            missing_analysis = None
            if self.config["enable_missing_value_analysis"]:
                logging.info("Phase 4: Missing value intelligence analysis...")
                missing_analysis = self.missing_value_intelligence.analyze_missing_patterns(df)

            # 5. PII Detection & Privacy Assessment
            pii_detection = None
            if self.config.get("enable_privacy_analysis", True):
                logging.info("Phase 5: PII detection and privacy assessment...")
                try:
                    privacy_engine = PrivacyEngine()
                    pii_detection = privacy_engine.assess_privacy_risk(df)
                except Exception as e:
                    logging.warning(f"PII detection failed: {e}")

            # 6. Drift Analysis (if reference provided)
            drift_results = []
            if self.config["enable_drift_monitoring"] and reference_df is not None:
                logging.info("Phase 6: Data drift monitoring...")
                try:
                    self.drift_monitor.fit_reference(reference_df)
                    drift_results = self.drift_monitor.detect_drift(df)
                except Exception as e:
                    logging.warning(f"Drift analysis failed: {e}")

            # 6. Generate Dataset Insights
            dataset_insights = self._generate_dataset_insights(
                df, semantic_profile, quality_report, anomaly_summary, context
            )

            # 7. Create AI Agent Context
            ai_agent_context = self._create_agent_context(
                df, semantic_profile, quality_report, anomaly_results, missing_analysis, context
            )

            # 8. Generate Comprehensive Recommendations
            recommendations = self._generate_comprehensive_recommendations(
                semantic_profile, quality_report, anomaly_results, missing_analysis, context
            )

            # 9. Compile Complete Profile
            profile_summary = DataProfileSummary(
                dataset_insights=dataset_insights,
                column_profiles=semantic_profile["column_profiles"],
                semantic_analysis={
                    "domain_analysis": semantic_profile["domain_analysis"],
                    "relationship_analysis": semantic_profile["relationship_analysis"],
                },
                quality_assessment={
                    "overall_score": quality_report.overall_score if quality_report else None,
                    "dimension_scores": (
                        {dim.value: asdict(score) for dim, score in quality_report.dimension_scores.items()}
                        if quality_report
                        else {}
                    ),
                    "critical_issues": quality_report.critical_issues if quality_report else [],
                },
                anomaly_detection={
                    "summary": anomaly_summary,
                    "anomalies": [asdict(anomaly) for anomaly in anomaly_results] if anomaly_results else [],
                },
                relationship_analysis={
                    "missing_patterns": missing_analysis.pattern_summary if missing_analysis else {},
                    "drift_results": [asdict(drift) for drift in drift_results] if drift_results else [],
                },
                recommendations=recommendations,
                ai_agent_context=ai_agent_context,
                profile_metadata={
                    "profiling_timestamp": start_time.isoformat(),
                    "profiling_duration": (datetime.now() - start_time).total_seconds(),
                    "config_used": self.config,
                    "modules_enabled": {
                        "semantic_profiling": True,
                        "quality_assessment": self.config["enable_quality_assessment"],
                        "anomaly_detection": self.config["enable_anomaly_detection"],
                        "missing_analysis": self.config["enable_missing_value_analysis"],
                        "drift_monitoring": self.config["enable_drift_monitoring"] and reference_df is not None,
                        "privacy_analysis": self.config.get("enable_privacy_analysis", True),
                    },
                    "pii_detection": pii_detection,
                },
            )

            # Store in history
            if self.config["save_profile_history"]:
                self.profile_history.append(profile_summary)

            duration = (datetime.now() - start_time).total_seconds()
            logging.info(f"Comprehensive profiling completed in {duration:.2f} seconds")

            return profile_summary

        except Exception as e:
            logging.error(f"Comprehensive profiling failed: {e}", exc_info=True)
            # Return minimal profile on error
            return self._create_error_profile(df, str(e), start_time)

    def _generate_dataset_insights(
        self,
        df: pd.DataFrame,
        semantic_profile: Dict[str, Any],
        quality_report,
        anomaly_summary: Dict[str, Any],
        context: Dict[str, Any],
    ) -> DatasetInsights:
        """Generate high-level dataset insights"""

        column_profiles = semantic_profile["column_profiles"]

        # Identify different column types
        key_columns = []
        temporal_columns = []
        text_columns = []
        high_cardinality_columns = []
        potential_targets = []

        for col_name, profile in column_profiles.items():
            semantic_type = (
                profile.semantic_type.value if hasattr(profile.semantic_type, "value") else str(profile.semantic_type)
            )

            if "key" in semantic_type.lower():
                key_columns.append(col_name)
            elif "datetime" in semantic_type.lower() or "temporal" in semantic_type.lower():
                temporal_columns.append(col_name)
            elif "text" in semantic_type.lower():
                text_columns.append(col_name)

            # High cardinality check
            if hasattr(profile, "evidence") and isinstance(profile.evidence, dict):
                cardinality = profile.evidence.get("cardinality", 0)
                if cardinality > 0.7 and semantic_type not in ["primary_key", "foreign_key"]:
                    high_cardinality_columns.append(col_name)

            # Potential target detection (numeric columns with reasonable distribution)
            if pd.api.types.is_numeric_dtype(df[col_name]) and col_name not in key_columns:
                if df[col_name].nunique() > 2 and df[col_name].nunique() < len(df) * 0.8:
                    potential_targets.append(col_name)

        # Calculate missing data percentage
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

        # Determine data freshness
        data_freshness = self._assess_data_freshness(df, temporal_columns)

        # Determine overall health
        quality_score = quality_report.overall_score if quality_report else 75
        anomaly_count = anomaly_summary.get("total_anomalies", 0)

        if quality_score >= 80 and anomaly_count == 0:
            overall_health = "Excellent"
        elif quality_score >= 60 and anomaly_count <= 5:
            overall_health = "Good"
        elif quality_score >= 40:
            overall_health = "Fair"
        else:
            overall_health = "Poor"

        # Detect domains
        detected_domains = []
        if "domain_analysis" in semantic_profile:
            domain_matches = semantic_profile["domain_analysis"].get("detected_domains", [])
            detected_domains = [match.get("domain", "") for match in domain_matches[:3]]

        return DatasetInsights(
            total_records=len(df),
            total_features=len(df.columns),
            missing_data_percentage=round(missing_percentage, 2),
            data_quality_score=round(quality_score, 2),
            anomaly_count=anomaly_count,
            detected_domains=detected_domains,
            key_columns=key_columns,
            temporal_columns=temporal_columns,
            text_columns=text_columns,
            high_cardinality_columns=high_cardinality_columns,
            potential_target_columns=potential_targets,
            data_freshness=data_freshness,
            overall_health=overall_health,
        )

    def _create_agent_context(
        self,
        df: pd.DataFrame,
        semantic_profile: Dict[str, Any],
        quality_report,
        anomaly_results,
        missing_analysis,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create AI agent context for dataset awareness"""

        # Basic column information
        column_info = {}
        for col_name in df.columns:
            col_data = df[col_name]

            basic_info = {
                "name": col_name,
                "dtype": str(col_data.dtype),
                "null_count": int(col_data.isnull().sum()),
                "unique_count": int(col_data.nunique()),
                "sample_values": col_data.dropna().head(3).tolist(),
            }

            # Add semantic information if available
            if col_name in semantic_profile["column_profiles"]:
                profile = semantic_profile["column_profiles"][col_name]
                basic_info["semantic_type"] = (
                    profile.semantic_type.value
                    if hasattr(profile.semantic_type, "value")
                    else str(profile.semantic_type)
                )
                basic_info["recommendations"] = profile.recommendations[:2]  # Limit to top 2

            # Add numeric statistics for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                basic_info.update(
                    {
                        "min": float(col_data.min()) if col_data.notna().any() else None,
                        "max": float(col_data.max()) if col_data.notna().any() else None,
                        "mean": float(col_data.mean()) if col_data.notna().any() else None,
                        "std": float(col_data.std()) if col_data.notna().any() else None,
                    }
                )

            column_info[col_name] = basic_info

        # Key insights for agent
        agent_context = {
            "dataset_overview": {
                "shape": df.shape,
                "column_names": list(df.columns),
                "dtypes_summary": df.dtypes.value_counts().to_dict(),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            },
            "column_details": column_info,
            "data_quality": {
                "overall_score": quality_report.overall_score if quality_report else None,
                "missing_data_summary": {
                    "total_missing": int(df.isnull().sum().sum()),
                    "columns_with_missing": df.columns[df.isnull().any()].tolist(),
                    "missing_percentages": {
                        col: round((df[col].isnull().sum() / len(df)) * 100, 2)
                        for col in df.columns
                        if df[col].isnull().any()
                    },
                },
            },
            "anomalies": {
                "total_count": (
                    len(anomaly_results[0].anomaly_indices) if anomaly_results and len(anomaly_results) > 0 else 0
                ),
                "affected_columns": (
                    list(set([col for result in anomaly_results for col in result.affected_features]))
                    if anomaly_results
                    else []
                ),
            },
            "visualization_suggestions": self._generate_visualization_suggestions(df, semantic_profile),
            "analysis_recommendations": self._generate_analysis_recommendations(df, semantic_profile, quality_report),
            "code_generation_hints": {
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
                "datetime_columns": df.select_dtypes(include=["datetime64"]).columns.tolist(),
                "suggested_target_columns": [
                    col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 2
                ][:3],
            },
        }

        return agent_context

    def _generate_visualization_suggestions(
        self, df: pd.DataFrame, semantic_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate visualization suggestions based on data profile"""
        suggestions = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Correlation heatmap for numeric data
        if len(numeric_cols) >= 2:
            suggestions.append(
                {
                    "type": "heatmap",
                    "description": "Correlation matrix of numeric features",
                    "columns": numeric_cols,
                    "code_hint": 'sns.heatmap(df[numeric_columns].corr(), annot=True, cmap="coolwarm")',
                }
            )

        # Distribution plots
        if len(numeric_cols) >= 1:
            suggestions.append(
                {
                    "type": "histogram",
                    "description": "Distribution analysis of numeric features",
                    "columns": numeric_cols[:3],  # Limit to first 3
                    "code_hint": "df[numeric_columns].hist(figsize=(12, 8))",
                }
            )

        # Bar charts for categorical data
        if len(categorical_cols) >= 1:
            suggestions.append(
                {
                    "type": "bar_chart",
                    "description": "Value counts for categorical features",
                    "columns": categorical_cols[:2],  # Limit to first 2
                    "code_hint": 'df[categorical_column].value_counts().plot(kind="bar")',
                }
            )

        # Scatter plots for relationships
        if len(numeric_cols) >= 2:
            suggestions.append(
                {
                    "type": "scatter_plot",
                    "description": "Relationship between numeric features",
                    "columns": numeric_cols[:2],
                    "code_hint": f'plt.scatter(df["{numeric_cols[0]}"], df["{numeric_cols[1]}"])',
                }
            )

        return suggestions

    def _generate_analysis_recommendations(
        self, df: pd.DataFrame, semantic_profile: Dict[str, Any], quality_report
    ) -> List[str]:
        """Generate analysis recommendations for the agent"""
        recommendations = []

        # Data exploration recommendations
        recommendations.append("Start with df.info() and df.describe() for basic data understanding")

        # Missing data recommendations
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            high_missing_cols = missing_data[missing_data > len(df) * 0.1].index.tolist()
            if high_missing_cols:
                recommendations.append(f"Address missing data in columns: {', '.join(high_missing_cols[:3])}")

        # Quality-based recommendations
        if quality_report and quality_report.overall_score < 70:
            recommendations.append("Data quality issues detected - consider data cleaning before analysis")

        # Column-specific recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            recommendations.append("Explore correlations between numeric features for feature selection")

        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) >= 1:
            recommendations.append("Analyze categorical feature distributions and consider encoding strategies")

        return recommendations[:5]  # Limit to top 5

    def _assess_data_freshness(self, df: pd.DataFrame, temporal_columns: List[str]) -> str:
        """Assess data freshness based on temporal columns"""
        if not temporal_columns:
            return "Unknown (no temporal columns detected)"

        try:
            # Check the most recent temporal column
            for col in temporal_columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    latest_date = df[col].max()
                    days_old = (datetime.now() - latest_date.to_pydatetime()).days

                    if days_old <= 7:
                        return "Very Fresh (< 1 week old)"
                    elif days_old <= 30:
                        return "Fresh (< 1 month old)"
                    elif days_old <= 365:
                        return "Moderate (< 1 year old)"
                    else:
                        return "Stale (> 1 year old)"
        except Exception as e:
            logging.warning(f"Failed to assess data freshness: {e}")

        return "Unknown"

    def _generate_comprehensive_recommendations(
        self,
        semantic_profile: Dict[str, Any],
        quality_report,
        anomaly_results,
        missing_analysis,
        context: Dict[str, Any],
    ) -> List[str]:
        """Generate comprehensive recommendations from all analyses"""
        recommendations = []

        # Semantic recommendations
        if "overall_recommendations" in semantic_profile:
            recommendations.extend(semantic_profile["overall_recommendations"][:3])

        # Quality recommendations
        if quality_report and quality_report.recommendations:
            recommendations.extend(quality_report.recommendations[:3])

        # Missing value recommendations
        if missing_analysis and hasattr(missing_analysis, "recommendations"):
            recommendations.extend(missing_analysis.recommendations[:2])

        # Anomaly-based recommendations
        if anomaly_results:
            high_severity_anomalies = [a for a in anomaly_results if a.severity in ["high", "critical"]]
            if high_severity_anomalies:
                recommendations.append("Critical anomalies detected - investigate before proceeding with analysis")

        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen and len(rec.strip()) > 0:
                unique_recommendations.append(rec)
                seen.add(rec)

        return unique_recommendations[: self.config["recommendation_limit"]]

    def _create_error_profile(self, df: pd.DataFrame, error_message: str, start_time: datetime) -> DataProfileSummary:
        """Create minimal profile when full profiling fails"""
        basic_insights = DatasetInsights(
            total_records=len(df),
            total_features=len(df.columns),
            missing_data_percentage=round((df.isnull().sum().sum() / df.size) * 100, 2),
            data_quality_score=50.0,  # Neutral score
            anomaly_count=0,
            detected_domains=[],
            key_columns=[],
            temporal_columns=[],
            text_columns=[],
            high_cardinality_columns=[],
            potential_target_columns=[],
            data_freshness="Unknown",
            overall_health="Unknown",
        )

        basic_context = {
            "dataset_overview": {
                "shape": df.shape,
                "column_names": list(df.columns),
                "dtypes_summary": df.dtypes.value_counts().to_dict(),
            },
            "error": error_message,
        }

        return DataProfileSummary(
            dataset_insights=basic_insights,
            column_profiles={},
            semantic_analysis={},
            quality_assessment={},
            anomaly_detection={},
            relationship_analysis={},
            recommendations=["Error occurred during profiling - using basic column names and types only"],
            ai_agent_context=basic_context,
            profile_metadata={
                "profiling_timestamp": start_time.isoformat(),
                "profiling_duration": (datetime.now() - start_time).total_seconds(),
                "status": "error",
                "error_message": error_message,
            },
        )

    def export_agent_context(self, profile: DataProfileSummary, format: str = "json") -> str:
        """Export agent context in specified format for LLM consumption"""
        agent_context = profile.ai_agent_context

        if format == "json":
            return json.dumps(agent_context, indent=2, default=str)
        elif format == "summary":
            # Create human-readable summary
            overview = agent_context["dataset_overview"]
            summary = f"""Dataset Overview:
- Shape: {overview['shape'][0]} rows × {overview['shape'][1]} columns  
- Columns: {', '.join(overview['column_names'][:5])}{'...' if len(overview['column_names']) > 5 else ''}
- Data Quality Score: {agent_context['data_quality']['overall_score']}/100

Key Insights:
- Missing Data: {agent_context['data_quality']['missing_data_summary']['total_missing']} cells
- Anomalies: {agent_context['anomalies']['total_count']} detected
- Numeric Columns: {len(agent_context['code_generation_hints']['numeric_columns'])}
- Categorical Columns: {len(agent_context['code_generation_hints']['categorical_columns'])}

Recommendations:
{chr(10).join(f'- {rec}' for rec in profile.recommendations[:3])}
"""
            return summary
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_column_context_for_agent(self, profile: DataProfileSummary, detailed: bool = False) -> Dict[str, Any]:
        """Get column-specific context optimized for AI agent code generation"""
        context = {}

        for col_name, col_info in profile.ai_agent_context["column_details"].items():
            context[col_name] = {
                "dtype": col_info["dtype"],
                "null_count": col_info["null_count"],
                "unique_count": col_info["unique_count"],
                "semantic_type": col_info.get("semantic_type", "unknown"),
            }

            if detailed:
                context[col_name].update(
                    {"sample_values": col_info["sample_values"], "recommendations": col_info.get("recommendations", [])}
                )

                # Add statistical info for numeric columns
                if "min" in col_info:
                    context[col_name].update(
                        {
                            "range": [col_info.get("min"), col_info.get("max")],
                            "mean": col_info.get("mean"),
                            "std": col_info.get("std"),
                        }
                    )

        return context


def generate_dataset_profile_for_agent(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    context: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> DataProfileSummary:
    """
    Convenience function to generate comprehensive dataset profile for AI agent awareness

    Args:
        df: DataFrame to profile
        reference_df: Optional reference DataFrame for comparison
        context: Optional context information
        config: Optional configuration overrides

    Returns:
        Complete dataset profile with AI agent context
    """
    profiler = HybridDataProfiler(config)
    return profiler.generate_comprehensive_profile(df, reference_df, context)
