import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")


@dataclass
class FeatureEngineeringRecommendation:
    feature_type: str
    priority: str
    description: str
    implementation: str
    expected_benefit: str
    computational_cost: str


class AdvancedFeatureIntelligence:
    """Intelligent feature engineering based on data understanding"""

    def __init__(self):
        self.feature_generators = {
            "interaction": self._generate_interaction_features,
            "polynomial": self._generate_polynomial_features,
            "aggregation": self._generate_aggregation_features,
            "temporal": self._generate_temporal_features,
            "text": self._generate_text_features,
            "categorical": self._generate_categorical_features,
            "domain_specific": self._generate_domain_specific_features,
        }

        self.scalers = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler()}

    def analyze_feature_engineering_opportunities(
        self, df: pd.DataFrame, intelligence_profile: Dict, target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze and recommend feature engineering strategies"""

        column_profiles = intelligence_profile.get("column_profiles", {})
        domain_analysis = intelligence_profile.get("domain_analysis", {})
        relationship_analysis = intelligence_profile.get("relationship_analysis", {})

        recommendations = []

        # Analyze each type of feature engineering opportunity
        for feature_type, generator in self.feature_generators.items():
            try:
                feature_recs = generator(df, column_profiles, domain_analysis, relationship_analysis, target_column)
                recommendations.extend(feature_recs)
            except Exception as e:
                continue

        # Add basic recommendations if none were generated
        if not recommendations:
            recommendations.extend(self._generate_basic_recommendations(df, column_profiles, target_column))

        # Prioritize recommendations
        prioritized_recs = self._prioritize_recommendations(recommendations, df, target_column)

        # Generate feature selection strategy
        selection_strategy = self._recommend_feature_selection(df, column_profiles, target_column)

        # Generate scaling recommendations
        scaling_strategy = self._recommend_scaling_strategy(df, column_profiles)

        return {
            "feature_engineering_recommendations": prioritized_recs,
            "feature_selection_strategy": selection_strategy,
            "scaling_strategy": scaling_strategy,
            "implementation_pipeline": self._generate_implementation_pipeline(prioritized_recs),
        }

    def _generate_interaction_features(
        self,
        df: pd.DataFrame,
        column_profiles: Dict,
        domain_analysis: Dict,
        relationship_analysis: Dict,
        target_column: Optional[str],
    ) -> List[FeatureEngineeringRecommendation]:
        """Generate interaction feature recommendations"""
        recommendations = []

        # Find correlated numerical features for interactions
        numeric_cols = [
            col
            for col, profile in column_profiles.items()
            if profile.semantic_type.value in ["currency", "percentage", "count", "ratio"]
        ]

        if len(numeric_cols) >= 2:
            # Recommend multiplicative interactions for financial data
            if any("finance" in domain.get("domain", "") for domain in domain_analysis.get("detected_domains", [])):
                recommendations.append(
                    FeatureEngineeringRecommendation(
                        feature_type="interaction",
                        priority="high",
                        description="Create multiplicative features for financial ratios",
                        implementation=f"Multiply pairs from: {numeric_cols[:3]}",
                        expected_benefit="Capture non-linear financial relationships",
                        computational_cost="low",
                    )
                )

        # Categorical-numerical interactions
        categorical_cols = [
            col for col, profile in column_profiles.items() if "categorical" in profile.semantic_type.value
        ]

        if categorical_cols and numeric_cols:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="interaction",
                    priority="medium",
                    description="Create categorical-numerical interaction features",
                    implementation=f"Group statistics of {numeric_cols[0]} by {categorical_cols[0]}",
                    expected_benefit="Capture group-specific patterns",
                    computational_cost="medium",
                )
            )

        return recommendations

    def _generate_polynomial_features(
        self,
        df: pd.DataFrame,
        column_profiles: Dict,
        domain_analysis: Dict,
        relationship_analysis: Dict,
        target_column: Optional[str],
    ) -> List[FeatureEngineeringRecommendation]:
        """Generate polynomial feature recommendations"""
        recommendations = []

        # Find features with non-linear relationships
        relationships = relationship_analysis.get("relationships", [])
        monotonic_relationships = [
            rel for rel in relationships if rel.get("relationship_type") == "monotonic_correlation"
        ]

        if monotonic_relationships:
            source_cols = [rel["column1"] for rel in monotonic_relationships[:2]]
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="polynomial",
                    priority="medium",
                    description="Add polynomial features for non-linear monotonic relationships",
                    implementation=f"Square and cube transformations for: {source_cols}",
                    expected_benefit="Capture non-linear patterns in monotonic relationships",
                    computational_cost="low",
                )
            )

        # Recommend polynomial features for specific domains
        detected_domains = [d.get("domain") for d in domain_analysis.get("detected_domains", [])]
        if "iot_sensor" in detected_domains or "logistics" in detected_domains:
            numeric_cols = [
                col for col, profile in column_profiles.items() if profile.semantic_type.value in ["count", "ratio"]
            ]
            if numeric_cols:
                recommendations.append(
                    FeatureEngineeringRecommendation(
                        feature_type="polynomial",
                        priority="high",
                        description="Polynomial features for sensor/physics relationships",
                        implementation=f"Quadratic transformations for: {numeric_cols[:2]}",
                        expected_benefit="Model physical relationships and sensor non-linearities",
                        computational_cost="low",
                    )
                )

        return recommendations

    def _generate_aggregation_features(
        self,
        df: pd.DataFrame,
        column_profiles: Dict,
        domain_analysis: Dict,
        relationship_analysis: Dict,
        target_column: Optional[str],
    ) -> List[FeatureEngineeringRecommendation]:
        """Generate aggregation feature recommendations"""
        recommendations = []

        # Find hierarchical relationships for aggregation
        relationships = relationship_analysis.get("relationships", [])
        hierarchical_rels = [rel for rel in relationships if rel.get("relationship_type") == "hierarchical"]

        for rel in hierarchical_rels:
            parent_col = rel["column1"]
            child_col = rel["column2"]

            # Find numeric columns to aggregate
            numeric_cols = [
                col
                for col, profile in column_profiles.items()
                if profile.semantic_type.value in ["currency", "count", "ratio"]
            ]

            if numeric_cols:
                recommendations.append(
                    FeatureEngineeringRecommendation(
                        feature_type="aggregation",
                        priority="high",
                        description=f"Hierarchical aggregation: {parent_col} -> {child_col}",
                        implementation=f"Group {numeric_cols[0]} by {parent_col}: mean, sum, std",
                        expected_benefit="Capture hierarchy-level patterns and variability",
                        computational_cost="medium",
                    )
                )

        # Key-based aggregations
        key_relationships = [rel for rel in relationships if "key" in rel.get("relationship_type", "")]

        for rel in key_relationships:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="aggregation",
                    priority="medium",
                    description=f"Key-based aggregation features",
                    implementation=f'Aggregate by {rel["column1"]}: count, nunique',
                    expected_benefit="Entity-level summary statistics",
                    computational_cost="medium",
                )
            )

        return recommendations

    def _generate_temporal_features(
        self,
        df: pd.DataFrame,
        column_profiles: Dict,
        domain_analysis: Dict,
        relationship_analysis: Dict,
        target_column: Optional[str],
    ) -> List[FeatureEngineeringRecommendation]:
        """Generate temporal feature recommendations"""
        recommendations = []

        # Find temporal columns
        temporal_cols = [col for col, profile in column_profiles.items() if "datetime" in profile.semantic_type.value]

        if not temporal_cols:
            return recommendations

        # Domain-specific temporal features
        detected_domains = [d.get("domain") for d in domain_analysis.get("detected_domains", [])]

        if "ecommerce" in detected_domains:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="temporal",
                    priority="high",
                    description="E-commerce seasonal and cyclical patterns",
                    implementation=f"Extract from {temporal_cols[0]}: day_of_week, month, quarter, is_weekend, is_holiday",
                    expected_benefit="Capture shopping seasonality and behavioral patterns",
                    computational_cost="low",
                )
            )

        if "finance" in detected_domains:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="temporal",
                    priority="high",
                    description="Financial market temporal features",
                    implementation=f"Extract from {temporal_cols[0]}: day_of_month, quarter_end, month_end, business_day",
                    expected_benefit="Capture financial reporting cycles and market patterns",
                    computational_cost="low",
                )
            )

        # Temporal trend analysis
        temporal_relationships = [
            rel
            for rel in relationship_analysis.get("relationships", [])
            if rel.get("relationship_type") == "temporal_trend"
        ]

        if temporal_relationships:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="temporal",
                    priority="high",
                    description="Lag and rolling window features for trends",
                    implementation=f"Create lag features (1,3,7 periods) and rolling statistics (mean, std)",
                    expected_benefit="Capture temporal dependencies and trend patterns",
                    computational_cost="medium",
                )
            )

        return recommendations

    def _generate_text_features(
        self,
        df: pd.DataFrame,
        column_profiles: Dict,
        domain_analysis: Dict,
        relationship_analysis: Dict,
        target_column: Optional[str],
    ) -> List[FeatureEngineeringRecommendation]:
        """Generate text feature recommendations"""
        recommendations = []

        # Find text columns
        text_cols = [
            col
            for col, profile in column_profiles.items()
            if "text" in profile.semantic_type.value or profile.semantic_type.value == "email"
        ]

        if not text_cols:
            return recommendations

        detected_domains = [d.get("domain") for d in domain_analysis.get("detected_domains", [])]

        if "social_media" in detected_domains:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="text",
                    priority="high",
                    description="Social media text analysis features",
                    implementation=f"Extract from {text_cols[0]}: sentiment, hashtag_count, mention_count, text_length",
                    expected_benefit="Capture social engagement and sentiment patterns",
                    computational_cost="medium",
                )
            )

        if "marketing" in detected_domains:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="text",
                    priority="medium",
                    description="Marketing text content features",
                    implementation=f"TF-IDF vectorization and topic modeling for {text_cols[0]}",
                    expected_benefit="Extract marketing message themes and effectiveness indicators",
                    computational_cost="high",
                )
            )

        # Email domain extraction
        email_cols = [col for col, profile in column_profiles.items() if profile.semantic_type.value == "email"]

        if email_cols:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="text",
                    priority="medium",
                    description="Email domain categorization",
                    implementation=f"Extract domain from {email_cols[0]} and categorize (business/personal/other)",
                    expected_benefit="Segment users by email provider type",
                    computational_cost="low",
                )
            )

        return recommendations

    def _generate_categorical_features(
        self,
        df: pd.DataFrame,
        column_profiles: Dict,
        domain_analysis: Dict,
        relationship_analysis: Dict,
        target_column: Optional[str],
    ) -> List[FeatureEngineeringRecommendation]:
        """Generate categorical feature recommendations"""
        recommendations = []

        # High cardinality categorical features
        high_card_cats = [
            col
            for col, profile in column_profiles.items()
            if "categorical" in profile.semantic_type.value and profile.evidence.get("cardinality", 0) > 0.1
        ]

        if high_card_cats:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="categorical",
                    priority="medium",
                    description="Target encoding for high cardinality categoricals",
                    implementation=f"Target encode: {high_card_cats[:2]}",
                    expected_benefit="Reduce dimensionality while preserving predictive power",
                    computational_cost="medium",
                )
            )

        # Ordinal categorical features
        ordinal_cats = [
            col for col, profile in column_profiles.items() if profile.semantic_type.value == "categorical_ordinal"
        ]

        if ordinal_cats:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="categorical",
                    priority="high",
                    description="Ordinal encoding preserving order relationships",
                    implementation=f"Ordinal encode: {ordinal_cats}",
                    expected_benefit="Preserve ordinal relationships for better model performance",
                    computational_cost="low",
                )
            )

        # Binary categorical features
        binary_cats = [
            col for col, profile in column_profiles.items() if profile.semantic_type.value == "categorical_binary"
        ]

        if binary_cats:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="categorical",
                    priority="low",
                    description="Binary encoding for binary categorical features",
                    implementation=f"Binary encode: {binary_cats}",
                    expected_benefit="Memory-efficient encoding for binary categories",
                    computational_cost="low",
                )
            )

        return recommendations

    def _generate_domain_specific_features(
        self,
        df: pd.DataFrame,
        column_profiles: Dict,
        domain_analysis: Dict,
        relationship_analysis: Dict,
        target_column: Optional[str],
    ) -> List[FeatureEngineeringRecommendation]:
        """Generate domain-specific feature recommendations"""
        recommendations = []

        detected_domains = [d.get("domain") for d in domain_analysis.get("detected_domains", [])]

        if "ecommerce" in detected_domains:
            recommendations.extend(
                [
                    FeatureEngineeringRecommendation(
                        feature_type="domain_specific",
                        priority="high",
                        description="E-commerce customer behavior features",
                        implementation="Customer lifetime value, purchase frequency, basket size metrics",
                        expected_benefit="Capture customer value and behavior patterns",
                        computational_cost="medium",
                    ),
                    FeatureEngineeringRecommendation(
                        feature_type="domain_specific",
                        priority="medium",
                        description="Product performance features",
                        implementation="Product popularity rank, category performance, price positioning",
                        expected_benefit="Product-level insights for recommendations",
                        computational_cost="medium",
                    ),
                ]
            )

        if "finance" in detected_domains:
            recommendations.extend(
                [
                    FeatureEngineeringRecommendation(
                        feature_type="domain_specific",
                        priority="high",
                        description="Financial risk indicators",
                        implementation="Debt-to-income ratio, credit utilization, payment history features",
                        expected_benefit="Risk assessment and creditworthiness indicators",
                        computational_cost="low",
                    ),
                    FeatureEngineeringRecommendation(
                        feature_type="domain_specific",
                        priority="medium",
                        description="Market volatility features",
                        implementation="Price volatility, moving averages, technical indicators",
                        expected_benefit="Market trend and volatility insights",
                        computational_cost="medium",
                    ),
                ]
            )

        return recommendations

    def _prioritize_recommendations(
        self, recommendations: List[FeatureEngineeringRecommendation], df: pd.DataFrame, target_column: Optional[str]
    ) -> List[FeatureEngineeringRecommendation]:
        """Prioritize recommendations based on impact and feasibility"""

        # Priority scoring
        priority_scores = {"high": 3, "medium": 2, "low": 1}
        cost_scores = {"low": 3, "medium": 2, "high": 1}

        def calculate_score(rec):
            priority_score = priority_scores.get(rec.priority, 1)
            cost_score = cost_scores.get(rec.computational_cost, 1)

            # Boost score for supervised learning if target is available
            if target_column and rec.feature_type in ["interaction", "aggregation"]:
                priority_score += 1

            return priority_score + cost_score

        return sorted(recommendations, key=calculate_score, reverse=True)

    def _recommend_feature_selection(
        self, df: pd.DataFrame, column_profiles: Dict, target_column: Optional[str]
    ) -> Dict[str, Any]:
        """Recommend feature selection strategy"""

        total_features = len(df.columns)
        numeric_features = sum(
            1
            for profile in column_profiles.values()
            if profile.semantic_type.value in ["currency", "count", "ratio", "percentage"]
        )

        strategies = []

        # High-dimensional datasets
        if total_features > 50:
            strategies.append(
                {
                    "method": "variance_threshold",
                    "description": "Remove low-variance features",
                    "threshold": 0.01,
                    "priority": "high",
                }
            )

        # Many numeric features
        if numeric_features > 20:
            strategies.append(
                {
                    "method": "correlation_filter",
                    "description": "Remove highly correlated features",
                    "threshold": 0.95,
                    "priority": "medium",
                }
            )

        # Supervised learning
        if target_column:
            strategies.extend(
                [
                    {
                        "method": "univariate_selection",
                        "description": "Select K best features using statistical tests",
                        "k": min(20, total_features // 2),
                        "priority": "high",
                    },
                    {
                        "method": "recursive_feature_elimination",
                        "description": "RFE with Random Forest",
                        "n_features": min(15, total_features // 3),
                        "priority": "medium",
                    },
                ]
            )

        return {
            "recommended_strategies": strategies,
            "feature_budget": min(25, total_features // 2),
            "dimensionality_reduction": total_features > 30,
        }

    def _recommend_scaling_strategy(self, df: pd.DataFrame, column_profiles: Dict) -> Dict[str, Any]:
        """Recommend scaling strategy based on data characteristics"""

        numeric_profiles = {
            col: profile
            for col, profile in column_profiles.items()
            if profile.semantic_type.value in ["currency", "count", "ratio", "percentage"]
        }

        scaling_recommendations = {}

        for col, profile in numeric_profiles.items():
            numeric_analysis = profile.evidence.get("numeric_analysis", {})

            # Check for outliers (high kurtosis or skewness)
            kurtosis = abs(numeric_analysis.get("kurtosis", 0))
            skewness = abs(numeric_analysis.get("skewness", 0))

            if kurtosis > 3 or skewness > 2:
                scaling_recommendations[col] = "robust"
            elif profile.semantic_type.value == "percentage":
                scaling_recommendations[col] = "minmax"
            else:
                scaling_recommendations[col] = "standard"

        return {
            "column_scaling": scaling_recommendations,
            "default_scaler": (
                "robust"
                if len([s for s in scaling_recommendations.values() if s == "robust"]) > len(numeric_profiles) // 2
                else "standard"
            ),
        }

    def _generate_implementation_pipeline(
        self, recommendations: List[FeatureEngineeringRecommendation]
    ) -> List[Dict[str, str]]:
        """Generate step-by-step implementation pipeline"""

        pipeline = []

        # Group by computational cost and priority
        high_priority_low_cost = [r for r in recommendations if r.priority == "high" and r.computational_cost == "low"]
        medium_cost = [r for r in recommendations if r.computational_cost == "medium"]
        high_cost = [r for r in recommendations if r.computational_cost == "high"]

        # Phase 1: Quick wins
        if high_priority_low_cost:
            pipeline.append(
                {
                    "phase": "Phase 1 - Quick Wins",
                    "features": [r.feature_type for r in high_priority_low_cost],
                    "description": "Low-cost, high-impact features to implement first",
                    "estimated_time": "1-2 hours",
                }
            )

        # Phase 2: Medium complexity
        if medium_cost:
            pipeline.append(
                {
                    "phase": "Phase 2 - Enhanced Features",
                    "features": [r.feature_type for r in medium_cost],
                    "description": "More complex features requiring moderate computation",
                    "estimated_time": "4-6 hours",
                }
            )

        # Phase 3: Advanced features
        if high_cost:
            pipeline.append(
                {
                    "phase": "Phase 3 - Advanced Analytics",
                    "features": [r.feature_type for r in high_cost],
                    "description": "Computationally intensive advanced features",
                    "estimated_time": "8-12 hours",
                }
            )

        return pipeline

    def _generate_basic_recommendations(
        self, df: pd.DataFrame, column_profiles: Dict, target_column: Optional[str]
    ) -> List[FeatureEngineeringRecommendation]:
        """Generate basic recommendations when no specialized ones are found"""
        recommendations = []

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Basic scaling recommendation
        if numeric_cols:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="scaling",
                    priority="high",
                    description="Standardize numeric features for better model performance",
                    implementation=f'Apply StandardScaler to: {", ".join(numeric_cols[:3])}{"..." if len(numeric_cols) > 3 else ""}',
                    expected_benefit="Improved model convergence and performance",
                    computational_cost="low",
                )
            )

        # Basic categorical encoding
        if categorical_cols:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="categorical",
                    priority="high",
                    description="Encode categorical variables for machine learning",
                    implementation=f'Apply encoding to: {", ".join(categorical_cols[:3])}{"..." if len(categorical_cols) > 3 else ""}',
                    expected_benefit="Enable ML algorithms to process categorical data",
                    computational_cost="low",
                )
            )

        # Missing value handling
        missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
        if missing_cols:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="imputation",
                    priority="high",
                    description="Handle missing values in the dataset",
                    implementation=f'Impute missing values in: {", ".join(missing_cols[:3])}{"..." if len(missing_cols) > 3 else ""}',
                    expected_benefit="Complete dataset for model training",
                    computational_cost="low",
                )
            )

        # Feature selection
        if len(df.columns) > 10:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="selection",
                    priority="medium",
                    description="Select most relevant features to reduce dimensionality",
                    implementation=f"Apply feature selection to {len(df.columns)} features",
                    expected_benefit="Reduced overfitting and improved performance",
                    computational_cost="medium",
                )
            )

        # Basic interaction features for numeric data
        if len(numeric_cols) >= 2:
            recommendations.append(
                FeatureEngineeringRecommendation(
                    feature_type="interaction",
                    priority="medium",
                    description="Create interaction features between numeric variables",
                    implementation=f"Generate interactions between: {numeric_cols[0]} and {numeric_cols[1]}",
                    expected_benefit="Capture relationships between variables",
                    computational_cost="low",
                )
            )

        return recommendations
