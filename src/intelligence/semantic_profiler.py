import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SemanticType(Enum):
    PRICE = "price"
    REVENUE = "revenue" 
    RATING = "rating"
    SCORE = "score"
    AMOUNT = "amount"
    QUANTITY = "quantity"
    CATEGORY = "category"
    STATUS = "status"
    TYPE = "type"
    LABEL = "label"
    CLASS = "class"
    BINARY = "binary"
    IDENTIFIER = "identifier"
    NAME = "name"
    DESCRIPTION = "description"
    DATE = "date"
    TIME = "time"
    LOCATION = "location"
    DEMOGRAPHIC = "demographic"
    FEATURE = "feature"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"


class TaskRecommendation(Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: SemanticType
    target_suitability: float
    business_importance: float
    data_quality: float
    null_percentage: float
    unique_count: int
    unique_ratio: float
    patterns: List[str]
    keywords: List[str]
    preprocessing_needs: List[str]
    feature_engineering_suggestions: List[str]


@dataclass
class DatasetIntelligence:
    column_profiles: Dict[str, ColumnProfile]
    recommended_tasks: List[Tuple[TaskRecommendation, float]]
    target_recommendations: List[Tuple[str, TaskRecommendation, float]]
    feature_recommendations: List[str]
    data_quality_score: float
    business_domain: Optional[str]
    complexity_level: str


class SemanticColumnProfiler:
    def __init__(self):
        self.price_patterns = [
            r'price', r'cost', r'amount', r'fee', r'charge', r'fare', r'rate',
            r'revenue', r'income', r'salary', r'wage', r'value', r'worth'
        ]
        
        self.category_patterns = [
            r'category', r'type', r'class', r'group', r'segment', r'status',
            r'level', r'grade', r'rank', r'tier', r'kind'
        ]
        
        self.identifier_patterns = [
            r'id', r'key', r'index', r'number', r'code', r'ref'
        ]
        
        self.location_patterns = [
            r'address', r'city', r'state', r'country', r'region', r'area',
            r'location', r'place', r'zone', r'district'
        ]
        
        self.time_patterns = [
            r'date', r'time', r'year', r'month', r'day', r'created', r'updated',
            r'timestamp', r'when', r'at'
        ]
        
        self.domain_indicators = {
            'real_estate': ['property', 'house', 'apartment', 'room', 'area', 'location', 'furnish'],
            'finance': ['account', 'balance', 'transaction', 'payment', 'credit', 'debit'],
            'retail': ['product', 'item', 'sku', 'inventory', 'stock', 'sale'],
            'healthcare': ['patient', 'diagnosis', 'treatment', 'medical', 'health'],
            'marketing': ['campaign', 'customer', 'lead', 'conversion', 'engagement'],
            'hr': ['employee', 'salary', 'department', 'position', 'performance']
        }
    
    def profile_dataset(self, df: pd.DataFrame) -> DatasetIntelligence:
        try:
            logger.info(f"Profiling dataset with shape {df.shape}")
            
            column_profiles = {}
            for col in df.columns:
                column_profiles[col] = self._profile_column(df, col)
            
            business_domain = self._detect_business_domain(df, column_profiles)
            task_recommendations = self._recommend_tasks(df, column_profiles)
            target_recommendations = self._recommend_targets(column_profiles, task_recommendations)
            feature_recommendations = self._recommend_features(column_profiles)
            data_quality = np.mean([profile.data_quality for profile in column_profiles.values()])
            complexity = self._assess_complexity(df, column_profiles)
            
            intelligence = DatasetIntelligence(
                column_profiles=column_profiles,
                recommended_tasks=task_recommendations,
                target_recommendations=target_recommendations,
                feature_recommendations=feature_recommendations,
                data_quality_score=data_quality,
                business_domain=business_domain,
                complexity_level=complexity
            )
            
            logger.info(f"Dataset intelligence generated: domain={business_domain}, complexity={complexity}")
            return intelligence
            
        except Exception as e:
            logger.error(f"Dataset profiling failed: {e}")
            return DatasetIntelligence(
                column_profiles={},
                recommended_tasks=[(TaskRecommendation.REGRESSION, 0.5)],
                target_recommendations=[],
                feature_recommendations=[],
                data_quality_score=0.5,
                business_domain=None,
                complexity_level="unknown"
            )
    
    def _profile_column(self, df: pd.DataFrame, col: str) -> ColumnProfile:
        series = df[col]
        
        null_pct = series.isnull().sum() / len(series)
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / len(series) if len(series) > 0 else 0
        
        semantic_type = self._detect_semantic_type(col, series)
        target_suitability = self._calculate_target_suitability(col, series, semantic_type)
        business_importance = self._calculate_business_importance(col, semantic_type)
        data_quality = self._calculate_data_quality(series)
        
        patterns = self._extract_patterns(col, series)
        keywords = self._extract_keywords(col)
        
        preprocessing_needs = self._suggest_preprocessing(series, semantic_type)
        feature_engineering = self._suggest_feature_engineering(col, series, semantic_type)
        
        return ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            semantic_type=semantic_type,
            target_suitability=target_suitability,
            business_importance=business_importance,
            data_quality=data_quality,
            null_percentage=null_pct,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            patterns=patterns,
            keywords=keywords,
            preprocessing_needs=preprocessing_needs,
            feature_engineering_suggestions=feature_engineering
        )
    
    def _detect_semantic_type(self, col_name: str, series: pd.Series) -> SemanticType:
        col_lower = col_name.lower()
        
        if any(re.search(pattern, col_lower) for pattern in self.price_patterns):
            if pd.api.types.is_numeric_dtype(series):
                return SemanticType.PRICE
            else:
                return SemanticType.AMOUNT
        
        if any(re.search(pattern, col_lower) for pattern in self.category_patterns):
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            if unique_ratio < 0.1:
                if series.nunique() == 2:
                    return SemanticType.BINARY
                else:
                    return SemanticType.CATEGORY
            else:
                return SemanticType.TYPE
        
        if any(re.search(pattern, col_lower) for pattern in self.identifier_patterns):
            return SemanticType.IDENTIFIER
        
        if any(re.search(pattern, col_lower) for pattern in self.location_patterns):
            return SemanticType.LOCATION
        
        if any(re.search(pattern, col_lower) for pattern in self.time_patterns):
            return SemanticType.DATE if 'date' in col_lower else SemanticType.TIME
        
        if pd.api.types.is_datetime64_any_dtype(series):
            return SemanticType.DATE
        
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() / len(series) > 0.8:
                if series.min() >= 0 and series.max() <= 10:
                    return SemanticType.RATING
                else:
                    return SemanticType.QUANTITY
            else:
                return SemanticType.SCORE
        
        if pd.api.types.is_categorical_dtype(series) or series.dtype == 'object':
            unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
            if unique_ratio < 0.05:
                return SemanticType.CATEGORY
            elif unique_ratio < 0.5:
                return SemanticType.STATUS
            else:
                return SemanticType.NAME
        
        return SemanticType.UNKNOWN
    
    def _calculate_target_suitability(self, col_name: str, series: pd.Series, semantic_type: SemanticType) -> float:
        score = 0.0
        
        target_semantic_scores = {
            SemanticType.PRICE: 0.9,
            SemanticType.REVENUE: 0.9,
            SemanticType.RATING: 0.8,
            SemanticType.SCORE: 0.8,
            SemanticType.CATEGORY: 0.7,
            SemanticType.BINARY: 0.8,
            SemanticType.STATUS: 0.6,
            SemanticType.IDENTIFIER: 0.1,
            SemanticType.NAME: 0.1,
            SemanticType.IRRELEVANT: 0.0
        }
        score += target_semantic_scores.get(semantic_type, 0.3)
        
        null_ratio = series.isnull().sum() / len(series)
        score *= (1 - null_ratio)
        
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        if semantic_type in [SemanticType.CATEGORY, SemanticType.BINARY, SemanticType.STATUS]:
            if unique_ratio < 0.1:
                score *= 1.2
            elif unique_ratio > 0.5:
                score *= 0.5
        
        business_keywords = ['target', 'outcome', 'result', 'predict', 'goal']
        if any(keyword in col_name.lower() for keyword in business_keywords):
            score *= 1.3
        
        return min(score, 1.0)
    
    def _calculate_business_importance(self, col_name: str, semantic_type: SemanticType) -> float:
        importance_scores = {
            SemanticType.PRICE: 0.9,
            SemanticType.REVENUE: 0.9,
            SemanticType.CATEGORY: 0.7,
            SemanticType.STATUS: 0.7,
            SemanticType.RATING: 0.8,
            SemanticType.LOCATION: 0.6,
            SemanticType.DATE: 0.6,
            SemanticType.IDENTIFIER: 0.2,
            SemanticType.IRRELEVANT: 0.1
        }
        return importance_scores.get(semantic_type, 0.5)
    
    def _calculate_data_quality(self, series: pd.Series) -> float:
        score = 1.0
        
        null_ratio = series.isnull().sum() / len(series)
        score *= (1 - null_ratio)
        
        unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
        if unique_ratio < 0.01 or unique_ratio > 0.99:
            score *= 0.8
        
        if pd.api.types.is_numeric_dtype(series):
            score *= 1.1
        elif pd.api.types.is_datetime64_any_dtype(series):
            score *= 1.1
        
        return min(score, 1.0)
    
    def _detect_business_domain(self, df: pd.DataFrame, profiles: Dict[str, ColumnProfile]) -> Optional[str]:
        domain_scores = {}
        
        for domain, keywords in self.domain_indicators.items():
            score = 0
            for col in df.columns:
                col_lower = col.lower()
                matches = sum(1 for keyword in keywords if keyword in col_lower)
                score += matches
            domain_scores[domain] = score
        
        if domain_scores and max(domain_scores.values()) > 0:
            return max(domain_scores.keys(), key=lambda k: domain_scores[k])
        
        return None
    
    def _recommend_tasks(self, df: pd.DataFrame, profiles: Dict[str, ColumnProfile]) -> List[Tuple[TaskRecommendation, float]]:
        recommendations = []
        
        semantic_counts = {}
        for profile in profiles.values():
            semantic_counts[profile.semantic_type] = semantic_counts.get(profile.semantic_type, 0) + 1
        
        regression_indicators = [SemanticType.PRICE, SemanticType.REVENUE, SemanticType.RATING, SemanticType.SCORE]
        regression_score = sum(semantic_counts.get(sem, 0) for sem in regression_indicators) / len(profiles)
        if regression_score > 0:
            recommendations.append((TaskRecommendation.REGRESSION, regression_score))
        
        classification_indicators = [SemanticType.CATEGORY, SemanticType.BINARY, SemanticType.STATUS]
        classification_score = sum(semantic_counts.get(sem, 0) for sem in classification_indicators) / len(profiles)
        if classification_score > 0:
            if semantic_counts.get(SemanticType.BINARY, 0) > 0:
                recommendations.append((TaskRecommendation.BINARY_CLASSIFICATION, classification_score * 1.2))
            else:
                recommendations.append((TaskRecommendation.MULTICLASS_CLASSIFICATION, classification_score))
        
        if semantic_counts.get(SemanticType.DATE, 0) > 0:
            recommendations.append((TaskRecommendation.TIME_SERIES, 0.6))
        
        recommendations.append((TaskRecommendation.CLUSTERING, 0.4))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:3]
    
    def _recommend_targets(self, profiles: Dict[str, ColumnProfile], 
                          task_recommendations: List[Tuple[TaskRecommendation, float]]) -> List[Tuple[str, TaskRecommendation, float]]:
        recommendations = []
        
        sorted_columns = sorted(profiles.items(), key=lambda x: x[1].target_suitability, reverse=True)
        
        for task, task_confidence in task_recommendations:
            for col_name, profile in sorted_columns[:3]:
                if task in [TaskRecommendation.REGRESSION]:
                    if profile.semantic_type in [SemanticType.PRICE, SemanticType.REVENUE, SemanticType.RATING, SemanticType.SCORE]:
                        confidence = profile.target_suitability * task_confidence
                        recommendations.append((col_name, task, confidence))
                        break
                elif task in [TaskRecommendation.BINARY_CLASSIFICATION, TaskRecommendation.MULTICLASS_CLASSIFICATION]:
                    if profile.semantic_type in [SemanticType.CATEGORY, SemanticType.BINARY, SemanticType.STATUS]:
                        confidence = profile.target_suitability * task_confidence
                        recommendations.append((col_name, task, confidence))
                        break
        
        return recommendations
    
    def _recommend_features(self, profiles: Dict[str, ColumnProfile]) -> List[str]:
        sorted_features = sorted(
            [(name, profile) for name, profile in profiles.items() 
             if profile.semantic_type not in [SemanticType.IDENTIFIER, SemanticType.IRRELEVANT]],
            key=lambda x: x[1].business_importance * x[1].data_quality,
            reverse=True
        )
        
        return [name for name, _ in sorted_features[:10]]
    
    def _assess_complexity(self, df: pd.DataFrame, profiles: Dict[str, ColumnProfile]) -> str:
        n_rows, n_cols = df.shape
        
        if n_rows < 1000 and n_cols < 10:
            return "simple"
        elif n_rows < 10000 and n_cols < 50:
            return "moderate"
        else:
            return "complex"
    
    def _extract_patterns(self, col_name: str, series: pd.Series) -> List[str]:
        patterns = []
        
        if pd.api.types.is_numeric_dtype(series):
            patterns.append("numeric")
            if series.min() >= 0:
                patterns.append("non_negative")
            if series.nunique() < 10:
                patterns.append("low_cardinality")
        
        if series.dtype == 'object':
            patterns.append("text")
            avg_length = series.astype(str).str.len().mean()
            if avg_length < 20:
                patterns.append("short_text")
            else:
                patterns.append("long_text")
        
        return patterns
    
    def _extract_keywords(self, col_name: str) -> List[str]:
        keywords = []
        col_lower = col_name.lower()
        
        all_patterns = (self.price_patterns + self.category_patterns + 
                       self.identifier_patterns + self.location_patterns + 
                       self.time_patterns)
        
        for pattern in all_patterns:
            if re.search(pattern, col_lower):
                keywords.append(pattern.strip('r'))
        
        return keywords
    
    def _suggest_preprocessing(self, series: pd.Series, semantic_type: SemanticType) -> List[str]:
        suggestions = []
        
        if series.isnull().sum() > 0:
            if semantic_type in [SemanticType.CATEGORY, SemanticType.STATUS]:
                suggestions.append("impute_mode")
            elif semantic_type in [SemanticType.PRICE, SemanticType.QUANTITY]:
                suggestions.append("impute_median")
            else:
                suggestions.append("handle_missing")
        
        if series.dtype == 'object' and semantic_type == SemanticType.CATEGORY:
            suggestions.append("encode_categorical")
        
        if pd.api.types.is_numeric_dtype(series):
            if series.std() > series.mean():
                suggestions.append("scale_features")
        
        return suggestions
    
    def _suggest_feature_engineering(self, col_name: str, series: pd.Series, semantic_type: SemanticType) -> List[str]:
        suggestions = []
        
        if semantic_type == SemanticType.DATE:
            suggestions.extend(["extract_year", "extract_month", "extract_day"])
        
        if semantic_type == SemanticType.LOCATION:
            suggestions.append("geocode_location")
        
        if series.dtype == 'object' and semantic_type == SemanticType.NAME:
            suggestions.append("extract_name_features")
        
        if pd.api.types.is_numeric_dtype(series):
            suggestions.extend(["create_bins", "polynomial_features"])
        
        return suggestions