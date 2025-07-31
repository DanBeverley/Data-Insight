"""Intelligent Data Understanding System for DataInsight AI"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from collections import Counter

class SemanticType(Enum):
    # Key types
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key" 
    NATURAL_KEY = "natural_key"
    
    # Contact information
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    IP_ADDRESS = "ip_address"
    
    # Numeric semantic types
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    COUNT = "count"
    RATIO = "ratio"
    
    # Categorical types
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_BINARY = "categorical_binary"
    
    # Temporal types
    DATETIME_TIMESTAMP = "datetime_timestamp"
    DATETIME_DATE = "datetime_date"
    DATETIME_PARTIAL = "datetime_partial"
    
    # Text types
    TEXT_SHORT = "text_short"
    TEXT_LONG = "text_long"
    TEXT_CODE = "text_code"
    
    # Geographic
    GEOLOCATION = "geolocation"
    ZIPCODE = "zipcode"
    COUNTRY_CODE = "country_code"
    
    # Default
    UNKNOWN = "unknown"

@dataclass
class SemanticProfile:
    column: str
    semantic_type: SemanticType
    confidence: float
    evidence: Dict[str, Any]
    recommendations: List[str]

class IntelligentDataProfiler:
    """AI-powered data understanding beyond basic dtypes"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.domain_keywords = self._initialize_domain_keywords()
        
    def _initialize_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for semantic type detection"""
        return {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'phone': re.compile(r'^[\+]?[1-9]?[\d\s\-\(\)\.]{7,15}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$', re.IGNORECASE),
            'ip_address': re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'),
            'zipcode': re.compile(r'^\d{5}(-\d{4})?$'),
            'country_code': re.compile(r'^[A-Z]{2,3}$'),
            'currency': re.compile(r'^[\$\€\£\¥]?[\d,]+\.?\d*$'),
            'percentage': re.compile(r'^\d+\.?\d*%?$'),
            'geolocation': re.compile(r'^-?\d+\.?\d*,-?\d+\.?\d*$')
        }
    
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain-specific keywords"""
        return {
            'id_indicators': ['id', 'key', 'pk', 'uid', 'uuid', 'identifier'],
            'temporal_indicators': ['date', 'time', 'timestamp', 'created', 'updated', 'modified'],
            'contact_indicators': ['email', 'phone', 'contact', 'address'],
            'financial_indicators': ['price', 'cost', 'amount', 'balance', 'revenue', 'profit'],
            'geographic_indicators': ['country', 'state', 'city', 'zip', 'postal', 'location']
        }
    
    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, SemanticProfile]:
        """Complete semantic profiling of dataset"""
        profiles = {}
        
        for column in df.columns:
            profile = self._profile_column(df[column], column)
            profiles[column] = profile
            
        # Post-process for relationship detection
        self._detect_relationships(df, profiles)
        
        return profiles
    
    def _profile_column(self, series: pd.Series, column_name: str) -> SemanticProfile:
        """Profile individual column for semantic type"""
        
        # Basic statistics
        non_null_count = series.count()
        total_count = len(series)
        null_ratio = 1 - (non_null_count / total_count)
        unique_count = series.nunique()
        cardinality = unique_count / non_null_count if non_null_count > 0 else 0
        
        evidence = {
            'dtype': str(series.dtype),
            'null_ratio': null_ratio,
            'cardinality': cardinality,
            'unique_count': unique_count,
            'sample_values': series.dropna().head(10).tolist()
        }
        
        # Detect semantic type
        semantic_type, confidence, type_evidence = self._detect_semantic_type(series, column_name)
        evidence.update(type_evidence)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(semantic_type, evidence)
        
        return SemanticProfile(
            column=column_name,
            semantic_type=semantic_type,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _detect_semantic_type(self, series: pd.Series, column_name: str) -> Tuple[SemanticType, float, Dict]:
        """Detect semantic type with confidence score"""
        
        column_lower = column_name.lower()
        evidence = {}
        
        # Key detection
        if self._is_key_column(series, column_name):
            key_type, confidence = self._classify_key_type(series, column_name)
            evidence['key_analysis'] = {'type': key_type.value, 'uniqueness': series.nunique() / len(series)}
            return key_type, confidence, evidence
        
        # Pattern-based detection for non-numeric types
        if series.dtype == 'object':
            pattern_result = self._detect_pattern_type(series)
            if pattern_result[0] != SemanticType.UNKNOWN:
                evidence['pattern_matches'] = pattern_result[2]
                return pattern_result[0], pattern_result[1], evidence
        
        # Temporal detection
        if self._is_temporal_column(series, column_name):
            temporal_type, confidence = self._classify_temporal_type(series)
            evidence['temporal_analysis'] = {'inferred_format': self._infer_datetime_format(series)}
            return temporal_type, confidence, evidence
        
        # Numeric semantic types
        if pd.api.types.is_numeric_dtype(series):
            numeric_type, confidence = self._classify_numeric_semantic_type(series, column_name)
            evidence['numeric_analysis'] = self._analyze_numeric_distribution(series)
            return numeric_type, confidence, evidence
        
        # Categorical detection
        if series.dtype == 'object' or pd.api.types.is_categorical_dtype(series):
            cat_type, confidence = self._classify_categorical_type(series)
            evidence['categorical_analysis'] = {
                'categories': series.value_counts().head(10).to_dict(),
                'category_count': series.nunique()
            }
            return cat_type, confidence, evidence
        
        return SemanticType.UNKNOWN, 0.3, evidence
    
    def _is_key_column(self, series: pd.Series, column_name: str) -> bool:
        """Detect if column is likely a key"""
        column_lower = column_name.lower()
        
        # Name-based detection
        if any(keyword in column_lower for keyword in self.domain_keywords['id_indicators']):
            return True
        
        # High cardinality with reasonable uniqueness
        cardinality = series.nunique() / len(series)
        if cardinality > 0.95 and series.nunique() > 10:
            return True
            
        return False
    
    def _classify_key_type(self, series: pd.Series, column_name: str) -> Tuple[SemanticType, float]:
        """Classify type of key column"""
        column_lower = column_name.lower()
        cardinality = series.nunique() / len(series)
        
        # Primary key indicators
        if cardinality == 1.0 and ('id' in column_lower or 'pk' in column_lower):
            return SemanticType.PRIMARY_KEY, 0.95
        
        # Foreign key indicators  
        if 0.7 <= cardinality < 1.0 and ('_id' in column_lower or 'fk' in column_lower):
            return SemanticType.FOREIGN_KEY, 0.85
        
        # Natural key (high uniqueness but not perfect)
        if cardinality > 0.8:
            return SemanticType.NATURAL_KEY, 0.75
            
        return SemanticType.PRIMARY_KEY, 0.6
    
    def _detect_pattern_type(self, series: pd.Series) -> Tuple[SemanticType, float, Dict]:
        """Detect semantic type based on string patterns"""
        sample_values = series.dropna().astype(str).head(100)
        pattern_matches = {}
        
        for pattern_name, pattern in self.patterns.items():
            matches = sum(1 for val in sample_values if pattern.match(val.strip()))
            match_ratio = matches / len(sample_values) if len(sample_values) > 0 else 0
            pattern_matches[pattern_name] = match_ratio
        
        # Find best match
        best_pattern = max(pattern_matches.items(), key=lambda x: x[1])
        
        if best_pattern[1] > 0.8:  # High confidence threshold
            semantic_type_map = {
                'email': SemanticType.EMAIL,
                'phone': SemanticType.PHONE,
                'url': SemanticType.URL,
                'ip_address': SemanticType.IP_ADDRESS,
                'zipcode': SemanticType.ZIPCODE,
                'country_code': SemanticType.COUNTRY_CODE,
                'currency': SemanticType.CURRENCY,
                'percentage': SemanticType.PERCENTAGE,
                'geolocation': SemanticType.GEOLOCATION
            }
            
            return semantic_type_map.get(best_pattern[0], SemanticType.UNKNOWN), best_pattern[1], pattern_matches
        
        return SemanticType.UNKNOWN, 0.0, pattern_matches
    
    def _is_temporal_column(self, series: pd.Series, column_name: str) -> bool:
        """Detect temporal columns"""
        column_lower = column_name.lower()
        
        # Name-based detection
        if any(keyword in column_lower for keyword in self.domain_keywords['temporal_indicators']):
            return True
        
        # Try parsing as datetime
        if series.dtype == 'object':
            try:
                pd.to_datetime(series.dropna().head(10))
                return True
            except:
                pass
                
        return pd.api.types.is_datetime64_any_dtype(series)
    
    def _classify_temporal_type(self, series: pd.Series) -> Tuple[SemanticType, float]:
        """Classify temporal column types"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return SemanticType.DATETIME_TIMESTAMP, 0.95
        
        # Try to infer from string patterns
        sample_values = series.dropna().astype(str).head(20)
        
        date_patterns = [
            (r'^\d{4}-\d{2}-\d{2}$', SemanticType.DATETIME_DATE),
            (r'^\d{2}/\d{2}/\d{4}$', SemanticType.DATETIME_DATE),
            (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', SemanticType.DATETIME_TIMESTAMP),
            (r'^\d{4}-\d{2}$', SemanticType.DATETIME_PARTIAL),
            (r'^\d{4}$', SemanticType.DATETIME_PARTIAL)
        ]
        
        for pattern, semantic_type in date_patterns:
            matches = sum(1 for val in sample_values if re.match(pattern, val.strip()))
            if matches / len(sample_values) > 0.7:
                return semantic_type, 0.8
        
        return SemanticType.DATETIME_TIMESTAMP, 0.6
    
    def _classify_numeric_semantic_type(self, series: pd.Series, column_name: str) -> Tuple[SemanticType, float]:
        """Classify numeric semantic types"""
        column_lower = column_name.lower()
        
        # Financial indicators
        if any(keyword in column_lower for keyword in self.domain_keywords['financial_indicators']):
            return SemanticType.CURRENCY, 0.85
        
        # Count detection (integers, non-negative, reasonable range)
        if pd.api.types.is_integer_dtype(series) and series.min() >= 0:
            if series.max() < 1000000:  # Reasonable count range
                return SemanticType.COUNT, 0.8
        
        # Percentage detection (0-100 range or 0-1 range)
        if series.min() >= 0:
            if series.max() <= 1.0:
                return SemanticType.PERCENTAGE, 0.85
            elif series.max() <= 100.0 and series.dtype == float:
                return SemanticType.PERCENTAGE, 0.75
        
        # Ratio detection (numeric with reasonable range)
        if series.min() >= 0 and series.max() < 1000:
            return SemanticType.RATIO, 0.6
        
        return SemanticType.UNKNOWN, 0.4
    
    def _classify_categorical_type(self, series: pd.Series) -> Tuple[SemanticType, float]:
        """Classify categorical types"""
        unique_count = series.nunique()
        total_count = len(series)
        
        # Binary categorical
        if unique_count == 2:
            return SemanticType.CATEGORICAL_BINARY, 0.9
        
        # Check for ordinal patterns
        if self._is_ordinal_categorical(series):
            return SemanticType.CATEGORICAL_ORDINAL, 0.8
        
        # High cardinality categorical (potential text)
        if unique_count / total_count > 0.5:
            avg_length = series.astype(str).str.len().mean()
            if avg_length > 50:
                return SemanticType.TEXT_LONG, 0.7
            elif avg_length > 10:
                return SemanticType.TEXT_SHORT, 0.75
            else:
                return SemanticType.TEXT_CODE, 0.7
        
        return SemanticType.CATEGORICAL_NOMINAL, 0.8
    
    def _is_ordinal_categorical(self, series: pd.Series) -> bool:
        """Detect if categorical data is ordinal"""
        unique_values = series.dropna().unique()
        
        # Common ordinal patterns
        ordinal_patterns = [
            ['low', 'medium', 'high'],
            ['small', 'medium', 'large'],
            ['poor', 'fair', 'good', 'excellent'],
            ['never', 'rarely', 'sometimes', 'often', 'always'],
            ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree']
        ]
        
        unique_lower = [str(val).lower() for val in unique_values]
        
        for pattern in ordinal_patterns:
            if set(unique_lower).issubset(set(pattern)):
                return True
        
        # Numeric-like strings
        if all(str(val).isdigit() for val in unique_values):
            return True
            
        return False
    
    def _analyze_numeric_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric distribution characteristics"""
        return {
            'min': float(series.min()),
            'max': float(series.max()),
            'mean': float(series.mean()),
            'std': float(series.std()),
            'skewness': float(series.skew()),
            'kurtosis': float(series.kurtosis()),
            'zeros_ratio': float((series == 0).sum() / len(series)),
            'negative_ratio': float((series < 0).sum() / len(series))
        }
    
    def _infer_datetime_format(self, series: pd.Series) -> Optional[str]:
        """Infer datetime format from string series"""
        sample_values = series.dropna().astype(str).head(5)
        
        common_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m',
            '%Y'
        ]
        
        for fmt in common_formats:
            try:
                parsed_count = sum(1 for val in sample_values 
                                 if pd.to_datetime(val, format=fmt, errors='coerce') is not pd.NaT)
                if parsed_count == len(sample_values):
                    return fmt
            except:
                continue
                
        return None
    
    def _detect_relationships(self, df: pd.DataFrame, profiles: Dict[str, SemanticProfile]) -> None:
        """Detect relationships between columns"""
        
        # Find potential foreign key relationships
        key_columns = {col: profile for col, profile in profiles.items() 
                      if profile.semantic_type in [SemanticType.PRIMARY_KEY, SemanticType.FOREIGN_KEY, SemanticType.NATURAL_KEY]}
        
        for col1, profile1 in key_columns.items():
            for col2, profile2 in key_columns.items():
                if col1 != col2:
                    # Check for value overlap
                    overlap = set(df[col1].dropna()) & set(df[col2].dropna())
                    overlap_ratio = len(overlap) / min(df[col1].nunique(), df[col2].nunique())
                    
                    if overlap_ratio > 0.1:  # Significant overlap
                        profile1.evidence['potential_relationships'] = profile1.evidence.get('potential_relationships', [])
                        profile1.evidence['potential_relationships'].append({
                            'related_column': col2,
                            'overlap_ratio': overlap_ratio,
                            'relationship_type': 'potential_fk' if profile1.semantic_type == SemanticType.FOREIGN_KEY else 'related_key'
                        })
    
    def _generate_recommendations(self, semantic_type: SemanticType, evidence: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on semantic type"""
        recommendations = []
        
        if semantic_type == SemanticType.EMAIL:
            recommendations.extend([
                "Extract email domain for feature engineering",
                "Validate email format and flag invalid entries", 
                "Consider email provider categorization (gmail, yahoo, corporate)"
            ])
        
        elif semantic_type == SemanticType.CURRENCY:
            recommendations.extend([
                "Normalize currency values (remove symbols, convert to float)",
                "Consider inflation adjustment for historical data",
                "Check for multiple currencies and standardize"
            ])
        
        elif semantic_type == SemanticType.DATETIME_TIMESTAMP:
            recommendations.extend([
                "Extract temporal features (day of week, month, quarter)",
                "Consider time zone normalization",
                "Generate lag and rolling window features for time series"
            ])
        
        elif semantic_type == SemanticType.CATEGORICAL_ORDINAL:
            recommendations.extend([
                "Use ordinal encoding instead of one-hot encoding",
                "Preserve order relationship in transformations",
                "Consider polynomial features to capture non-linear order effects"
            ])
        
        elif semantic_type == SemanticType.PRIMARY_KEY:
            recommendations.extend([
                "Exclude from model training (high cardinality, no predictive value)",
                "Use for data joining and relationship modeling",
                "Consider as index for time series or panel data"
            ])
        
        elif semantic_type == SemanticType.TEXT_LONG:
            recommendations.extend([
                "Apply NLP preprocessing (tokenization, stemming)",
                "Generate text features (length, word count, sentiment)", 
                "Consider topic modeling or text embeddings"
            ])
        
        # Add quality-based recommendations
        null_ratio = evidence.get('null_ratio', 0)
        if null_ratio > 0.5:
            recommendations.append("High missing values - consider imputation strategy or column removal")
        
        cardinality = evidence.get('cardinality', 0)
        if cardinality > 0.95 and semantic_type not in [SemanticType.PRIMARY_KEY, SemanticType.NATURAL_KEY]:
            recommendations.append("Very high cardinality - consider grouping or encoding strategies")
        
        return recommendations