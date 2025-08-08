import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import re
from collections import Counter
from dataclasses import dataclass

@dataclass
class DomainMatch:
    domain: str
    confidence: float
    matched_columns: List[str]
    evidence: List[str]

class DomainDetector:
    def __init__(self):
        self.domain_signatures = self._initialize_domain_signatures()
        self.column_patterns = self._initialize_column_patterns()
        self.value_patterns = self._initialize_value_patterns()
    
    def _initialize_domain_signatures(self) -> Dict[str, Dict[str, List[str]]]:
        """Define domain-specific column names and value patterns"""
        return {
            'ecommerce': {
                'columns': ['product_id', 'sku', 'price', 'quantity', 'order_id', 'customer_id', 
                           'cart', 'checkout', 'payment', 'shipping', 'inventory', 'category'],
                'values': ['add_to_cart', 'purchase', 'checkout', 'shipped', 'delivered', 'refund']
            },
            'finance': {
                'columns': ['account', 'balance', 'transaction', 'amount', 'credit', 'debit', 
                           'portfolio', 'investment', 'loan', 'interest', 'fee', 'commission'],
                'values': ['deposit', 'withdrawal', 'transfer', 'payment', 'dividend', 'interest']
            },
            'healthcare': {
                'columns': ['patient_id', 'diagnosis', 'treatment', 'medication', 'dosage', 
                           'doctor', 'hospital', 'symptoms', 'vital_signs', 'lab_results'],
                'values': ['prescribed', 'diagnosed', 'treated', 'admitted', 'discharged']
            },
            'marketing': {
                'columns': ['campaign', 'channel', 'impression', 'click', 'conversion', 'ctr', 
                           'cpc', 'cpm', 'roi', 'lead', 'funnel', 'attribution'],
                'values': ['email', 'social', 'search', 'display', 'video', 'mobile']
            },
            'hr': {
                'columns': ['employee_id', 'department', 'position', 'salary', 'hire_date', 
                           'performance', 'benefits', 'vacation', 'overtime', 'manager'],
                'values': ['hired', 'promoted', 'terminated', 'reviewed', 'trained']
            },
            'logistics': {
                'columns': ['shipment', 'tracking', 'delivery', 'warehouse', 'route', 'carrier', 
                           'weight', 'dimensions', 'origin', 'destination', 'transit_time'],
                'values': ['picked', 'packed', 'shipped', 'in_transit', 'delivered', 'returned']
            },
            'iot_sensor': {
                'columns': ['sensor_id', 'timestamp', 'temperature', 'humidity', 'pressure', 
                           'vibration', 'voltage', 'current', 'status', 'location'],
                'values': ['online', 'offline', 'error', 'maintenance', 'calibrated']
            },
            'social_media': {
                'columns': ['user_id', 'post', 'comment', 'like', 'share', 'follower', 'hashtag', 
                           'mention', 'engagement', 'reach', 'impression'],
                'values': ['posted', 'liked', 'shared', 'commented', 'followed', 'unfollowed']
            },
            'gaming': {
                'columns': ['player_id', 'level', 'score', 'achievement', 'session', 'playtime', 
                           'character', 'guild', 'item', 'currency', 'experience'],
                'values': ['login', 'logout', 'levelup', 'died', 'completed', 'purchased']
            },
            'education': {
                'columns': ['student_id', 'course', 'grade', 'assignment', 'exam', 'attendance', 
                           'gpa', 'credit', 'semester', 'major', 'instructor'],
                'values': ['enrolled', 'completed', 'passed', 'failed', 'graduated', 'dropped']
            }
        }
    
    def _initialize_column_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for column name matching"""
        return {
            'id_pattern': re.compile(r'.*_?id$|^id_?.*|.*key.*', re.IGNORECASE),
            'date_pattern': re.compile(r'.*date.*|.*time.*|.*created.*|.*updated.*', re.IGNORECASE),
            'amount_pattern': re.compile(r'.*amount.*|.*price.*|.*cost.*|.*fee.*', re.IGNORECASE),
            'location_pattern': re.compile(r'.*location.*|.*address.*|.*city.*|.*country.*', re.IGNORECASE),
            'status_pattern': re.compile(r'.*status.*|.*state.*|.*condition.*', re.IGNORECASE)
        }
    
    def _initialize_value_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for value matching"""
        return {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'phone': re.compile(r'^[\+]?[1-9][\d\s\-\(\)]{7,15}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'ip_address': re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'),
            'currency': re.compile(r'^[\$\€\£\¥]?[\d,]+\.?\d*$'),
            'percentage': re.compile(r'^\d+\.?\d*%?$')
        }
    
    def detect_domain(self, df: pd.DataFrame, column_profiles: Optional[Dict] = None) -> List[DomainMatch]:
        """Detect business domain from dataset characteristics"""
        matches = []
        
        for domain, signatures in self.domain_signatures.items():
            confidence, evidence = self._calculate_domain_confidence(df, domain, signatures, column_profiles)
            
            if confidence > 0.3:
                matched_columns = self._find_matching_columns(df, signatures['columns'])
                matches.append(DomainMatch(
                    domain=domain,
                    confidence=confidence,
                    matched_columns=matched_columns,
                    evidence=evidence
                ))
        
        return sorted(matches, key=lambda x: x.confidence, reverse=True)
    
    def _calculate_domain_confidence(self, df: pd.DataFrame, domain: str, 
                                   signatures: Dict[str, List[str]], 
                                   column_profiles: Optional[Dict] = None) -> Tuple[float, List[str]]:
        """Calculate confidence score for a specific domain"""
        evidence = []
        scores = []
        
        # Column name matching
        column_matches = self._match_column_names(df.columns.tolist(), signatures['columns'])
        if column_matches:
            column_score = len(column_matches) / len(df.columns)
            scores.append(column_score * 0.4)
            evidence.extend([f"Column match: {col}" for col in column_matches[:3]])
        
        # Value pattern matching
        value_matches = self._match_value_patterns(df, signatures['values'])
        if value_matches:
            value_score = sum(value_matches.values()) / len(df.columns)
            scores.append(min(value_score, 1.0) * 0.3)
            evidence.extend([f"Value pattern: {pattern}" for pattern in list(value_matches.keys())[:3]])
        
        # Semantic type consistency
        if column_profiles:
            semantic_score = self._calculate_semantic_consistency(column_profiles, domain)
            scores.append(semantic_score * 0.3)
            if semantic_score > 0.5:
                evidence.append(f"Strong semantic alignment with {domain}")
        
        confidence = sum(scores) if scores else 0.0
        return min(confidence, 1.0), evidence
    
    def _match_column_names(self, columns: List[str], domain_columns: List[str]) -> List[str]:
        """Find columns that match domain-specific naming patterns"""
        matches = []
        for col in columns:
            col_lower = col.lower().replace('_', '').replace('-', '')
            for domain_col in domain_columns:
                domain_col_clean = domain_col.lower().replace('_', '').replace('-', '')
                if domain_col_clean in col_lower or col_lower in domain_col_clean:
                    matches.append(col)
                    break
        return matches
    
    def _match_value_patterns(self, df: pd.DataFrame, domain_values: List[str]) -> Dict[str, float]:
        """Find values that match domain-specific patterns"""
        matches = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_values = df[col].dropna().astype(str).str.lower().unique()
                
                for domain_value in domain_values:
                    pattern_matches = sum(1 for val in unique_values 
                                        if domain_value.lower() in val or val in domain_value.lower())
                    
                    if pattern_matches > 0:
                        match_ratio = pattern_matches / len(unique_values)
                        if match_ratio > 0.1:
                            matches[f"{col}:{domain_value}"] = match_ratio
        
        return matches
    
    def _calculate_semantic_consistency(self, column_profiles: Dict, domain: str) -> float:
        """Calculate how well semantic types align with expected domain patterns"""
        domain_semantic_expectations = {
            'ecommerce': ['currency', 'primary_key', 'foreign_key', 'categorical'],
            'finance': ['currency', 'percentage', 'decimal', 'temporal'],
            'healthcare': ['primary_key', 'categorical', 'temporal', 'text'],
            'marketing': ['percentage', 'decimal', 'categorical', 'temporal'],
            'hr': ['currency', 'temporal', 'categorical', 'text'],
            'logistics': ['geolocation', 'temporal', 'categorical', 'decimal'],
            'iot_sensor': ['decimal', 'temporal', 'categorical', 'geolocation'],
            'social_media': ['text', 'temporal', 'categorical', 'primary_key'],
            'gaming': ['primary_key', 'decimal', 'categorical', 'temporal'],
            'education': ['primary_key', 'decimal', 'categorical', 'temporal']
        }
        
        expected_types = set(domain_semantic_expectations.get(domain, []))
        found_types = set()
        
        for col_profile in column_profiles.values():
            if hasattr(col_profile, 'semantic_type'):
                found_types.add(col_profile.semantic_type.value)
        
        if not expected_types:
            return 0.0
        
        overlap = len(expected_types.intersection(found_types))
        return overlap / len(expected_types)
    
    def _find_matching_columns(self, df: pd.DataFrame, domain_columns: List[str]) -> List[str]:
        """Find specific columns that contributed to domain matching"""
        return self._match_column_names(df.columns.tolist(), domain_columns)
    
    def get_domain_recommendations(self, domain_matches: List[DomainMatch]) -> Dict[str, List[str]]:
        """Generate domain-specific recommendations"""
        if not domain_matches:
            return {'general': ['Consider adding domain-specific identifiers', 
                               'Add temporal columns for trend analysis']}
        
        top_domain = domain_matches[0]
        domain = top_domain.domain
        
        recommendations = {
            'ecommerce': [
                'Add customer segmentation features',
                'Calculate cart abandonment metrics',
                'Create seasonal purchase patterns',
                'Generate product recommendation features'
            ],
            'finance': [
                'Calculate moving averages for amounts',
                'Add risk assessment features',
                'Create volatility indicators',
                'Generate portfolio balance features'
            ],
            'healthcare': [
                'Add patient risk stratification',
                'Create treatment outcome predictions',
                'Generate medication interaction features',
                'Add temporal health trend analysis'
            ],
            'marketing': [
                'Calculate customer lifetime value',
                'Add attribution modeling features',
                'Create funnel conversion metrics',
                'Generate campaign performance indicators'
            ],
            'hr': [
                'Add employee retention predictors',
                'Create performance trend analysis',
                'Generate compensation benchmarking',
                'Add skill gap identification features'
            ]
        }
        
        return {domain: recommendations.get(domain, ['Add domain-specific derived features'])}