import hashlib
import secrets
import re
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

class SecurityLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MAXIMUM = "maximum"

class AccessLevel(Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    SCIENTIST = "scientist"
    ADMIN = "admin"

@dataclass
class PIIDetectionResult:
    column: str
    pii_type: str
    confidence: float
    sample_values: List[str]
    masking_applied: bool = False

@dataclass
class AccessEvent:
    user_id: str
    action: str
    resource: str
    timestamp: str
    ip_address: Optional[str] = None
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityPolicy:
    encryption_enabled: bool = True
    pii_detection_enabled: bool = True
    audit_logging_enabled: bool = True
    session_timeout_minutes: int = 30
    max_failed_attempts: int = 3
    anonymization_k: int = 5
    differential_privacy_epsilon: float = 1.0

class SecurityManager:
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        self.security_level = security_level
        self.policy = self._get_security_policy()
        self.pii_patterns = self._initialize_pii_patterns()
        self.access_log = []
        self.failed_attempts = {}
        self.active_sessions = {}
        
    def _get_security_policy(self) -> SecurityPolicy:
        if self.security_level == SecurityLevel.BASIC:
            return SecurityPolicy(
                encryption_enabled=False,
                pii_detection_enabled=True,
                session_timeout_minutes=60,
                anonymization_k=3
            )
        elif self.security_level == SecurityLevel.HIGH:
            return SecurityPolicy(
                session_timeout_minutes=15,
                max_failed_attempts=2,
                anonymization_k=10,
                differential_privacy_epsilon=0.5
            )
        elif self.security_level == SecurityLevel.MAXIMUM:
            return SecurityPolicy(
                session_timeout_minutes=10,
                max_failed_attempts=1,
                anonymization_k=20,
                differential_privacy_epsilon=0.1
            )
        else:
            return SecurityPolicy()
    
    def _initialize_pii_patterns(self) -> Dict[str, Dict[str, Any]]:
        return {
            'ssn': {
                'pattern': r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
                'confidence_threshold': 0.9,
                'description': 'Social Security Number'
            },
            'email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'confidence_threshold': 0.95,
                'description': 'Email Address'
            },
            'phone': {
                'pattern': r'\b(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
                'confidence_threshold': 0.8,
                'description': 'Phone Number'
            },
            'credit_card': {
                'pattern': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                'confidence_threshold': 0.85,
                'description': 'Credit Card Number'
            },
            'ip_address': {
                'pattern': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                'confidence_threshold': 0.9,
                'description': 'IP Address'
            }
        }
    
    def detect_pii(self, df: pd.DataFrame, sample_size: int = 100) -> List[PIIDetectionResult]:
        results = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                pii_result = self._analyze_column_for_pii(df[column], column, sample_size)
                if pii_result:
                    results.append(pii_result)
        
        return results
    
    def _analyze_column_for_pii(self, series: pd.Series, column_name: str, 
                               sample_size: int) -> Optional[PIIDetectionResult]:
        
        sample_data = series.dropna().astype(str).sample(min(sample_size, len(series.dropna())))
        
        for pii_type, config in self.pii_patterns.items():
            pattern = config['pattern']
            matches = 0
            sample_matches = []
            
            for value in sample_data:
                if re.search(pattern, str(value)):
                    matches += 1
                    if len(sample_matches) < 3:
                        sample_matches.append(str(value))
            
            confidence = matches / len(sample_data) if len(sample_data) > 0 else 0
            
            if confidence >= config['confidence_threshold']:
                return PIIDetectionResult(
                    column=column_name,
                    pii_type=pii_type,
                    confidence=confidence,
                    sample_values=sample_matches
                )
        
        if self._is_likely_identifier(series, column_name):
            return PIIDetectionResult(
                column=column_name,
                pii_type='identifier',
                confidence=0.8,
                sample_values=sample_data.head(3).tolist()
            )
        
        return None
    
    def _is_likely_identifier(self, series: pd.Series, column_name: str) -> bool:
        identifier_keywords = ['id', 'key', 'uuid', 'guid', 'name', 'username']
        column_lower = column_name.lower()
        
        if any(keyword in column_lower for keyword in identifier_keywords):
            uniqueness_ratio = series.nunique() / len(series)
            return uniqueness_ratio > 0.8
        
        return False
    
    def mask_pii_data(self, df: pd.DataFrame, pii_results: List[PIIDetectionResult]) -> pd.DataFrame:
        masked_df = df.copy()
        
        for pii_result in pii_results:
            column = pii_result.column
            pii_type = pii_result.pii_type
            
            if pii_type == 'email':
                masked_df[column] = masked_df[column].apply(self._mask_email)
            elif pii_type == 'phone':
                masked_df[column] = masked_df[column].apply(self._mask_phone)
            elif pii_type == 'ssn':
                masked_df[column] = masked_df[column].apply(self._mask_ssn)
            elif pii_type == 'credit_card':
                masked_df[column] = masked_df[column].apply(self._mask_credit_card)
            else:
                masked_df[column] = masked_df[column].apply(self._generic_mask)
            
            pii_result.masking_applied = True
        
        return masked_df
    
    def _mask_email(self, email: str) -> str:
        if pd.isna(email) or not isinstance(email, str):
            return email
        
        try:
            local, domain = email.split('@')
            masked_local = local[0] + '*' * (len(local) - 1) if len(local) > 1 else '*'
            return f"{masked_local}@{domain}"
        except:
            return self._generic_mask(email)
    
    def _mask_phone(self, phone: str) -> str:
        if pd.isna(phone) or not isinstance(phone, str):
            return phone
        
        digits_only = re.sub(r'\D', '', str(phone))
        if len(digits_only) >= 10:
            return f"***-***-{digits_only[-4:]}"
        return self._generic_mask(phone)
    
    def _mask_ssn(self, ssn: str) -> str:
        if pd.isna(ssn) or not isinstance(ssn, str):
            return ssn
        
        digits_only = re.sub(r'\D', '', str(ssn))
        if len(digits_only) == 9:
            return f"***-**-{digits_only[-4:]}"
        return self._generic_mask(ssn)
    
    def _mask_credit_card(self, cc: str) -> str:
        if pd.isna(cc) or not isinstance(cc, str):
            return cc
        
        digits_only = re.sub(r'\D', '', str(cc))
        if len(digits_only) >= 12:
            return f"****-****-****-{digits_only[-4:]}"
        return self._generic_mask(cc)
    
    def _generic_mask(self, value: str) -> str:
        if pd.isna(value) or not isinstance(value, str):
            return value
        
        if len(str(value)) <= 3:
            return '*' * len(str(value))
        else:
            return str(value)[:2] + '*' * (len(str(value)) - 2)
    
    def anonymize_data(self, df: pd.DataFrame, k_value: Optional[int] = None) -> pd.DataFrame:
        k = k_value or self.policy.anonymization_k
        
        anonymized_df = df.copy()
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                anonymized_df[column] = self._add_noise(df[column])
            elif df[column].dtype == 'object':
                if df[column].nunique() < len(df) * 0.1:
                    anonymized_df[column] = self._generalize_categorical(df[column], k)
        
        return anonymized_df
    
    def _add_noise(self, series: pd.Series, noise_factor: float = 0.01) -> pd.Series:
        if self.policy.differential_privacy_epsilon:
            epsilon = self.policy.differential_privacy_epsilon
            sensitivity = series.std()
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale, len(series))
            return series + noise
        else:
            std_dev = series.std() * noise_factor
            noise = np.random.normal(0, std_dev, len(series))
            return series + noise
    
    def _generalize_categorical(self, series: pd.Series, k: int) -> pd.Series:
        value_counts = series.value_counts()
        rare_values = value_counts[value_counts < k].index
        
        generalized_series = series.copy()
        generalized_series[series.isin(rare_values)] = 'Other'
        
        return generalized_series
    
    def encrypt_data(self, data: Union[str, bytes], key: Optional[str] = None) -> Tuple[str, str]:
        if not self.policy.encryption_enabled:
            return str(data), ""
        
        try:
            import cryptography.fernet as fernet
            
            if key is None:
                key = fernet.Fernet.generate_key()
            elif isinstance(key, str):
                key = key.encode()
            
            f = fernet.Fernet(key)
            
            if isinstance(data, str):
                data = data.encode()
            
            encrypted_data = f.encrypt(data)
            return encrypted_data.decode(), key.decode()
            
        except ImportError:
            return str(data), ""
    
    def decrypt_data(self, encrypted_data: str, key: str) -> str:
        if not self.policy.encryption_enabled:
            return encrypted_data
        
        try:
            import cryptography.fernet as fernet
            
            f = fernet.Fernet(key.encode())
            decrypted_data = f.decrypt(encrypted_data.encode())
            return decrypted_data.decode()
            
        except ImportError:
            return encrypted_data
    
    def create_session(self, user_id: str, access_level: AccessLevel) -> str:
        session_id = secrets.token_urlsafe(32)
        
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'access_level': access_level,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'ip_address': None
        }
        
        self._log_access_event(user_id, 'session_created', session_id, success=True)
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        now = datetime.now()
        
        timeout_delta = timedelta(minutes=self.policy.session_timeout_minutes)
        if now - session['last_activity'] > timeout_delta:
            self.invalidate_session(session_id)
            return None
        
        session['last_activity'] = now
        return session
    
    def invalidate_session(self, session_id: str) -> bool:
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            self._log_access_event(session['user_id'], 'session_invalidated', session_id)
            del self.active_sessions[session_id]
            return True
        return False
    
    def check_access_permission(self, session_id: str, resource: str, action: str) -> bool:
        session = self.validate_session(session_id)
        if not session:
            return False
        
        access_level = session['access_level']
        user_id = session['user_id']
        
        permission_granted = self._evaluate_permission(access_level, resource, action)
        
        self._log_access_event(
            user_id, f"{action}_{resource}", resource, 
            success=permission_granted
        )
        
        return permission_granted
    
    def _evaluate_permission(self, access_level: AccessLevel, resource: str, action: str) -> bool:
        permissions = {
            AccessLevel.VIEWER: {
                'data': ['read'],
                'model': ['read'],
                'pipeline': ['read']
            },
            AccessLevel.ANALYST: {
                'data': ['read', 'analyze'],
                'model': ['read', 'validate'],
                'pipeline': ['read', 'execute']
            },
            AccessLevel.SCIENTIST: {
                'data': ['read', 'analyze', 'modify'],
                'model': ['read', 'train', 'validate', 'deploy'],
                'pipeline': ['read', 'execute', 'modify']
            },
            AccessLevel.ADMIN: {
                'data': ['read', 'analyze', 'modify', 'delete'],
                'model': ['read', 'train', 'validate', 'deploy', 'delete'],
                'pipeline': ['read', 'execute', 'modify', 'delete'],
                'system': ['configure', 'manage', 'audit']
            }
        }
        
        allowed_actions = permissions.get(access_level, {}).get(resource, [])
        return action in allowed_actions
    
    def _log_access_event(self, user_id: str, action: str, resource: str, 
                         success: bool = True, ip_address: Optional[str] = None):
        if not self.policy.audit_logging_enabled:
            return
        
        event = AccessEvent(
            user_id=user_id,
            action=action,
            resource=resource,
            timestamp=datetime.now().isoformat(),
            ip_address=ip_address,
            success=success
        )
        
        self.access_log.append(event)
        
        if not success:
            self._handle_failed_attempt(user_id)
    
    def _handle_failed_attempt(self, user_id: str):
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = {
                'count': 0,
                'first_attempt': datetime.now(),
                'locked_until': None
            }
        
        self.failed_attempts[user_id]['count'] += 1
        
        if self.failed_attempts[user_id]['count'] >= self.policy.max_failed_attempts:
            lockout_duration = timedelta(minutes=15)
            self.failed_attempts[user_id]['locked_until'] = datetime.now() + lockout_duration
    
    def is_user_locked(self, user_id: str) -> bool:
        if user_id not in self.failed_attempts:
            return False
        
        locked_until = self.failed_attempts[user_id].get('locked_until')
        if locked_until and datetime.now() < locked_until:
            return True
        
        return False
    
    def get_security_audit(self, hours_back: int = 24) -> Dict[str, Any]:
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_events = [
            event for event in self.access_log
            if datetime.fromisoformat(event.timestamp) > cutoff_time
        ]
        
        failed_events = [event for event in recent_events if not event.success]
        
        audit_summary = {
            'total_events': len(recent_events),
            'failed_events': len(failed_events),
            'active_sessions': len(self.active_sessions),
            'locked_users': len([uid for uid, data in self.failed_attempts.items() 
                               if self.is_user_locked(uid)]),
            'event_breakdown': {},
            'user_activity': {},
            'security_violations': []
        }
        
        for event in recent_events:
            action = event.action
            user = event.user_id
            
            if action not in audit_summary['event_breakdown']:
                audit_summary['event_breakdown'][action] = 0
            audit_summary['event_breakdown'][action] += 1
            
            if user not in audit_summary['user_activity']:
                audit_summary['user_activity'][user] = {'total': 0, 'failed': 0}
            audit_summary['user_activity'][user]['total'] += 1
            
            if not event.success:
                audit_summary['user_activity'][user]['failed'] += 1
                audit_summary['security_violations'].append({
                    'user': user,
                    'action': action,
                    'resource': event.resource,
                    'timestamp': event.timestamp
                })
        
        return audit_summary
    
    def scan_data_security(self, df: pd.DataFrame) -> Dict[str, Any]:
        pii_results = self.detect_pii(df)
        
        security_score = 1.0
        risks = []
        
        if pii_results:
            pii_risk = len(pii_results) / len(df.columns)
            security_score -= min(0.5, pii_risk)
            risks.append(f"PII detected in {len(pii_results)} columns")
        
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_rate > 0.3:
            security_score -= 0.1
            risks.append("High missing data rate may indicate data quality issues")
        
        duplicate_rate = df.duplicated().sum() / len(df)
        if duplicate_rate > 0.1:
            security_score -= 0.1
            risks.append("High duplicate rate detected")
        
        return {
            'security_score': max(0, security_score),
            'pii_detected': len(pii_results) > 0,
            'pii_details': [
                {
                    'column': result.column,
                    'type': result.pii_type,
                    'confidence': result.confidence
                }
                for result in pii_results
            ],
            'security_risks': risks,
            'recommendations': self._generate_security_recommendations(pii_results, security_score)
        }
    
    def _generate_security_recommendations(self, pii_results: List[PIIDetectionResult], 
                                         security_score: float) -> List[str]:
        recommendations = []
        
        if pii_results:
            recommendations.append("Apply data masking or anonymization before processing")
            recommendations.append("Consider removing PII columns if not essential for analysis")
        
        if security_score < 0.7:
            recommendations.append("Implement additional data validation and cleaning")
            recommendations.append("Enable enhanced audit logging and monitoring")
        
        if self.security_level == SecurityLevel.BASIC:
            recommendations.append("Consider upgrading to higher security level for sensitive data")
        
        return recommendations