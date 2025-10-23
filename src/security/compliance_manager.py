import json
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np


class ComplianceRegulation(Enum):
    GDPR = "gdpr"
    CCPA = "ccpa"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    SOX = "sox"


class DataCategory(Enum):
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ProcessingPurpose(Enum):
    ANALYTICS = "analytics"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    RESEARCH = "research"
    LEGAL = "legal"


@dataclass
class DataSubject:
    subject_id: str
    data_categories: List[DataCategory]
    consent_status: Dict[ProcessingPurpose, bool]
    data_retention_date: Optional[datetime] = None
    anonymization_applied: bool = False
    deletion_requested: bool = False


@dataclass
class ProcessingRecord:
    record_id: str
    data_subject_ids: List[str]
    processing_purpose: ProcessingPurpose
    data_categories: List[DataCategory]
    legal_basis: str
    processing_start: datetime
    processing_end: Optional[datetime] = None
    data_processor: str = "system"
    security_measures: List[str] = field(default_factory=list)


@dataclass
class ComplianceViolation:
    violation_id: str
    regulation: ComplianceRegulation
    violation_type: str
    severity: str
    description: str
    affected_subjects: List[str]
    detection_time: datetime
    resolution_status: str = "open"


class ComplianceManager:
    def __init__(self, applicable_regulations: List[ComplianceRegulation] = None):
        self.regulations = applicable_regulations or [ComplianceRegulation.GDPR, ComplianceRegulation.CCPA]
        self.data_subjects = {}
        self.processing_records = {}
        self.violations = {}
        self.retention_policies = self._initialize_retention_policies()
        self.consent_expiry_days = 365

    def _initialize_retention_policies(self) -> Dict[DataCategory, int]:
        return {
            DataCategory.PERSONAL: 2555,  # 7 years
            DataCategory.SENSITIVE: 1095,  # 3 years
            DataCategory.PUBLIC: -1,  # No limit
            DataCategory.CONFIDENTIAL: 1825,  # 5 years
            DataCategory.RESTRICTED: 730,  # 2 years
        }

    def register_data_subject(self, subject_data: Dict[str, Any]) -> str:
        subject_id = str(uuid.uuid4())

        data_categories = self._classify_data_categories(subject_data)
        initial_consent = {purpose: False for purpose in ProcessingPurpose}

        retention_period = max(
            [
                self.retention_policies.get(category, 365)
                for category in data_categories
                if self.retention_policies.get(category, 365) > 0
            ],
            default=365,
        )

        retention_date = datetime.now() + timedelta(days=retention_period)

        self.data_subjects[subject_id] = DataSubject(
            subject_id=subject_id,
            data_categories=data_categories,
            consent_status=initial_consent,
            data_retention_date=retention_date,
        )

        return subject_id

    def _classify_data_categories(self, data: Dict[str, Any]) -> List[DataCategory]:
        categories = []

        personal_indicators = ["name", "email", "phone", "address", "id"]
        sensitive_indicators = ["ssn", "passport", "license", "medical", "biometric"]

        data_str = str(data).lower()

        if any(indicator in data_str for indicator in personal_indicators):
            categories.append(DataCategory.PERSONAL)

        if any(indicator in data_str for indicator in sensitive_indicators):
            categories.append(DataCategory.SENSITIVE)

        if not categories:
            categories.append(DataCategory.PUBLIC)

        return categories

    def update_consent(self, subject_id: str, purpose: ProcessingPurpose, consent_given: bool) -> bool:
        if subject_id not in self.data_subjects:
            return False

        self.data_subjects[subject_id].consent_status[purpose] = consent_given

        if consent_given:
            self._create_processing_record(subject_id, purpose)
        else:
            self._revoke_processing_consent(subject_id, purpose)

        return True

    def _create_processing_record(self, subject_id: str, purpose: ProcessingPurpose):
        record_id = str(uuid.uuid4())
        subject = self.data_subjects[subject_id]

        legal_basis = self._determine_legal_basis(purpose, subject.data_categories)
        security_measures = self._determine_security_measures(subject.data_categories)

        self.processing_records[record_id] = ProcessingRecord(
            record_id=record_id,
            data_subject_ids=[subject_id],
            processing_purpose=purpose,
            data_categories=subject.data_categories,
            legal_basis=legal_basis,
            processing_start=datetime.now(),
            security_measures=security_measures,
        )

    def _determine_legal_basis(self, purpose: ProcessingPurpose, categories: List[DataCategory]) -> str:
        if DataCategory.SENSITIVE in categories:
            return "explicit_consent"
        elif purpose in [ProcessingPurpose.ANALYTICS, ProcessingPurpose.RESEARCH]:
            return "legitimate_interest"
        elif purpose == ProcessingPurpose.LEGAL:
            return "legal_obligation"
        else:
            return "consent"

    def _determine_security_measures(self, categories: List[DataCategory]) -> List[str]:
        measures = ["encryption_at_rest", "access_logging"]

        if DataCategory.SENSITIVE in categories:
            measures.extend(["end_to_end_encryption", "multi_factor_auth"])

        if DataCategory.PERSONAL in categories:
            measures.append("pseudonymization")

        return measures

    def _revoke_processing_consent(self, subject_id: str, purpose: ProcessingPurpose):
        for record in self.processing_records.values():
            if (
                subject_id in record.data_subject_ids
                and record.processing_purpose == purpose
                and record.processing_end is None
            ):
                record.processing_end = datetime.now()

    def request_data_deletion(self, subject_id: str) -> Dict[str, Any]:
        if subject_id not in self.data_subjects:
            return {"success": False, "reason": "Subject not found"}

        subject = self.data_subjects[subject_id]

        can_delete = self._check_deletion_eligibility(subject_id)

        if can_delete["eligible"]:
            subject.deletion_requested = True
            deletion_timeline = self._schedule_deletion(subject_id)

            return {
                "success": True,
                "deletion_scheduled": deletion_timeline,
                "affected_records": can_delete["affected_records"],
            }
        else:
            return {
                "success": False,
                "reason": can_delete["reason"],
                "retention_until": can_delete.get("retention_until"),
            }

    def _check_deletion_eligibility(self, subject_id: str) -> Dict[str, Any]:
        active_records = [
            record
            for record in self.processing_records.values()
            if subject_id in record.data_subject_ids and record.processing_end is None
        ]

        legal_holds = [
            record for record in active_records if record.legal_basis in ["legal_obligation", "vital_interests"]
        ]

        if legal_holds:
            return {
                "eligible": False,
                "reason": "Data subject to legal hold",
                "affected_records": [r.record_id for r in legal_holds],
            }

        return {"eligible": True, "affected_records": [r.record_id for r in active_records]}

    def _schedule_deletion(self, subject_id: str) -> datetime:
        if ComplianceRegulation.GDPR in self.regulations:
            deletion_deadline = datetime.now() + timedelta(days=30)
        else:
            deletion_deadline = datetime.now() + timedelta(days=45)

        return deletion_deadline

    def export_subject_data(self, subject_id: str) -> Dict[str, Any]:
        if subject_id not in self.data_subjects:
            return {"error": "Subject not found"}

        subject = self.data_subjects[subject_id]
        related_records = [
            record for record in self.processing_records.values() if subject_id in record.data_subject_ids
        ]

        export_data = {
            "subject_information": {
                "subject_id": subject.subject_id,
                "data_categories": [cat.value for cat in subject.data_categories],
                "consent_status": {purpose.value: status for purpose, status in subject.consent_status.items()},
                "retention_date": subject.data_retention_date.isoformat() if subject.data_retention_date else None,
                "anonymization_applied": subject.anonymization_applied,
            },
            "processing_activities": [
                {
                    "record_id": record.record_id,
                    "purpose": record.processing_purpose.value,
                    "legal_basis": record.legal_basis,
                    "processing_period": {
                        "start": record.processing_start.isoformat(),
                        "end": record.processing_end.isoformat() if record.processing_end else None,
                    },
                    "security_measures": record.security_measures,
                }
                for record in related_records
            ],
            "export_timestamp": datetime.now().isoformat(),
        }

        return export_data

    def scan_compliance_violations(self, df: pd.DataFrame) -> List[ComplianceViolation]:
        violations = []

        violations.extend(self._check_consent_violations())
        violations.extend(self._check_retention_violations())
        violations.extend(self._check_data_minimization(df))
        violations.extend(self._check_purpose_limitation())

        for violation in violations:
            self.violations[violation.violation_id] = violation

        return violations

    def _check_consent_violations(self) -> List[ComplianceViolation]:
        violations = []

        for subject_id, subject in self.data_subjects.items():
            expired_consents = []

            for purpose, consent_given in subject.consent_status.items():
                if consent_given:
                    consent_age = datetime.now() - datetime.now()
                    if consent_age.days > self.consent_expiry_days:
                        expired_consents.append(purpose.value)

            if expired_consents:
                violations.append(
                    ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        regulation=ComplianceRegulation.GDPR,
                        violation_type="expired_consent",
                        severity="medium",
                        description=f"Expired consent for purposes: {', '.join(expired_consents)}",
                        affected_subjects=[subject_id],
                        detection_time=datetime.now(),
                    )
                )

        return violations

    def _check_retention_violations(self) -> List[ComplianceViolation]:
        violations = []

        for subject_id, subject in self.data_subjects.items():
            if (
                subject.data_retention_date
                and datetime.now() > subject.data_retention_date
                and not subject.deletion_requested
            ):
                violations.append(
                    ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        regulation=ComplianceRegulation.GDPR,
                        violation_type="retention_period_exceeded",
                        severity="high",
                        description=f"Data retention period exceeded for subject {subject_id}",
                        affected_subjects=[subject_id],
                        detection_time=datetime.now(),
                    )
                )

        return violations

    def _check_data_minimization(self, df: pd.DataFrame) -> List[ComplianceViolation]:
        violations = []

        if len(df.columns) > 50:
            unnecessary_columns = self._identify_unnecessary_columns(df)

            if unnecessary_columns:
                violations.append(
                    ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        regulation=ComplianceRegulation.GDPR,
                        violation_type="data_minimization",
                        severity="medium",
                        description=f"Potentially unnecessary data collection: {len(unnecessary_columns)} columns with low utility",
                        affected_subjects=["all"],
                        detection_time=datetime.now(),
                    )
                )

        return violations

    def _identify_unnecessary_columns(self, df: pd.DataFrame) -> List[str]:
        unnecessary = []

        for column in df.columns:
            if df[column].nunique() == 1:
                unnecessary.append(column)
            elif df[column].isnull().sum() / len(df) > 0.95:
                unnecessary.append(column)

        return unnecessary

    def _check_purpose_limitation(self) -> List[ComplianceViolation]:
        violations = []

        for record in self.processing_records.values():
            if record.processing_end is None:
                processing_duration = datetime.now() - record.processing_start

                if processing_duration.days > 90 and record.processing_purpose == ProcessingPurpose.MARKETING:
                    violations.append(
                        ComplianceViolation(
                            violation_id=str(uuid.uuid4()),
                            regulation=ComplianceRegulation.GDPR,
                            violation_type="purpose_limitation",
                            severity="medium",
                            description=f"Extended processing beyond typical purpose scope: {record.processing_purpose.value}",
                            affected_subjects=record.data_subject_ids,
                            detection_time=datetime.now(),
                        )
                    )

        return violations

    def generate_compliance_report(self) -> Dict[str, Any]:
        current_violations = list(self.violations.values())
        active_violations = [v for v in current_violations if v.resolution_status == "open"]

        consent_stats = self._calculate_consent_statistics()
        retention_stats = self._calculate_retention_statistics()

        return {
            "compliance_overview": {
                "applicable_regulations": [reg.value for reg in self.regulations],
                "total_data_subjects": len(self.data_subjects),
                "active_processing_records": len(
                    [r for r in self.processing_records.values() if r.processing_end is None]
                ),
                "total_violations": len(current_violations),
                "active_violations": len(active_violations),
            },
            "violation_summary": {
                "by_severity": self._group_violations_by_severity(active_violations),
                "by_regulation": self._group_violations_by_regulation(active_violations),
                "by_type": self._group_violations_by_type(active_violations),
            },
            "consent_management": consent_stats,
            "data_retention": retention_stats,
            "recommendations": self._generate_compliance_recommendations(active_violations),
            "report_timestamp": datetime.now().isoformat(),
        }

    def _calculate_consent_statistics(self) -> Dict[str, Any]:
        total_subjects = len(self.data_subjects)

        if total_subjects == 0:
            return {"consent_rate_by_purpose": {}, "overall_consent_rate": 0}

        consent_stats = {}
        for purpose in ProcessingPurpose:
            consented = sum(1 for subject in self.data_subjects.values() if subject.consent_status.get(purpose, False))
            consent_stats[purpose.value] = consented / total_subjects

        overall_consent_rate = np.mean(list(consent_stats.values()))

        return {
            "consent_rate_by_purpose": consent_stats,
            "overall_consent_rate": overall_consent_rate,
            "total_subjects": total_subjects,
        }

    def _calculate_retention_statistics(self) -> Dict[str, Any]:
        now = datetime.now()
        subjects_near_expiry = 0
        subjects_expired = 0

        for subject in self.data_subjects.values():
            if subject.data_retention_date:
                days_until_expiry = (subject.data_retention_date - now).days

                if days_until_expiry < 0:
                    subjects_expired += 1
                elif days_until_expiry < 30:
                    subjects_near_expiry += 1

        return {
            "subjects_near_expiry": subjects_near_expiry,
            "subjects_expired": subjects_expired,
            "deletion_requests_pending": sum(1 for s in self.data_subjects.values() if s.deletion_requested),
        }

    def _group_violations_by_severity(self, violations: List[ComplianceViolation]) -> Dict[str, int]:
        severity_count = {}
        for violation in violations:
            severity_count[violation.severity] = severity_count.get(violation.severity, 0) + 1
        return severity_count

    def _group_violations_by_regulation(self, violations: List[ComplianceViolation]) -> Dict[str, int]:
        regulation_count = {}
        for violation in violations:
            reg = violation.regulation.value
            regulation_count[reg] = regulation_count.get(reg, 0) + 1
        return regulation_count

    def _group_violations_by_type(self, violations: List[ComplianceViolation]) -> Dict[str, int]:
        type_count = {}
        for violation in violations:
            type_count[violation.violation_type] = type_count.get(violation.violation_type, 0) + 1
        return type_count

    def _generate_compliance_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        recommendations = []

        if any(v.violation_type == "retention_period_exceeded" for v in violations):
            recommendations.append("Implement automated data deletion for expired retention periods")

        if any(v.violation_type == "expired_consent" for v in violations):
            recommendations.append("Set up consent renewal notifications and automated consent expiry handling")

        if any(v.violation_type == "data_minimization" for v in violations):
            recommendations.append("Review data collection practices to ensure minimal necessary data collection")

        if any(v.severity == "high" for v in violations):
            recommendations.append("Address high-severity violations immediately to avoid regulatory penalties")

        if not recommendations:
            recommendations.append("Compliance status is good - maintain current practices and monitoring")

        return recommendations

    def process_regulatory_request(
        self, request_type: str, subject_id: str, additional_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if request_type == "access_request":
            return self.export_subject_data(subject_id)

        elif request_type == "deletion_request":
            return self.request_data_deletion(subject_id)

        elif request_type == "rectification_request":
            return self._process_rectification_request(subject_id, additional_params or {})

        elif request_type == "portability_request":
            return self._process_portability_request(subject_id)

        elif request_type == "restriction_request":
            return self._process_restriction_request(subject_id, additional_params or {})

        else:
            return {"error": f"Unknown request type: {request_type}"}

    def _process_rectification_request(self, subject_id: str, corrections: Dict[str, Any]) -> Dict[str, Any]:
        if subject_id not in self.data_subjects:
            return {"success": False, "reason": "Subject not found"}

        return {
            "success": True,
            "message": "Rectification request logged - manual data correction required",
            "corrections_requested": corrections,
            "processing_timeline": "30 days",
        }

    def _process_portability_request(self, subject_id: str) -> Dict[str, Any]:
        export_data = self.export_subject_data(subject_id)

        if "error" in export_data:
            return export_data

        return {
            "success": True,
            "portable_data": export_data,
            "format": "structured_json",
            "transfer_options": ["download", "api_access", "direct_transfer"],
        }

    def _process_restriction_request(self, subject_id: str, restrictions: Dict[str, Any]) -> Dict[str, Any]:
        if subject_id not in self.data_subjects:
            return {"success": False, "reason": "Subject not found"}

        restricted_purposes = restrictions.get("purposes", [])

        for purpose_str in restricted_purposes:
            try:
                purpose = ProcessingPurpose(purpose_str)
                self.update_consent(subject_id, purpose, False)
            except ValueError:
                continue

        return {"success": True, "restricted_purposes": restricted_purposes, "effective_immediately": True}
