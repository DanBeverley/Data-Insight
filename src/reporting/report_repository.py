"""Database operations for reports"""

import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from src.database.models import Report, ReportArtifact


class ReportRepository:
    """Handles all database operations for reports"""

    def __init__(self, db: Session):
        self.db = db

    def create_report(self, session_id: str, dataset_name: str, status: str = "generating") -> Report:
        """
        Create a new report entry.

        Args:
            session_id: User session identifier
            dataset_name: Name of the dataset
            status: Initial status (default: "generating")

        Returns:
            Created Report instance
        """
        report = Report(
            id=str(uuid.uuid4()),
            session_id=session_id,
            dataset_name=dataset_name,
            status=status,
            report_data={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self.db.add(report)
        self.db.commit()
        self.db.refresh(report)

        return report

    def get_report(self, report_id: str) -> Optional[Report]:
        """Get report by ID"""
        return self.db.query(Report).filter(Report.id == report_id).first()

    def get_reports_by_session(self, session_id: str) -> List[Report]:
        """Get all reports for a session"""
        return self.db.query(Report).filter(Report.session_id == session_id).order_by(Report.created_at.desc()).all()

    def update_report_status(self, report_id: str, status: str) -> bool:
        """
        Update report status.

        Args:
            report_id: Report identifier
            status: New status ('generating', 'completed', 'failed')

        Returns:
            True if successful, False otherwise
        """
        report = self.get_report(report_id)
        if not report:
            return False

        report.status = status
        report.updated_at = datetime.utcnow()
        self.db.commit()

        return True

    def update_report_data(self, report_id: str, section: str, content: str) -> bool:
        """
        Update report data with new section content.

        Args:
            report_id: Report identifier
            section: Section name
            content: HTML content for section

        Returns:
            True if successful
        """
        report = self.get_report(report_id)
        if not report:
            return False

        if not report.report_data:
            report.report_data = {}

        report.report_data[section] = content
        report.updated_at = datetime.utcnow()
        self.db.commit()

        return True

    def add_artifact(
        self,
        report_id: str,
        artifact_type: str,
        file_path: str,
        file_size_bytes: Optional[int] = None,
        artifact_metadata: Optional[Dict[str, Any]] = None,
    ) -> ReportArtifact:
        """
        Add artifact to report.

        Args:
            report_id: Report identifier
            artifact_type: Type of artifact ('model', 'plot', 'csv', 'script')
            file_path: R2 storage path
            file_size_bytes: Size of file in bytes
            artifact_metadata: Additional metadata

        Returns:
            Created ReportArtifact instance
        """
        artifact = ReportArtifact(
            id=str(uuid.uuid4()),
            report_id=report_id,
            artifact_type=artifact_type,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            artifact_metadata=artifact_metadata or {},
            created_at=datetime.utcnow(),
        )

        self.db.add(artifact)
        self.db.commit()
        self.db.refresh(artifact)

        return artifact

    def get_artifacts(self, report_id: str) -> List[ReportArtifact]:
        """Get all artifacts for a report"""
        return (
            self.db.query(ReportArtifact)
            .filter(ReportArtifact.report_id == report_id)
            .order_by(ReportArtifact.created_at.asc())
            .all()
        )

    def delete_report(self, report_id: str) -> bool:
        """
        Delete report and all associated artifacts.

        Args:
            report_id: Report identifier

        Returns:
            True if successful
        """
        report = self.get_report(report_id)
        if not report:
            return False

        self.db.delete(report)
        self.db.commit()

        return True
