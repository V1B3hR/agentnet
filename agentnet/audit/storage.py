"""Audit Storage Implementation for P6 Enterprise Hardening.

This module provides persistent audit event storage with querying capabilities
for compliance reporting and forensic analysis.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .workflow import AuditEvent, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)


@dataclass
class AuditQuery:
    """Query parameters for audit event retrieval."""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    severity_levels: Optional[List[AuditSeverity]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    compliance_tags: Optional[List[str]] = None
    limit: int = 1000
    offset: int = 0


class AuditStorage:
    """Persistent storage for audit events with querying capabilities."""

    def __init__(self, storage_path: str = "./audit_storage.db"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for audit storage."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    agent_id TEXT,
                    resource_id TEXT,
                    action TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    compliance_tags TEXT,
                    retention_class TEXT DEFAULT 'standard',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indices for common queries
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_id ON audit_events(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_id ON audit_events(agent_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)"
            )

            conn.commit()

    def store_event(self, event: AuditEvent) -> None:
        """Store an audit event."""
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO audit_events (
                    event_id, timestamp, event_type, severity, user_id, session_id,
                    agent_id, resource_id, action, details, ip_address, user_agent,
                    compliance_tags, retention_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.event_type.value,
                    event.severity.value,
                    event.user_id,
                    event.session_id,
                    event.agent_id,
                    event.resource_id,
                    event.action,
                    json.dumps(event.details),
                    event.ip_address,
                    event.user_agent,
                    json.dumps(event.compliance_tags),
                    event.retention_class,
                ),
            )
            conn.commit()

    def get_events(
        self, query: Optional[AuditQuery] = None, limit: int = 100
    ) -> List[AuditEvent]:
        """Retrieve audit events based on query parameters."""
        if query is None:
            query = AuditQuery(limit=limit)

        sql = "SELECT * FROM audit_events WHERE 1=1"
        params = []

        # Build WHERE clause
        if query.start_time:
            sql += " AND timestamp >= ?"
            params.append(query.start_time.isoformat())

        if query.end_time:
            sql += " AND timestamp <= ?"
            params.append(query.end_time.isoformat())

        if query.event_types:
            placeholders = ",".join("?" * len(query.event_types))
            sql += f" AND event_type IN ({placeholders})"
            params.extend([et.value for et in query.event_types])

        if query.severity_levels:
            placeholders = ",".join("?" * len(query.severity_levels))
            sql += f" AND severity IN ({placeholders})"
            params.extend([sl.value for sl in query.severity_levels])

        if query.user_id:
            sql += " AND user_id = ?"
            params.append(query.user_id)

        if query.session_id:
            sql += " AND session_id = ?"
            params.append(query.session_id)

        if query.agent_id:
            sql += " AND agent_id = ?"
            params.append(query.agent_id)

        if query.compliance_tags:
            for tag in query.compliance_tags:
                sql += " AND compliance_tags LIKE ?"
                params.append(f"%{tag}%")

        sql += " ORDER BY timestamp DESC"

        if query.limit:
            sql += " LIMIT ?"
            params.append(query.limit)

        if query.offset:
            sql += " OFFSET ?"
            params.append(query.offset)

        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        events = []
        for row in rows:
            event_data = dict(row)
            event_data["details"] = json.loads(event_data["details"] or "{}")
            event_data["compliance_tags"] = json.loads(
                event_data["compliance_tags"] or "[]"
            )
            events.append(AuditEvent.from_dict(event_data))

        return events

    def get_event_statistics(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get audit event statistics for a time period."""
        if not start_time:
            start_time = datetime.utcnow() - timedelta(days=7)
        if not end_time:
            end_time = datetime.utcnow()

        with sqlite3.connect(self.storage_path) as conn:
            # Total events in period
            cursor = conn.execute(
                "SELECT COUNT(*) FROM audit_events WHERE timestamp BETWEEN ? AND ?",
                (start_time.isoformat(), end_time.isoformat()),
            )
            total_events = cursor.fetchone()[0]

            # Events by type
            cursor = conn.execute(
                """
                SELECT event_type, COUNT(*) as count 
                FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? 
                GROUP BY event_type
            """,
                (start_time.isoformat(), end_time.isoformat()),
            )
            events_by_type = {row[0]: row[1] for row in cursor.fetchall()}

            # Events by severity
            cursor = conn.execute(
                """
                SELECT severity, COUNT(*) as count 
                FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? 
                GROUP BY severity
            """,
                (start_time.isoformat(), end_time.isoformat()),
            )
            events_by_severity = {row[0]: row[1] for row in cursor.fetchall()}

            # Top users by activity
            cursor = conn.execute(
                """
                SELECT user_id, COUNT(*) as count 
                FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? AND user_id IS NOT NULL
                GROUP BY user_id 
                ORDER BY count DESC 
                LIMIT 10
            """,
                (start_time.isoformat(), end_time.isoformat()),
            )
            top_users = {row[0]: row[1] for row in cursor.fetchall()}

            # Compliance tag frequency
            cursor = conn.execute(
                """
                SELECT compliance_tags 
                FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? AND compliance_tags != '[]'
            """,
                (start_time.isoformat(), end_time.isoformat()),
            )

            tag_counts = {}
            for row in cursor.fetchall():
                tags = json.loads(row[0])
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "total_events": total_events,
            "events_by_type": events_by_type,
            "events_by_severity": events_by_severity,
            "top_users": top_users,
            "compliance_tags": tag_counts,
            "time_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
        }

    def cleanup_expired_events(self, retention_days: int = 2555) -> int:
        """Remove expired audit events based on retention policy."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute(
                "DELETE FROM audit_events WHERE timestamp < ? AND retention_class != 'permanent'",
                (cutoff_date.isoformat(),),
            )
            deleted_count = cursor.rowcount
            conn.commit()

        logger.info(f"Cleaned up {deleted_count} expired audit events")
        return deleted_count

    def export_events(
        self, filepath: str, query: Optional[AuditQuery] = None, format: str = "json"
    ) -> None:
        """Export audit events to file."""
        events = self.get_events(query)

        if format == "json":
            export_data = {
                "exported_at": datetime.utcnow().isoformat(),
                "event_count": len(events),
                "events": [event.to_dict() for event in events],
            }

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2)

        elif format == "csv":
            import csv

            with open(filepath, "w", newline="") as f:
                if events:
                    writer = csv.DictWriter(f, fieldnames=events[0].to_dict().keys())
                    writer.writeheader()
                    for event in events:
                        row = event.to_dict()
                        # Flatten complex fields for CSV
                        row["details"] = json.dumps(row["details"])
                        row["compliance_tags"] = json.dumps(row["compliance_tags"])
                        writer.writerow(row)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported {len(events)} audit events to {filepath}")

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM audit_events")
            total_events = cursor.fetchone()[0]

            cursor = conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM audit_events"
            )
            time_range = cursor.fetchone()

            # Get database file size
            db_size = (
                self.storage_path.stat().st_size if self.storage_path.exists() else 0
            )

        return {
            "database_path": str(self.storage_path),
            "total_events": total_events,
            "earliest_event": time_range[0],
            "latest_event": time_range[1],
            "database_size_bytes": db_size,
            "database_size_mb": round(db_size / 1024 / 1024, 2),
        }
