"""Session persistence management."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentnet.persistence")


@dataclass
class SessionRecord:
    """Record of an agent session."""

    session_id: str
    topic_start: str
    topic_final: str
    topic_evolution: List[str]
    transcript: List[Dict[str, Any]]
    participants: List[str]
    rounds_executed: int
    metrics: Dict[str, Any]
    final_summary: Dict[str, Any]
    converged: bool
    mode: str
    strategy: str
    metadata: Dict[str, Any]
    timestamp: float
    parallel_round: bool = False

    # P3: Workflow mode support
    task_graph: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    task_results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        """Create from dictionary."""
        return cls(**data)


class SessionManager:
    """Manages session persistence operations."""

    def __init__(self, storage_dir: str = "sessions"):
        """Initialize session manager.

        Args:
            storage_dir: Directory for storing session files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        logger.info(f"SessionManager initialized with storage_dir: {self.storage_dir}")

    def persist_session(
        self, session_record: Dict[str, Any], agent_name: str = "unknown"
    ) -> str:
        """Persist a session record to storage.

        Args:
            session_record: Session data to persist
            agent_name: Name of the agent persisting the session

        Returns:
            Path to the persisted session file
        """
        session_id = session_record.get(
            "session_id", f"session_{int(time.time()*1000)}"
        )
        timestamp = session_record.get("timestamp", time.time())
        filename = f"{session_id}_{int(timestamp)}.json"
        filepath = self.storage_dir / filename

        # Add persistence metadata
        session_copy = dict(session_record)
        session_copy["persistence_metadata"] = {
            "saved_at": time.time(),
            "saved_by_agent": agent_name,
            "filepath": str(filepath),
            "version": "1.0",
        }

        # Write to file
        filepath.write_text(json.dumps(session_copy, indent=2))
        logger.info(f"Session '{session_id}' persisted to {filepath}")
        return str(filepath)

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a session by ID.

        Args:
            session_id: Session ID to load

        Returns:
            Session data or None if not found
        """
        # Find session file by ID
        for file_path in self.storage_dir.glob(f"{session_id}_*.json"):
            try:
                data = json.loads(file_path.read_text())
                logger.info(f"Loaded session '{session_id}' from {file_path}")
                return data
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load session from {file_path}: {e}")

        logger.warning(f"Session '{session_id}' not found")
        return None

    def list_sessions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List available sessions.

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session metadata
        """
        sessions = []
        session_files = sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if limit:
            session_files = session_files[:limit]

        for file_path in session_files:
            try:
                data = json.loads(file_path.read_text())
                metadata = {
                    "session_id": data.get("session_id"),
                    "timestamp": data.get("timestamp"),
                    "participants": data.get("participants", []),
                    "rounds_executed": data.get("rounds_executed", 0),
                    "converged": data.get("converged", False),
                    "mode": data.get("mode", "unknown"),
                    "filepath": str(file_path),
                }
                sessions.append(metadata)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to read session file {file_path}: {e}")

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        for file_path in self.storage_dir.glob(f"{session_id}_*.json"):
            try:
                file_path.unlink()
                logger.info(f"Deleted session '{session_id}' at {file_path}")
                return True
            except IOError as e:
                logger.error(f"Failed to delete session file {file_path}: {e}")

        logger.warning(f"Session '{session_id}' not found for deletion")
        return False

    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Clean up old session files.

        Args:
            max_age_days: Maximum age in days to keep sessions

        Returns:
            Number of sessions deleted
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        deleted_count = 0

        for file_path in self.storage_dir.glob("*.json"):
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Cleaned up old session: {file_path}")
            except IOError as e:
                logger.error(f"Failed to clean up session file {file_path}: {e}")

        logger.info(f"Cleaned up {deleted_count} old sessions")
        return deleted_count

    def create_checkpoint(
        self, session_id: str, checkpoint_data: Dict[str, Any]
    ) -> str:
        """Create a checkpoint for session state.

        Args:
            session_id: Session ID to checkpoint
            checkpoint_data: Current session state to save

        Returns:
            Checkpoint identifier
        """
        checkpoint_id = f"{session_id}_checkpoint_{int(time.time() * 1000)}"
        checkpoint_dir = self.storage_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_file = checkpoint_dir / f"{checkpoint_id}.json"

        checkpoint_record = {
            "checkpoint_id": checkpoint_id,
            "session_id": session_id,
            "timestamp": time.time(),
            "data": checkpoint_data,
            "version": "1.0",
        }

        checkpoint_file.write_text(json.dumps(checkpoint_record, indent=2))
        logger.info(f"Created checkpoint '{checkpoint_id}' for session '{session_id}'")
        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Load a session checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to load

        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_dir = self.storage_dir / "checkpoints"
        checkpoint_file = checkpoint_dir / f"{checkpoint_id}.json"

        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint '{checkpoint_id}' not found")
            return None

        try:
            data = json.loads(checkpoint_file.read_text())
            logger.info(f"Loaded checkpoint '{checkpoint_id}'")
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load checkpoint '{checkpoint_id}': {e}")
            return None

    def resume_session(
        self, checkpoint_id: str, additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Resume a session from checkpoint.

        Args:
            checkpoint_id: Checkpoint to resume from
            additional_context: Additional context to merge with checkpoint

        Returns:
            Merged session state ready for resumption
        """
        checkpoint = self.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return None

        session_data = checkpoint["data"]

        # Merge additional context if provided
        if additional_context:
            session_data.update(additional_context)

        # Mark as resumed
        session_data["resumed_from_checkpoint"] = checkpoint_id
        session_data["resume_timestamp"] = time.time()

        logger.info(f"Session resumed from checkpoint '{checkpoint_id}'")
        return session_data

    def list_checkpoints(
        self, session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available checkpoints.

        Args:
            session_id: Filter by session ID (optional)

        Returns:
            List of checkpoint metadata
        """
        checkpoint_dir = self.storage_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        checkpoints = []
        pattern = f"{session_id}_checkpoint_*.json" if session_id else "*.json"

        for checkpoint_file in sorted(
            checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                data = json.loads(checkpoint_file.read_text())
                metadata = {
                    "checkpoint_id": data.get("checkpoint_id"),
                    "session_id": data.get("session_id"),
                    "timestamp": data.get("timestamp"),
                    "filepath": str(checkpoint_file),
                }
                checkpoints.append(metadata)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to read checkpoint file {checkpoint_file}: {e}")

        return checkpoints
