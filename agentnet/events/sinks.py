"""
Event sinks for processing and outputting events.

Provides console, file, and other output sinks for the event bus.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from .bus import Event

logger = logging.getLogger("agentnet.events.sinks")


class EventSink(ABC):
    """Abstract base class for event sinks."""
    
    @abstractmethod
    def process_event(self, event: Event) -> None:
        """Process a single event."""
        pass
    
    def close(self) -> None:
        """Close the sink and cleanup resources."""
        pass


class ConsoleSink(EventSink):
    """Event sink that outputs events to console/stdout."""
    
    def __init__(
        self,
        format_template: str = "[{datetime}] {event_type} | {source} | {data}",
        include_metadata: bool = False,
        log_level: str = "INFO"
    ):
        """
        Initialize console sink.
        
        Args:
            format_template: Template for formatting events
            include_metadata: Whether to include metadata in output
            log_level: Logging level for output
        """
        self.format_template = format_template
        self.include_metadata = include_metadata
        self.log_level = getattr(logging, log_level.upper())
        
        self.events_processed = 0
        self.logger = logging.getLogger("agentnet.events.console")
    
    def process_event(self, event: Event) -> None:
        """Process event by printing to console."""
        try:
            # Format the event
            formatted = self._format_event(event)
            
            # Log at appropriate level
            if event.event_type.value in ["error"]:
                self.logger.error(formatted)
            elif event.event_type.value in ["warning"]:
                self.logger.warning(formatted)
            else:
                self.logger.log(self.log_level, formatted)
            
            self.events_processed += 1
            
        except Exception as e:
            logger.error(f"Error in ConsoleSink: {e}")
    
    def _format_event(self, event: Event) -> str:
        """Format an event for console output."""
        # Prepare data for formatting
        format_data = {
            "datetime": event.datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": event.event_type.value,
            "source": event.source,
            "event_id": event.event_id,
            "data": self._format_data(event.data),
            "metadata": self._format_data(event.metadata) if self.include_metadata else ""
        }
        
        try:
            return self.format_template.format(**format_data)
        except KeyError as e:
            # Fallback format if template has invalid keys
            return f"[{format_data['datetime']}] {format_data['event_type']} | {format_data['source']} | {format_data['data']}"
    
    def _format_data(self, data: Dict[str, Any]) -> str:
        """Format data dictionary for display."""
        if not data:
            return ""
        
        # For simple data, create a compact representation
        if len(data) == 1:
            key, value = next(iter(data.items()))
            return f"{key}={value}"
        
        # For complex data, use key=value pairs
        pairs = []
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                pairs.append(f"{key}={value}")
            else:
                pairs.append(f"{key}=<{type(value).__name__}>")
        
        return " ".join(pairs[:3])  # Limit to first 3 items


class FileSink(EventSink):
    """Event sink that writes events to a file."""
    
    def __init__(
        self,
        file_path: str,
        format_type: str = "json",
        rotation_size: Optional[int] = None,
        max_files: int = 5,
        append: bool = True
    ):
        """
        Initialize file sink.
        
        Args:
            file_path: Path to output file
            format_type: Output format ("json", "text", "csv")
            rotation_size: Rotate file when it exceeds this size (bytes)
            max_files: Maximum number of rotated files to keep
            append: Whether to append to existing file
        """
        self.file_path = Path(file_path)
        self.format_type = format_type.lower()
        self.rotation_size = rotation_size
        self.max_files = max_files
        self.append = append
        
        self.events_processed = 0
        self.bytes_written = 0
        self.current_file: Optional[TextIO] = None
        
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open initial file
        self._open_file()
        
        logger.info(f"FileSink initialized: {self.file_path} ({self.format_type})")
    
    def process_event(self, event: Event) -> None:
        """Process event by writing to file."""
        try:
            # Check if rotation is needed
            if self.rotation_size and self.bytes_written > self.rotation_size:
                self._rotate_file()
            
            # Format and write event
            formatted = self._format_event(event)
            
            if self.current_file:
                self.current_file.write(formatted + "\n")
                self.current_file.flush()
                
                self.events_processed += 1
                self.bytes_written += len(formatted) + 1
            
        except Exception as e:
            logger.error(f"Error in FileSink: {e}")
    
    def _format_event(self, event: Event) -> str:
        """Format event based on format type."""
        if self.format_type == "json":
            return json.dumps(event.to_dict(), separators=(',', ':'))
        
        elif self.format_type == "csv":
            # Simple CSV format
            return f"{event.timestamp},{event.event_type.value},{event.source},{json.dumps(event.data)}"
        
        else:  # text format
            return f"[{event.datetime.isoformat()}] {event.event_type.value} | {event.source} | {event.data}"
    
    def _open_file(self) -> None:
        """Open the output file."""
        mode = "a" if self.append else "w"
        
        try:
            self.current_file = open(self.file_path, mode, encoding="utf-8")
            self.bytes_written = self.file_path.stat().st_size if self.append else 0
        except Exception as e:
            logger.error(f"Failed to open file {self.file_path}: {e}")
            self.current_file = None
    
    def _rotate_file(self) -> None:
        """Rotate the current file."""
        if not self.current_file:
            return
        
        # Close current file
        self.current_file.close()
        
        # Rotate existing files
        for i in range(self.max_files - 1, 0, -1):
            old_file = self.file_path.with_suffix(f".{i}{self.file_path.suffix}")
            new_file = self.file_path.with_suffix(f".{i+1}{self.file_path.suffix}")
            
            if old_file.exists():
                if new_file.exists():
                    new_file.unlink()
                old_file.rename(new_file)
        
        # Move current file to .1
        rotated_file = self.file_path.with_suffix(f".1{self.file_path.suffix}")
        if rotated_file.exists():
            rotated_file.unlink()
        self.file_path.rename(rotated_file)
        
        # Open new file
        self._open_file()
        
        logger.info(f"Rotated log file: {self.file_path}")
    
    def close(self) -> None:
        """Close the file sink."""
        if self.current_file:
            self.current_file.close()
            self.current_file = None
        
        logger.info(f"FileSink closed: {self.file_path}")


class StructuredSink(EventSink):
    """Event sink that processes events into structured data."""
    
    def __init__(
        self,
        storage_backend: str = "memory",
        max_events: int = 10000,
        **kwargs
    ):
        """
        Initialize structured sink.
        
        Args:
            storage_backend: Storage backend ("memory", "sqlite", etc.)
            max_events: Maximum events to store
            **kwargs: Additional backend-specific options
        """
        self.storage_backend = storage_backend
        self.max_events = max_events
        self.kwargs = kwargs
        
        self.events_processed = 0
        self.stored_events: List[Event] = []
        
        if storage_backend == "sqlite":
            self._init_sqlite()
        
        logger.info(f"StructuredSink initialized: {storage_backend}")
    
    def process_event(self, event: Event) -> None:
        """Process event by storing in structured format."""
        try:
            if self.storage_backend == "memory":
                self._store_memory(event)
            elif self.storage_backend == "sqlite":
                self._store_sqlite(event)
            
            self.events_processed += 1
            
        except Exception as e:
            logger.error(f"Error in StructuredSink: {e}")
    
    def _store_memory(self, event: Event) -> None:
        """Store event in memory."""
        self.stored_events.append(event)
        
        # Trim if over limit
        if len(self.stored_events) > self.max_events:
            self.stored_events.pop(0)
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite storage."""
        try:
            import sqlite3
            
            db_path = self.kwargs.get("db_path", "agentnet_events.db")
            self.conn = sqlite3.connect(db_path)
            
            # Create events table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    timestamp REAL,
                    source TEXT,
                    data TEXT,
                    metadata TEXT
                )
            """)
            self.conn.commit()
            
        except ImportError:
            logger.error("SQLite not available, falling back to memory storage")
            self.storage_backend = "memory"
        except Exception as e:
            logger.error(f"Failed to initialize SQLite: {e}")
            self.storage_backend = "memory"
    
    def _store_sqlite(self, event: Event) -> None:
        """Store event in SQLite."""
        if not hasattr(self, 'conn'):
            return
        
        try:
            self.conn.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.event_type.value,
                    event.timestamp,
                    event.source,
                    json.dumps(event.data),
                    json.dumps(event.metadata)
                )
            )
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing event in SQLite: {e}")
    
    def query_events(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """Query stored events."""
        if self.storage_backend == "memory":
            events = self.stored_events
            
            if event_type:
                events = [e for e in events if e.event_type.value == event_type]
            
            if source:
                events = [e for e in events if e.source == source]
            
            return events[-limit:]
        
        elif self.storage_backend == "sqlite" and hasattr(self, 'conn'):
            # SQLite query implementation would go here
            pass
        
        return []
    
    def close(self) -> None:
        """Close the structured sink."""
        if hasattr(self, 'conn'):
            self.conn.close()
        
        logger.info("StructuredSink closed")