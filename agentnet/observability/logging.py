"""
Structured Logging for AgentNet.

Implements JSON structured logging with correlation_id = session_id as specified
in docs/RoadmapAgentNet.md section 18.

Provides correlation-aware logging that can be easily parsed and analyzed.
"""

import json
import logging
import logging.config
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Context variable for correlation ID tracking across async operations
correlation_id_context: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)
session_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "session_context", default={}
)


class CorrelationFormatter(logging.Formatter):
    """
    JSON formatter that includes correlation_id and session context in all log messages.

    Automatically adds correlation metadata to structured log output.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with correlation data."""
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available
        correlation_id = correlation_id_context.get()
        if correlation_id:
            log_data["correlation_id"] = correlation_id
            log_data["session_id"] = (
                correlation_id  # session_id is the primary correlation ID
            )

        # Add session context if available
        session_ctx = session_context.get()
        if session_ctx:
            log_data["session_context"] = session_ctx

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": (
                    self.formatException(record.exc_info) if record.exc_info else None
                ),
            }

        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "exc_info",
                "exc_text",
                "stack_info",
                "message",
            }:
                extra_fields[key] = value

        if extra_fields:
            log_data["extra"] = extra_fields

        return json.dumps(log_data, default=str, ensure_ascii=False)


class CorrelationFilter(logging.Filter):
    """
    Logging filter that can optionally filter messages based on correlation context.

    Useful for debugging specific sessions or filtering out noise.
    """

    def __init__(self, required_correlation_id: Optional[str] = None):
        """
        Initialize filter.

        Args:
            required_correlation_id: If set, only log messages with this correlation ID
        """
        super().__init__()
        self.required_correlation_id = required_correlation_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on correlation context."""
        if self.required_correlation_id is None:
            return True

        current_correlation_id = correlation_id_context.get()
        return current_correlation_id == self.required_correlation_id


class CorrelationLogger:
    """
    Wrapper around Python logger that automatically includes correlation context.

    Provides convenient methods for logging with agent-specific context.
    """

    def __init__(self, logger_name: str):
        """
        Initialize correlation logger.

        Args:
            logger_name: Name of the underlying logger
        """
        self.logger = logging.getLogger(logger_name)
        self._session_id: Optional[str] = None
        self._agent_name: Optional[str] = None
        self._operation: Optional[str] = None

    def set_correlation_context(
        self,
        session_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        operation: Optional[str] = None,
        **extra_context,
    ):
        """
        Set correlation context for this logger instance.

        Args:
            session_id: Session ID for correlation
            agent_name: Current agent name
            operation: Current operation name
            **extra_context: Additional context fields
        """
        self._session_id = session_id
        self._agent_name = agent_name
        self._operation = operation

        # Update context variables
        if session_id:
            correlation_id_context.set(session_id)

        # Build session context
        ctx = {}
        if agent_name:
            ctx["agent_name"] = agent_name
        if operation:
            ctx["operation"] = operation
        if extra_context:
            ctx.update(extra_context)

        if ctx:
            session_context.set(ctx)

    def clear_context(self):
        """Clear correlation context."""
        self._session_id = None
        self._agent_name = None
        self._operation = None
        correlation_id_context.set(None)
        session_context.set({})

    def _log_with_context(self, level: int, message: str, **extra):
        """Log message with current correlation context."""
        # Add correlation info to extra fields
        if self._session_id:
            extra["session_id"] = self._session_id
        if self._agent_name:
            extra["agent_name"] = self._agent_name
        if self._operation:
            extra["operation"] = self._operation

        self.logger.log(level, message, extra=extra)

    def debug(self, message: str, **extra):
        """Log debug message with correlation context."""
        self._log_with_context(logging.DEBUG, message, **extra)

    def info(self, message: str, **extra):
        """Log info message with correlation context."""
        self._log_with_context(logging.INFO, message, **extra)

    def warning(self, message: str, **extra):
        """Log warning message with correlation context."""
        self._log_with_context(logging.WARNING, message, **extra)

    def error(self, message: str, **extra):
        """Log error message with correlation context."""
        self._log_with_context(logging.ERROR, message, **extra)

    def critical(self, message: str, **extra):
        """Log critical message with correlation context."""
        self._log_with_context(logging.CRITICAL, message, **extra)

    def log_agent_inference(
        self, model: str, provider: str, token_count: int, duration_ms: float, **extra
    ):
        """Log agent inference operation with structured data."""
        self.info(
            f"Agent inference completed: {model} on {provider}",
            event_type="agent_inference",
            model=model,
            provider=provider,
            token_count=token_count,
            duration_ms=duration_ms,
            **extra,
        )

    def log_tool_invocation(
        self, tool_name: str, status: str, duration_ms: float, **extra
    ):
        """Log tool invocation with structured data."""
        self.info(
            f"Tool invocation: {tool_name} -> {status}",
            event_type="tool_invocation",
            tool_name=tool_name,
            status=status,
            duration_ms=duration_ms,
            **extra,
        )

    def log_violation(self, rule_name: str, severity: str, details: str, **extra):
        """Log policy violation with structured data."""
        self.warning(
            f"Policy violation: {rule_name} ({severity})",
            event_type="policy_violation",
            rule_name=rule_name,
            severity=severity,
            details=details,
            **extra,
        )

    def log_cost_event(
        self, provider: str, model: str, cost_usd: float, tokens: int, **extra
    ):
        """Log cost tracking event with structured data."""
        self.info(
            f"Cost event: ${cost_usd:.4f} for {tokens} tokens on {model}",
            event_type="cost_tracking",
            provider=provider,
            model=model,
            cost_usd=cost_usd,
            tokens=tokens,
            **extra,
        )

    def log_session_event(self, event_type: str, round_number: int, mode: str, **extra):
        """Log session lifecycle event with structured data."""
        self.info(
            f"Session {event_type}: round {round_number} in {mode} mode",
            event_type=f"session_{event_type}",
            round_number=round_number,
            mode=mode,
            **extra,
        )


def setup_structured_logging(
    log_level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    json_format: bool = True,
    correlation_filter: Optional[str] = None,
) -> None:
    """
    Setup structured logging configuration for AgentNet.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        console_output: Whether to output logs to console
        json_format: Whether to use JSON formatting
        correlation_filter: Optional correlation ID to filter logs
    """
    # Convert string log level to int
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    # Create formatters
    if json_format:
        formatter = CorrelationFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handlers = []

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)

        if correlation_filter:
            console_handler.addFilter(CorrelationFilter(correlation_filter))

        handlers.append(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)

        if correlation_filter:
            file_handler.addFilter(CorrelationFilter(correlation_filter))

        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Set specific logger levels for AgentNet components
    agentnet_loggers = [
        "agentnet",
        "agentnet.core",
        "agentnet.providers",
        "agentnet.monitors",
        "agentnet.tools",
        "agentnet.memory",
        "agentnet.observability",
    ]

    for logger_name in agentnet_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

    logging.info(
        f"Structured logging setup complete - level: {logging.getLevelName(log_level)}"
    )


def get_correlation_logger(name: str) -> CorrelationLogger:
    """
    Get a correlation-aware logger instance.

    Args:
        name: Logger name (typically module name)

    Returns:
        CorrelationLogger instance
    """
    return CorrelationLogger(name)


def set_correlation_id(correlation_id: str):
    """Set correlation ID in current context."""
    correlation_id_context.set(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return correlation_id_context.get()


def set_session_context(**context):
    """Set session context in current context."""
    current_context = session_context.get() or {}
    current_context.update(context)
    session_context.set(current_context)


def get_session_context() -> Dict[str, Any]:
    """Get current session context."""
    return session_context.get() or {}


def clear_correlation_context():
    """Clear all correlation context."""
    correlation_id_context.set(None)
    session_context.set({})


# Convenience logger instances
agentnet_logger = get_correlation_logger("agentnet")
core_logger = get_correlation_logger("agentnet.core")
provider_logger = get_correlation_logger("agentnet.providers")
monitor_logger = get_correlation_logger("agentnet.monitors")
tool_logger = get_correlation_logger("agentnet.tools")
observability_logger = get_correlation_logger("agentnet.observability")
