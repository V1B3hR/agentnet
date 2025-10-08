"""
Partial JSON Parser for Streaming Responses

Implements robust parsing of partial JSON streams with error recovery
and incremental parsing capabilities. Handles incomplete JSON objects
and arrays during streaming operations.
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
import logging

logger = logging.getLogger(__name__)


class ParseState(str, Enum):
    """State of the partial JSON parser."""

    INIT = "init"
    IN_OBJECT = "in_object"
    IN_ARRAY = "in_array"
    IN_STRING = "in_string"
    IN_VALUE = "in_value"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ParseResult:
    """Result of partial JSON parsing."""

    parsed_data: Optional[Dict[str, Any]] = None
    partial_data: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False
    is_valid: bool = True

    # Parser state
    state: ParseState = ParseState.INIT
    bytes_processed: int = 0
    total_bytes: int = 0

    # Error information
    error: Optional[str] = None
    error_position: Optional[int] = None

    # Streaming metadata
    chunk_count: int = 0
    last_update: float = 0.0

    @property
    def progress(self) -> float:
        """Progress as percentage of total bytes processed."""
        if self.total_bytes == 0:
            return 0.0
        return min(100.0, (self.bytes_processed / self.total_bytes) * 100.0)


class PartialJSONParser:
    """
    Parser for incomplete JSON streams with graceful error handling.

    Provides incremental parsing of JSON data as it arrives,
    handling incomplete objects, arrays, and strings gracefully.
    """

    def __init__(self, max_depth: int = 32, strict_mode: bool = False):
        self.max_depth = max_depth
        self.strict_mode = strict_mode
        self.reset()

    def reset(self) -> None:
        """Reset parser state for new stream."""
        self._buffer = ""
        self._state = ParseState.INIT
        self._depth = 0
        self._in_string = False
        self._escape_next = False
        self._current_key = None
        self._stack = []
        self._result = ParseResult()

    def feed(self, chunk: str) -> ParseResult:
        """
        Feed a chunk of data to the parser.

        Args:
            chunk: String chunk of JSON data

        Returns:
            ParseResult with current parsing state and extracted data
        """
        if not chunk:
            return self._result

        self._buffer += chunk
        self._result.bytes_processed += len(chunk.encode("utf-8"))
        self._result.chunk_count += 1

        try:
            self._parse_buffer()
        except Exception as e:
            self._handle_parse_error(str(e))

        return self._result.copy() if hasattr(self._result, "copy") else self._result

    def _parse_buffer(self) -> None:
        """Parse the current buffer content."""

        # Try complete JSON parse first
        if self._try_complete_parse():
            return

        # Fall back to incremental parsing
        self._incremental_parse()

    def _try_complete_parse(self) -> bool:
        """Try to parse buffer as complete JSON."""

        try:
            # Remove common incomplete patterns
            cleaned = self._clean_partial_json(self._buffer)
            if cleaned:
                parsed = json.loads(cleaned)
                self._result.parsed_data = parsed
                self._result.is_complete = True
                self._result.is_valid = True
                self._result.state = ParseState.COMPLETE
                return True
        except json.JSONDecodeError:
            pass

        return False

    def _clean_partial_json(self, text: str) -> Optional[str]:
        """Clean partial JSON to make it parseable."""

        if not text.strip():
            return None

        # Remove trailing incomplete tokens
        text = text.strip()

        # Handle incomplete strings
        if text.count('"') % 2 == 1:
            # Find last unescaped quote
            last_quote = -1
            for i in range(len(text) - 1, -1, -1):
                if text[i] == '"' and (i == 0 or text[i - 1] != "\\"):
                    last_quote = i
                    break

            if last_quote > 0:
                text = text[:last_quote] + '"'

        # Handle incomplete objects/arrays
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        # Close incomplete structures
        text += "}" * open_braces
        text += "]" * open_brackets

        # Remove trailing commas
        text = re.sub(r",(\s*[}\]])", r"\1", text)

        return text if text.strip() else None

    def _incremental_parse(self) -> None:
        """Perform incremental parsing to extract partial data."""

        partial_data = {}
        current_buffer = self._buffer.strip()

        if not current_buffer:
            return

        # Extract key-value pairs using regex
        self._extract_partial_kvps(current_buffer, partial_data)

        # Extract array elements
        self._extract_partial_arrays(current_buffer, partial_data)

        # Update result
        self._result.partial_data = partial_data
        self._result.state = self._determine_state(current_buffer)

        # Check if we have meaningful partial data
        if partial_data:
            self._result.is_valid = True

    def _extract_partial_kvps(self, text: str, partial_data: Dict[str, Any]) -> None:
        """Extract partial key-value pairs from JSON text."""

        # Pattern for key-value pairs
        kvp_pattern = r'"([^"]+)"\s*:\s*([^,}\]]+(?:[,}\]]|$))'

        for match in re.finditer(kvp_pattern, text):
            key = match.group(1)
            value_str = match.group(2).rstrip(",}]").strip()

            try:
                # Try to parse the value
                if value_str.startswith('"') and value_str.endswith('"'):
                    value = value_str[1:-1]  # String value
                elif value_str in ("true", "false"):
                    value = value_str == "true"  # Boolean
                elif value_str == "null":
                    value = None
                else:
                    # Try numeric parsing
                    try:
                        value = (
                            int(value_str) if "." not in value_str else float(value_str)
                        )
                    except ValueError:
                        value = value_str  # Keep as string if unparseable

                partial_data[key] = value

            except Exception as e:
                logger.debug(f"Failed to parse value for key '{key}': {e}")
                continue

    def _extract_partial_arrays(self, text: str, partial_data: Dict[str, Any]) -> None:
        """Extract partial arrays from JSON text."""

        # Pattern for array fields
        array_pattern = r'"([^"]+)"\s*:\s*\[([^\]]*)'

        for match in re.finditer(array_pattern, text):
            key = match.group(1)
            array_content = match.group(2).strip()

            if not array_content:
                partial_data[key] = []
                continue

            try:
                # Split array elements and parse
                elements = []
                for element in array_content.split(","):
                    element = element.strip()
                    if element:
                        if element.startswith('"') and element.endswith('"'):
                            elements.append(element[1:-1])
                        elif element in ("true", "false"):
                            elements.append(element == "true")
                        elif element == "null":
                            elements.append(None)
                        else:
                            try:
                                elements.append(
                                    int(element)
                                    if "." not in element
                                    else float(element)
                                )
                            except ValueError:
                                elements.append(element)

                partial_data[key] = elements

            except Exception as e:
                logger.debug(f"Failed to parse array for key '{key}': {e}")
                continue

    def _determine_state(self, text: str) -> ParseState:
        """Determine current parser state from buffer content."""

        if not text:
            return ParseState.INIT

        # Count brackets and braces
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        if open_braces > 0:
            return ParseState.IN_OBJECT
        elif open_brackets > 0:
            return ParseState.IN_ARRAY
        elif text.count('"') % 2 == 1:
            return ParseState.IN_STRING
        else:
            return ParseState.IN_VALUE

    def _handle_parse_error(self, error_msg: str) -> None:
        """Handle parsing errors gracefully."""

        self._result.is_valid = False
        self._result.error = error_msg
        self._result.state = ParseState.ERROR

        logger.warning(f"JSON parsing error: {error_msg}")

        # Try to salvage partial data despite error
        if not self.strict_mode:
            try:
                self._incremental_parse()
            except Exception as e:
                logger.debug(f"Failed to recover partial data: {e}")

    def get_result(self) -> ParseResult:
        """Get current parsing result."""
        return self._result

    def is_complete(self) -> bool:
        """Check if parsing is complete."""
        return self._result.is_complete

    def has_partial_data(self) -> bool:
        """Check if partial data is available."""
        return bool(self._result.partial_data) or self._result.parsed_data is not None


class StreamingParser:
    """
    High-level streaming parser for real-time JSON processing.

    Manages multiple partial parsers and provides callbacks
    for streaming data updates.
    """

    def __init__(
        self,
        on_partial_update=None,
        on_complete=None,
        on_error=None,
        buffer_size: int = 8192,
    ):
        self.on_partial_update = on_partial_update
        self.on_complete = on_complete
        self.on_error = on_error
        self.buffer_size = buffer_size

        self._parsers: Dict[str, PartialJSONParser] = {}
        self._streams: Dict[str, str] = {}

    def create_stream(self, stream_id: str) -> None:
        """Create a new parsing stream."""

        self._parsers[stream_id] = PartialJSONParser()
        self._streams[stream_id] = ""

        logger.debug(f"Created streaming parser for {stream_id}")

    def feed_stream(self, stream_id: str, chunk: str) -> ParseResult:
        """Feed data to a specific stream."""

        if stream_id not in self._parsers:
            self.create_stream(stream_id)

        parser = self._parsers[stream_id]
        result = parser.feed(chunk)

        # Trigger callbacks
        if result.partial_data and self.on_partial_update:
            try:
                self.on_partial_update(stream_id, result.partial_data, result)
            except Exception as e:
                logger.error(f"Error in partial update callback: {e}")

        if result.is_complete and self.on_complete:
            try:
                self.on_complete(stream_id, result.parsed_data, result)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")

        if result.error and self.on_error:
            try:
                self.on_error(stream_id, result.error, result)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

        return result

    def close_stream(self, stream_id: str) -> Optional[ParseResult]:
        """Close and cleanup a stream."""

        if stream_id not in self._parsers:
            return None

        result = self._parsers[stream_id].get_result()

        # Cleanup
        del self._parsers[stream_id]
        del self._streams[stream_id]

        logger.debug(f"Closed streaming parser for {stream_id}")
        return result

    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs."""
        return list(self._parsers.keys())

    def get_stream_result(self, stream_id: str) -> Optional[ParseResult]:
        """Get current result for a stream."""

        if stream_id not in self._parsers:
            return None

        return self._parsers[stream_id].get_result()


# Utility functions for common streaming scenarios


def parse_streaming_json(
    data_stream: Generator[str, None, None], on_update=None
) -> ParseResult:
    """
    Parse a streaming JSON generator with update callbacks.

    Args:
        data_stream: Generator yielding JSON chunks
        on_update: Callback for partial updates

    Returns:
        Final ParseResult
    """

    parser = PartialJSONParser()

    for chunk in data_stream:
        result = parser.feed(chunk)

        if on_update and (result.partial_data or result.parsed_data):
            on_update(result)

        if result.is_complete:
            break

    return parser.get_result()


def safe_partial_json_parse(json_str: str) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Safely parse potentially incomplete JSON.

    Args:
        json_str: JSON string (potentially incomplete)

    Returns:
        Tuple of (parsed_data, is_complete)
    """

    parser = PartialJSONParser()
    result = parser.feed(json_str)

    if result.parsed_data:
        return result.parsed_data, result.is_complete
    elif result.partial_data:
        return result.partial_data, False
    else:
        return None, False
