"""
Enhanced and flexible regex-based monitor implementation.

This module provides a highly configurable regex monitor that can validate agent
outputs against complex patterns with fine-grained control over matching logic,
target data, and violation conditions.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List

from .base import MonitorFn, MonitorSpec

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors.regex")

# A mapping from string names to re module flag constants for safe configuration
RE_FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "VERBOSE": re.VERBOSE,
    "ASCII": re.ASCII,
    "UNICODE": re.UNICODE,
}


def create_regex_monitor(spec: MonitorSpec) -> MonitorFn:
    """
    Create a powerful and flexible regex-based monitor.

    This factory configures a monitor that can match patterns against various parts
    of an agent's output, with support for inverse matching, match counting,
    and named group extraction.

    Args:
        spec: Monitor specification with parameters:
            - pattern (str): Regex pattern to match (required).
            - target_fields (List[str]): Fields to scan in the result dict. Can
              also include "task" to scan the input prompt.
              Defaults to ["content"].
            - match_mode (str): Matching strategy. One of 'search', 'fullmatch',
              or 'finditer'. Defaults to 'search'.
            - invert_match (bool): If True, triggers a violation if the pattern
              is NOT found. Defaults to False.
            - flags (List[str]): A list of standard regex flag names (e.g.,
              ["IGNORECASE", "DOTALL"]). Defaults to ["IGNORECASE", "MULTILINE"].
            - min_matches (int): Minimum number of matches required to avoid a
              violation (used with 'finditer').
            - max_matches (int): Maximum number of matches allowed to avoid a
              violation (used with 'finditer').
            - extract_named_groups (bool): If True, extracts named capture groups
              from the first match into violation metadata. Defaults to True.
            - violation_name (str): Custom name for the violation.

    Returns:
        The configured monitor function.

    Raises:
        ValueError: If the pattern is missing or if the configuration
                    (e.g., flags, pattern syntax) is invalid.
    """
    # --- 1. Parameter Extraction and Validation ---
    pattern = spec.params.get("pattern")
    if not pattern:
        raise ValueError("Regex monitor requires 'pattern' in spec.params")

    violation_name = spec.params.get("violation_name", f"{spec.name}_regex_violation")
    target_fields = spec.params.get("target_fields", ["content"])
    if isinstance(target_fields, str):
        target_fields = [target_fields]

    match_mode = spec.params.get("match_mode", "search")
    invert_match = spec.params.get("invert_match", False)
    flag_names = spec.params.get("flags", ["IGNORECASE", "MULTILINE"])
    min_matches = spec.params.get("min_matches")
    max_matches = spec.params.get("max_matches")
    extract_named_groups = spec.params.get("extract_named_groups", True)

    # --- 2. Compile Regex with specified flags ---
    try:
        compiled_flags = 0
        for flag in flag_names:
            compiled_flags |= RE_FLAG_MAP[flag.upper()]
        rx = re.compile(pattern, compiled_flags)
    except KeyError as e:
        raise ValueError(f"Invalid regex flag specified: {e}. Available: {list(RE_FLAG_MAP.keys())}")
    except re.error as e:
        raise ValueError(f"Invalid regex pattern provided: {e}")

    # --- 3. The Monitor Function ---
    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        from .factory import MonitorFactory  # Avoid circular import

        if MonitorFactory._should_cooldown(spec, task):
            return

        scannable_data = {"task": task, **(result if isinstance(result, dict) else {"content": str(result)})}
        found_match_in_any_field = False

        for field_name in target_fields:
            content_to_scan = scannable_data.get(field_name)
            if not isinstance(content_to_scan, str):
                continue  # Skip fields that are missing or not strings

            # --- 4. Perform Matching ---
            matches = list(rx.finditer(content_to_scan)) if match_mode in ["finditer", "search", "fullmatch"] else []
            if match_mode == "search" and matches:
                matches = matches[:1]
            elif match_mode == "fullmatch" and (not matches or matches[0].group(0) != content_to_scan):
                matches = []

            is_match_found = bool(matches)
            found_match_in_any_field |= is_match_found

            # --- 5. Evaluate Violation Conditions for forbidden patterns ---
            if is_match_found and not invert_match:
                num_matches = len(matches)
                violation_triggered = False
                rationale = ""

                if min_matches is not None and num_matches < min_matches:
                    violation_triggered = True
                    rationale = f"Pattern matched {num_matches} time(s) in field '{field_name}', which is below the required minimum of {min_matches}."
                elif max_matches is not None and num_matches > max_matches:
                    violation_triggered = True
                    rationale = f"Pattern matched {num_matches} time(s) in field '{field_name}', exceeding the maximum of {max_matches}."
                elif min_matches is None and max_matches is None:
                    violation_triggered = True
                    rationale = f"Forbidden pattern matched in field '{field_name}'. First match: '{matches[0].group(0)[:100]}'"

                if violation_triggered:
                    _handle_violation(
                        spec, agent, task, content_to_scan, rationale, violation_name,
                        pattern, field_name, match_mode, matches, extract_named_groups
                    )
                    return  # Violation found, stop processing

        # --- 6. Evaluate Violation for required patterns that were not found ---
        if invert_match and not found_match_in_any_field:
            rationale = f"Required pattern was not found in any target fields: {target_fields}."
            _handle_violation(
                spec, agent, task, str(scannable_data), rationale, violation_name,
                pattern, ", ".join(target_fields), match_mode, [], False
            )

    return monitor


def _handle_violation(
    spec: MonitorSpec, agent: "AgentNet", task: str, content: str, rationale: str,
    violation_name: str, pattern: str, field_name: str, match_mode: str,
    matches: List[re.Match], extract_named_groups: bool
) -> None:
    """Helper function to build and dispatch a violation."""
    from .factory import MonitorFactory

    first_match_text = ""
    named_groups = {}
    if matches:
        first_match = matches[0]
        first_match_text = first_match.group(0)
        if extract_named_groups:
            named_groups = first_match.groupdict()

    meta = {
        "pattern": pattern,
        "target_field": field_name,
        "match_mode": match_mode,
        "match_count": len(matches),
        "first_match_text": first_match_text[:200],  # Truncate for readability
        "named_groups": named_groups,
    }

    violations = [
        MonitorFactory._build_violation(
            name=violation_name,
            vtype="regex",
            severity=spec.severity,
            description=spec.description or "A regex-based guardrail was triggered.",
            rationale=rationale,
            meta=meta,
        )
    ]
    detail = {"outcome": {"content_scanned": content[:500]}, "violations": violations}
    MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
