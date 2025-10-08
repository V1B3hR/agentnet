"""
Enhanced and flexible regex-based monitor implementation.

This module provides a highly configurable regex monitor that can validate agent
outputs against complex patterns with fine-grained control over matching logic,
target data, violation conditions, and result metadata.

Key Improvements (enhanced version):
- Centralized and stricter parameter validation with detailed error messages.
- Added new match modes: 'anyline' (line-wise search) and 'startswith', 'endswith'.
- Added support for pattern pre-validation and optional safe length truncation.
- Added ability to:
  * capture all matches (optionally including named groups for each)
  * include / exclude specific target fields
  * configure violation triggering strategy (first vs aggregate)
  * differentiate between required vs forbidden patterns with explicit 'pattern_type'
  * configure logical aggregation for invert / required patterns (any vs all)
  * set soft vs hard thresholds (min/max) with custom messages
- Introduced 'collect_all_matches' and 'max_capture' params to avoid excessive memory usage.
- Added richer meta reporting including timing, field scan order, and aggregated counts.
- Added optional 'strip' and 'normalize_whitespace' preprocessing of field content.
- Added optional 'redact_patterns' to hide sensitive substrings in violation detail.
- Added compiled pattern caching (per spec id) to reduce repeated compile overhead.
- Improved type hints and documentation.
- Graceful handling of non-string fields with optional 'coerce_non_string' flag.
- Extended flag handling with short aliases (e.g. I, M, S).

Backward compatibility:
- Existing parameters from previous implementation continue to work:
  pattern, target_fields, match_mode, invert_match, flags,
  min_matches, max_matches, extract_named_groups, violation_name.

"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

from .base import MonitorFn, MonitorSpec

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors.regex")

# A mapping from string names to re module flag constants for safe configuration
RE_FLAG_MAP = {
    "I": re.IGNORECASE,
    "IGNORECASE": re.IGNORECASE,
    "M": re.MULTILINE,
    "MULTILINE": re.MULTILINE,
    "S": re.DOTALL,
    "DOTALL": re.DOTALL,
    "X": re.VERBOSE,
    "VERBOSE": re.VERBOSE,
    "A": re.ASCII,
    "ASCII": re.ASCII,
    "U": re.UNICODE,
    "UNICODE": re.UNICODE,
}

MatchMode = Literal[
    "search",
    "fullmatch",
    "finditer",
    "anyline",
    "startswith",
    "endswith",
]

PatternType = Literal["forbidden", "required"]

RequiredAggregation = Literal["any_field", "all_fields"]

ViolationTriggerStrategy = Literal["first", "aggregate"]


@dataclass(frozen=True)
class CompiledRegexConfig:
    pattern: str
    flags: int
    match_mode: MatchMode
    raw_flags: Sequence[str]


# Simple in-process cache keyed by (pattern, flags_tuple, match_mode)
_COMPILED_CACHE: Dict[Tuple[str, Tuple[str, ...], MatchMode], re.Pattern] = {}


def _compile_pattern(pattern: str, flag_names: Sequence[Union[str, int]], match_mode: MatchMode) -> re.Pattern:
    cache_key = (pattern, tuple(map(str, flag_names)), match_mode)
    cached = _COMPILED_CACHE.get(cache_key)
    if cached:
        return cached

    flags_value = 0
    for f in flag_names:
        if isinstance(f, int):
            flags_value |= f
        else:
            key = f.upper()
            if key not in RE_FLAG_MAP:
                raise ValueError(f"Invalid regex flag '{f}'. Available: {sorted(RE_FLAG_MAP.keys())}")
            flags_value |= RE_FLAG_MAP[key]
    try:
        compiled = re.compile(pattern, flags_value)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
    _COMPILED_CACHE[cache_key] = compiled
    return compiled


def create_regex_monitor(spec: MonitorSpec) -> MonitorFn:
    """
    Create a powerful and flexible regex-based monitor.

    Args:
        spec.params supported keys (all optional unless noted):

        Core:
        - pattern (str | Pattern): REQUIRED. Regex pattern.
        - violation_name (str): Custom violation name. Default: f"{spec.name}_regex_violation".
        - pattern_type ("forbidden" | "required"): Default: inferred (if invert_match True -> required).
        - invert_match (bool): Deprecated alias for pattern_type="required". Default False.

        Target fields:
        - target_fields (List[str] | str): Fields to scan (can include "task"). Default ["content"].
        - exclude_fields (List[str]): Fields to exclude after inclusion.
        - coerce_non_string (bool): Convert non-str values with str() before scanning. Default False.
        - field_scan_limit (int): Maximum number of fields to scan (after filtering).

        Matching:
        - match_mode (MatchMode): One of 'search','fullmatch','finditer','anyline','startswith','endswith'. Default 'search'.
        - flags (List[str] | List[int]): Regex flags by name or int. Default ["IGNORECASE","MULTILINE"].
        - min_matches (int): Required minimum (applies mainly to finditer / anyline).
        - max_matches (int): Maximum allowed matches.
        - soft_min_message (str): Custom rationale when below min.
        - soft_max_message (str): Custom rationale when above max.
        - required_aggregation ("any_field" | "all_fields"): For required patterns: condition across fields. Default "any_field".

        Extraction / Reporting:
        - extract_named_groups (bool): Extract named groups from first match. Default True.
        - collect_all_matches (bool): Include all matched texts in meta. Default False.
        - capture_group_limit (int): Max number of matches to record. Default 25.
        - max_capture (int): Alias for capture_group_limit.
        - redact_patterns (List[str]): Additional regex patterns whose matches are replaced with '***' in content snippet.
        - include_field_order (bool): Include scanning order in meta. Default True.
        - truncate_content (int): Truncate scanned content in meta to this length. Default 500.
        - truncate_match (int): Truncate first match text to length. Default 200.

        Preprocessing:
        - strip (bool): strip() each field text before matching. Default False.
        - normalize_whitespace (bool): Collapse consecutive whitespace to single space. Default False.

        Control:
        - violation_trigger_strategy ("first"|"aggregate"): Stop at first violation or aggregate counts and then decide. Default "first".
        - cooldown_* (handled upstream by MonitorFactory if present).

    Returns:
        MonitorFn: A callable monitor function.

    Raises:
        ValueError: On invalid configuration.
    """
    params = spec.params or {}

    # ---- Extract core parameters ----
    pattern = params.get("pattern")
    if isinstance(pattern, re.Pattern):
        pattern_str = pattern.pattern
        flag_names = [pattern.flags]  # treat as single int
    else:
        pattern_str = pattern
        flag_names = params.get("flags", ["IGNORECASE", "MULTILINE"])

    if not pattern_str or not isinstance(pattern_str, str):
        raise ValueError("Regex monitor requires a 'pattern' (string or compiled).")

    violation_name: str = params.get("violation_name", f"{spec.name}_regex_violation")

    invert_match: bool = bool(params.get("invert_match", False))
    pattern_type: PatternType = params.get("pattern_type") or ("required" if invert_match else "forbidden")
    if pattern_type not in ("forbidden", "required"):
        raise ValueError("pattern_type must be one of ['forbidden','required'].")

    # Target fields
    target_fields_param = params.get("target_fields", ["content"])
    if isinstance(target_fields_param, str):
        target_fields: List[str] = [target_fields_param]
    else:
        target_fields = list(target_fields_param)

    exclude_fields: List[str] = list(params.get("exclude_fields", []) or [])
    field_scan_limit: Optional[int] = params.get("field_scan_limit")
    coerce_non_string: bool = bool(params.get("coerce_non_string", False))

    # Matching
    match_mode: MatchMode = params.get("match_mode", "search")
    valid_match_modes: Tuple[str, ...] = ("search", "fullmatch", "finditer", "anyline", "startswith", "endswith")
    if match_mode not in valid_match_modes:
        raise ValueError(f"Invalid match_mode '{match_mode}'. Allowed: {valid_match_modes}")

    min_matches = params.get("min_matches")
    max_matches = params.get("max_matches")
    if min_matches is not None and (not isinstance(min_matches, int) or min_matches < 0):
        raise ValueError("min_matches must be a non-negative integer.")
    if max_matches is not None and (not isinstance(max_matches, int) or max_matches < 0):
        raise ValueError("max_matches must be a non-negative integer.")
    if (min_matches is not None and max_matches is not None) and min_matches > max_matches:
        raise ValueError("min_matches cannot exceed max_matches.")

    required_aggregation: RequiredAggregation = params.get("required_aggregation", "any_field")
    if required_aggregation not in ("any_field", "all_fields"):
        raise ValueError("required_aggregation must be 'any_field' or 'all_fields'.")

    # Reporting / extraction
    extract_named_groups: bool = bool(params.get("extract_named_groups", True))
    collect_all_matches: bool = bool(params.get("collect_all_matches", False))
    capture_group_limit = params.get("capture_group_limit", params.get("max_capture", 25))
    if not isinstance(capture_group_limit, int) or capture_group_limit <= 0:
        raise ValueError("capture_group_limit must be a positive integer.")
    truncate_content: int = int(params.get("truncate_content", 500))
    truncate_match: int = int(params.get("truncate_match", 200))
    include_field_order: bool = bool(params.get("include_field_order", True))
    redact_patterns: List[str] = list(params.get("redact_patterns", []) or [])

    # Preprocessing
    do_strip: bool = bool(params.get("strip", False))
    normalize_ws: bool = bool(params.get("normalize_whitespace", False))

    # Control
    violation_trigger_strategy: ViolationTriggerStrategy = params.get("violation_trigger_strategy", "first")
    if violation_trigger_strategy not in ("first", "aggregate"):
        raise ValueError("violation_trigger_strategy must be 'first' or 'aggregate'.")

    soft_min_message: Optional[str] = params.get("soft_min_message")
    soft_max_message: Optional[str] = params.get("soft_max_message")

    # Compile main pattern
    rx = _compile_pattern(pattern_str, flag_names if isinstance(flag_names, (list, tuple)) else [flag_names], match_mode)

    # Pre-compile redact patterns
    redact_compiled: List[Tuple[str, re.Pattern]] = []
    for rp in redact_patterns:
        try:
            redact_compiled.append((rp, re.compile(rp)))
        except re.error as e:
            raise ValueError(f"Invalid redact pattern '{rp}': {e}")

    def _preprocess_text(text: str) -> str:
        if do_strip:
            text = text.strip()
        if normalize_ws:
            # Collapse any sequence of whitespace to a single space
            text = re.sub(r"\s+", " ", text)
        return text

    def _redact(text: str) -> str:
        for orig, rp in redact_compiled:
            text = rp.sub("***", text)
        return text

    def _collect_matches(content: str) -> List[re.Match]:
        if match_mode in ("finditer", "search", "fullmatch"):
            matches = list(rx.finditer(content))
            if match_mode == "search":
                matches = matches[:1]
            elif match_mode == "fullmatch":
                if not matches or matches[0].group(0) != content:
                    matches = []
            return matches
        elif match_mode == "anyline":
            matches: List[re.Match] = []
            for line in content.splitlines():
                # Each line: behave like search
                line_matches = list(rx.finditer(line))
                if line_matches:
                    matches.extend(line_matches)
            return matches
        elif match_mode == "startswith":
            return [m for m in [rx.match(content)] if m and content.startswith(m.group(0))]
        elif match_mode == "endswith":
            # Use search and filter where match ends at len(content)
            m_list = list(rx.finditer(content))
            return [m for m in m_list if m.end() == len(content)]
        else:
            return []

    # ---------- Monitor Function ----------
    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        start_time = time.time()
        from .factory import MonitorFactory  # local import to avoid circular dependency

        if MonitorFactory._should_cooldown(spec, task):
            return

        # Build scannable data
        scannable_data: Dict[str, Any] = {"task": task}
        if isinstance(result, dict):
            scannable_data.update(result)
        else:
            scannable_data["content"] = str(result)

        # Select fields
        fields: List[str] = [f for f in target_fields if f not in exclude_fields]
        if field_scan_limit is not None:
            fields = fields[:field_scan_limit]

        field_order_map = {fname: idx for idx, fname in enumerate(fields)}

        # State tracking
        required_matches_per_field: Dict[str, bool] = {}
        violations_emitted = False

        aggregated_details: List[Dict[str, Any]] = []
        total_matches_count = 0

        for field_name in fields:
            raw_value = scannable_data.get(field_name)
            if raw_value is None:
                required_matches_per_field[field_name] = False
                continue
            if not isinstance(raw_value, str):
                if coerce_non_string:
                    try:
                        raw_value = str(raw_value)
                    except Exception:
                        required_matches_per_field[field_name] = False
                        continue
                else:
                    required_matches_per_field[field_name] = False
                    continue

            content_to_scan = _preprocess_text(raw_value)

            matches = _collect_matches(content_to_scan)
            num_matches = len(matches)
            total_matches_count += num_matches
            is_match_found = num_matches > 0
            required_matches_per_field[field_name] = is_match_found

            def emit_violation(rationale: str, match_list: List[re.Match]) -> None:
                nonlocal violations_emitted
                first_match_text = match_list[0].group(0) if match_list else ""
                named_groups = match_list[0].groupdict() if (match_list and extract_named_groups) else {}
                all_matches_serialized: Optional[List[str]] = None
                if collect_all_matches:
                    truncated = match_list[:capture_group_limit]
                    all_matches_serialized = [m.group(0) for m in truncated]

                meta = {
                    "pattern": pattern_str,
                    "pattern_type": pattern_type,
                    "target_field": field_name,
                    "match_mode": match_mode,
                    "match_count": len(match_list),
                    "first_match_text": first_match_text[:truncate_match],
                    "named_groups": named_groups,
                    "all_matches": all_matches_serialized,
                    "field_order": field_order_map.get(field_name) if include_field_order else None,
                    "min_matches": min_matches,
                    "max_matches": max_matches,
                    "required_aggregation": required_aggregation if pattern_type == "required" else None,
                    "elapsed_ms": round((time.time() - start_time) * 1000, 3),
                }

                redacted_content = _redact(content_to_scan)[:truncate_content]

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
                detail = {"outcome": {"content_scanned": redacted_content}, "violations": violations}
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
                violations_emitted = True

            # Decide violation logic depending on pattern type
            if pattern_type == "forbidden":
                # If any match is found, treat as potential violation after threshold evaluation
                if is_match_found:
                    # Evaluate thresholds if provided
                    violation_triggered = False
                    if min_matches is not None and num_matches < min_matches:
                        rationale = (
                            soft_min_message
                            or f"Forbidden pattern matched {num_matches} time(s) in field '{field_name}', below configured minimum {min_matches}."
                        )
                        violation_triggered = True
                    elif max_matches is not None and num_matches > max_matches:
                        rationale = (
                            soft_max_message
                            or f"Forbidden pattern matched {num_matches} time(s) in field '{field_name}', exceeding maximum {max_matches}."
                        )
                        violation_triggered = True
                    elif min_matches is None and max_matches is None:
                        rationale = (
                            f"Forbidden pattern matched in field '{field_name}'. "
                            f"First match: '{matches[0].group(0)[:truncate_match]}'"
                        )
                        violation_triggered = True

                    if violation_triggered:
                        emit_violation(rationale, matches)
                        if violation_trigger_strategy == "first":
                            return
                    else:
                        # No violation under thresholds – continue scanning
                        pass
            else:  # required pattern
                # Only evaluate at end unless strategy 'first' and we can assert failure early
                pass

            aggregated_details.append(
                {
                    "field": field_name,
                    "match_count": num_matches,
                    "matched": is_match_found,
                }
            )

            if violations_emitted and violation_trigger_strategy == "first":
                return

        if violations_emitted:
            return  # Already emitted (forbidden case)

        # End-of-scan evaluation for required pattern
        if pattern_type == "required":
            if required_aggregation == "any_field":
                success = any(required_matches_per_field.values())
            else:  # all_fields
                # Only consider fields that were actually scanned (present)
                present_fields = [f for f in fields if f in scannable_data]
                success = present_fields and all(
                    required_matches_per_field.get(f, False) for f in present_fields
                )

            if not success:
                missing_fields = [
                    f for f in fields if not required_matches_per_field.get(f, False)
                ]
                rationale = (
                    f"Required pattern was not found with aggregation='{required_aggregation}'. "
                    f"Missing fields: {missing_fields[:10]} (total {len(missing_fields)})."
                )
                # Use first missing field value (if any) for context
                first_context_field = next(iter(missing_fields), None)
                content_for_detail = ""
                if first_context_field and isinstance(scannable_data.get(first_context_field), str):
                    content_for_detail = scannable_data[first_context_field]
                else:
                    # fallback combined
                    content_for_detail = "\n".join(
                        f"{k}={str(scannable_data.get(k))[:100]}"
                        for k in fields[:5]
                    )

                # Build a synthetic match list empty
                from .factory import MonitorFactory  # re-import to satisfy linter

                meta = {
                    "pattern": pattern_str,
                    "pattern_type": pattern_type,
                    "required_aggregation": required_aggregation,
                    "fields_scanned": fields,
                    "found_map": required_matches_per_field,
                    "total_matches": total_matches_count,
                    "elapsed_ms": round((time.time() - start_time) * 1000, 3),
                }

                violations = [
                    MonitorFactory._build_violation(
                        name=violation_name,
                        vtype="regex",
                        severity=spec.severity,
                        description=spec.description or "A required regex pattern was not satisfied.",
                        rationale=rationale,
                        meta=meta,
                    )
                ]
                detail = {
                    "outcome": {
                        "content_scanned": _redact(content_for_detail)[:truncate_content]
                    },
                    "violations": violations,
                }
                MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
                return

        # If we reached here without emitting a violation => passed
        from .factory import MonitorFactory  # local import

        # Include a short pass detail (optional)
        pass_meta = {
            "pattern": pattern_str,
            "pattern_type": pattern_type,
            "match_mode": match_mode,
            "total_matches": total_matches_count,
            "elapsed_ms": round((time.time() - start_time) * 1000, 3),
        }
        MonitorFactory._handle(
            spec,
            agent,
            task,
            passed=True,
            detail={"outcome": {"summary": pass_meta}},
        )

    return monitor


# Backward compatibility: retain the original helper for direct violation handling (used internally).
def _handle_violation(
    spec: MonitorSpec,
    agent: "AgentNet",
    task: str,
    content: str,
    rationale: str,
    violation_name: str,
    pattern: str,
    field_name: str,
    match_mode: str,
    matches: List[re.Match],
    extract_named_groups: bool,
) -> None:
    """
    Deprecated helper retained for backward compatibility with earlier code paths.
    New logic routes through enhanced monitor; this remains for compatibility.
    """
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
        "first_match_text": first_match_text[:200],
        "named_groups": named_groups,
        "legacy_path": True,
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
