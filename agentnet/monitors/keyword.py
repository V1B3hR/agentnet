"""
Advanced keyword-based monitor with support for case sensitivity, whole word matching,
inverse logic, and match counting for precise content validation.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List

from .base import MonitorFn, MonitorSpec

if TYPE_CHECKING:
    from ..core.agent import AgentNet

logger = logging.getLogger("agentnet.monitors.keyword")


def create_keyword_monitor(spec: MonitorSpec) -> MonitorFn:
    """
    Create a powerful and flexible keyword-based monitor.

    Args:
        spec: Monitor specification with parameters:
            - keywords (List[str]): List of keywords to match (required).
            - target_fields (List[str]): Fields to scan in the result/task.
              Defaults to ["content"].
            - case_sensitive (bool): If True, performs a case-sensitive match.
              Defaults to False.
            - match_whole_word (bool): If True, only matches whole words, not
              substrings. Defaults to False.
            - invert_match (bool): If True, triggers a violation if keywords are
              NOT found. Defaults to False.
            - match_mode (str): "any" (triggers on one or more keywords) or "all"
              (triggers only if all keywords are present/absent). Defaults to "any".
            - min_matches (int): Minimum total number of matches required to
              avoid a violation (used with invert_match=True).
            - max_matches (int): Maximum total number of matches allowed to
              avoid a violation (used with invert_match=False).
            - violation_name (str): Custom name for the violation.

    Returns:
        The configured monitor function.
    """
    # --- 1. Configuration and Validation ---
    keywords = spec.params.get("keywords", [])
    if not keywords:
        # Return a no-op function if no keywords are provided
        return lambda agent, task, result: None

    target_fields = spec.params.get("target_fields", ["content"])
    case_sensitive = spec.params.get("case_sensitive", False)
    match_whole_word = spec.params.get("match_whole_word", False)
    invert_match = spec.params.get("invert_match", False)
    match_mode_all = spec.params.get("match_mode", "any").lower() == "all"
    min_matches = spec.params.get("min_matches")
    max_matches = spec.params.get("max_matches")
    violation_name = spec.params.get("violation_name", f"{spec.name}_keyword_violation")

    # --- 2. Prepare Keywords and Regex ---
    re_flags = 0 if case_sensitive else re.IGNORECASE
    
    # For whole word matching, pre-compile regex for performance
    keyword_finders = {}
    for kw in keywords:
        pattern = re.escape(kw)
        if match_whole_word:
            pattern = r'\b' + pattern + r'\b'
        keyword_finders[kw] = re.compile(pattern, re_flags)

    def monitor(agent: "AgentNet", task: str, result: Dict[str, Any]) -> None:
        from .factory import MonitorFactory
        if MonitorFactory._should_cooldown(spec, task):
            return

        scannable_data = {"task": task, **(result if isinstance(result, dict) else {"content": str(result)})}

        for field_name in target_fields:
            content_to_scan = scannable_data.get(field_name)
            if not isinstance(content_to_scan, str):
                continue

            # --- 3. Find and Count Matches ---
            present_keywords = set()
            total_matches = 0
            match_details = {}

            for kw, finder in keyword_finders.items():
                matches = finder.findall(content_to_scan)
                if matches:
                    count = len(matches)
                    present_keywords.add(kw)
                    total_matches += count
                    match_details[kw] = count

            # --- 4. Evaluate Violation Conditions ---
            violation_triggered = False
            rationale = ""

            # Condition 1: Check for forbidden keywords
            if not invert_match:
                found_any = len(present_keywords) > 0
                found_all = len(present_keywords) == len(keywords)
                
                if (match_mode_all and found_all) or (not match_mode_all and found_any):
                    if max_matches is not None and total_matches > max_matches:
                        violation_triggered = True
                        rationale = f"Found {total_matches} total keyword matches in '{field_name}', exceeding the maximum of {max_matches}."
                    elif max_matches is None:
                        violation_triggered = True
                        rationale = f"Forbidden keyword(s) found in field '{field_name}': {', '.join(sorted(present_keywords))}"

            # Condition 2: Check for required keywords
            else: # invert_match is True
                missing_any = len(present_keywords) < len(keywords)
                missing_all = len(present_keywords) == 0

                if (match_mode_all and missing_any) or (not match_mode_all and missing_all):
                    violation_triggered = True
                    missing_kws = set(keywords) - present_keywords
                    rationale = f"Required keyword(s) missing from field '{field_name}': {', '.join(sorted(missing_kws))}"
                elif min_matches is not None and total_matches < min_matches:
                    violation_triggered = True
                    rationale = f"Found only {total_matches} total keyword matches in '{field_name}', below the required minimum of {min_matches}."

            # --- 5. Handle Violation ---
            if violation_triggered:
                _handle_violation(
                    spec, agent, task, content_to_scan, rationale, violation_name,
                    field_name, list(present_keywords), total_matches, match_details
                )
                return # Stop after first violation

    return monitor


def _handle_violation(
    spec: MonitorSpec, agent: "AgentNet", task: str, content: str, rationale: str,
    violation_name: str, field_name: str, matched_keywords: List[str],
    total_matches: int, match_details: Dict[str, int]
) -> None:
    """Helper function to build and dispatch a violation."""
    from .factory import MonitorFactory

    meta = {
        "target_field": field_name,
        "matched_keywords": sorted(matched_keywords),
        "total_matches": total_matches,
        "match_counts": match_details,
        "config": {
            "keywords": spec.params.get("keywords"),
            "case_sensitive": spec.params.get("case_sensitive", False),
            "match_whole_word": spec.params.get("match_whole_word", False),
            "invert_match": spec.params.get("invert_match", False),
            "match_mode": spec.params.get("match_mode", "any"),
        }
    }

    violation = MonitorFactory._build_violation(
        name=violation_name,
        vtype="keyword",
        severity=spec.severity,
        description=spec.description or "A keyword-based guardrail was triggered.",
        rationale=rationale,
        meta=meta,
    )
    detail = {"outcome": {"content_scanned": content[:500]}, "violations": [violation]}
    MonitorFactory._handle(spec, agent, task, passed=False, detail=detail)
