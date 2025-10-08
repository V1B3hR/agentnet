"""
Temporal reasoning capabilities for Phase 7.

Implements temporal reasoning patterns and sequences for enhanced memory systems.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

from .types import BaseReasoning, ReasoningResult, ReasoningType

logger = logging.getLogger("agentnet.reasoning.temporal")


class TemporalRelation(str, Enum):
    """Types of temporal relationships."""

    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    MEETS = "meets"
    STARTS = "starts"
    FINISHES = "finishes"
    EQUALS = "equals"
    CONTAINS = "contains"


@dataclass
class TemporalEvent:
    """An event with temporal information."""

    id: str
    content: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalPattern:
    """A pattern of temporal events."""

    name: str
    events: List[TemporalEvent]
    relations: List[
        Tuple[str, str, TemporalRelation]
    ]  # (event1_id, event2_id, relation)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalSequence:
    """A sequence of temporal events with ordering."""

    events: List[TemporalEvent]
    ordering: List[Tuple[str, str]]  # (predecessor_id, successor_id)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalRule:
    """A temporal reasoning rule."""

    name: str
    conditions: List[str]
    conclusions: List[str]
    temporal_constraints: List[Tuple[str, str, TemporalRelation]]
    confidence: float = 1.0


class TemporalReasoning(BaseReasoning):
    """Temporal reasoning for episodic memory and event analysis."""

    def __init__(self, style_weights: Dict[str, float]):
        super().__init__(style_weights)
        self.temporal_rules = self._initialize_temporal_rules()
        self.patterns = []
        self.sequences = []

    def reason(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """Perform temporal reasoning on events or sequences."""
        context = context or {}

        reasoning_steps = []

        # Extract temporal information from context
        events = self._extract_temporal_events(task, context)
        reasoning_steps.append(f"Extracted {len(events)} temporal events")

        # Identify temporal patterns
        patterns = self._identify_patterns(events)
        reasoning_steps.extend([f"Pattern: {p.name}" for p in patterns])

        # Build temporal sequences
        sequences = self._build_sequences(events)
        reasoning_steps.append(f"Built {len(sequences)} temporal sequences")

        # Apply temporal reasoning rules
        inferences = self._apply_temporal_rules(events, patterns)
        reasoning_steps.extend([f"Inference: {inf}" for inf in inferences])

        # Form temporal conclusion
        conclusion = self._form_temporal_conclusion(
            events, patterns, sequences, inferences
        )

        # Calculate confidence based on temporal coherence
        confidence = self._calculate_temporal_confidence(events, patterns, sequences)
        final_confidence = self._calculate_confidence(confidence)

        return ReasoningResult(
            reasoning_type=ReasoningType.CAUSAL,
            content=conclusion,
            confidence=final_confidence,
            reasoning_steps=reasoning_steps,
            metadata={
                "events_count": len(events),
                "patterns_found": len(patterns),
                "sequences_built": len(sequences),
                "inferences_made": len(inferences),
                "temporal_coherence": confidence,
            },
        )

    def _extract_temporal_events(
        self, task: str, context: Dict[str, Any]
    ) -> List[TemporalEvent]:
        """Extract temporal events from task and context."""
        events = []

        # Check if context provides events directly
        if "events" in context:
            for i, event_data in enumerate(context["events"]):
                if isinstance(event_data, dict):
                    event = TemporalEvent(
                        id=event_data.get("id", f"event_{i}"),
                        content=event_data.get("content", ""),
                        start_time=self._parse_datetime(event_data.get("start_time")),
                        end_time=self._parse_datetime(event_data.get("end_time")),
                        metadata=event_data.get("metadata", {}),
                    )
                else:
                    event = TemporalEvent(
                        id=f"event_{i}",
                        content=str(event_data),
                        metadata={"source": "context"},
                    )
                events.append(event)

        # Extract events from task text (simplified)
        if not events:
            events = self._extract_events_from_text(task)

        return events

    def _extract_events_from_text(self, text: str) -> List[TemporalEvent]:
        """Extract events from text using simple patterns."""
        events = []

        # Look for temporal markers
        temporal_markers = [
            "then",
            "after",
            "before",
            "during",
            "while",
            "when",
            "since",
        ]
        sentences = text.split(".")

        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if sentence and any(
                marker in sentence.lower() for marker in temporal_markers
            ):
                event = TemporalEvent(
                    id=f"text_event_{i}",
                    content=sentence,
                    metadata={"source": "text_extraction", "sentence_index": i},
                )
                events.append(event)

        # If no temporal markers, create events from sentences
        if not events and sentences:
            for i, sentence in enumerate(sentences[:3]):  # Limit to first 3 sentences
                sentence = sentence.strip()
                if sentence:
                    event = TemporalEvent(
                        id=f"sentence_{i}",
                        content=sentence,
                        metadata={"source": "sentence_split", "index": i},
                    )
                    events.append(event)

        return events

    def _identify_patterns(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """Identify temporal patterns in events."""
        patterns = []

        if len(events) < 2:
            return patterns

        # Sequential pattern (common in narratives)
        if self._is_sequential_pattern(events):
            relations = []
            for i in range(len(events) - 1):
                relations.append(
                    (events[i].id, events[i + 1].id, TemporalRelation.BEFORE)
                )

            pattern = TemporalPattern(
                name="sequential", events=events, relations=relations, confidence=0.8
            )
            patterns.append(pattern)

        # Causal chain pattern
        if self._is_causal_chain(events):
            relations = []
            for i in range(len(events) - 1):
                relations.append(
                    (events[i].id, events[i + 1].id, TemporalRelation.BEFORE)
                )

            pattern = TemporalPattern(
                name="causal_chain",
                events=events,
                relations=relations,
                confidence=0.7,
                metadata={"pattern_type": "causal"},
            )
            patterns.append(pattern)

        return patterns

    def _build_sequences(self, events: List[TemporalEvent]) -> List[TemporalSequence]:
        """Build temporal sequences from events."""
        sequences = []

        if len(events) < 2:
            return sequences

        # Build simple chronological sequence
        ordering = []
        sorted_events = self._sort_events_chronologically(events)

        for i in range(len(sorted_events) - 1):
            ordering.append((sorted_events[i].id, sorted_events[i + 1].id))

        sequence = TemporalSequence(
            events=sorted_events,
            ordering=ordering,
            properties={"type": "chronological", "length": len(sorted_events)},
        )
        sequences.append(sequence)

        return sequences

    def _apply_temporal_rules(
        self, events: List[TemporalEvent], patterns: List[TemporalPattern]
    ) -> List[str]:
        """Apply temporal reasoning rules to generate inferences."""
        inferences = []

        # Rule: If A happens before B, and B causes C, then A happens before C
        for pattern in patterns:
            if pattern.name == "causal_chain" and len(pattern.events) >= 3:
                first_event = pattern.events[0]
                last_event = pattern.events[-1]
                inferences.append(
                    f"{first_event.content} ultimately leads to {last_event.content}"
                )

        # Rule: Sequential events form a process
        sequential_patterns = [p for p in patterns if p.name == "sequential"]
        if sequential_patterns:
            inferences.append("Events form a coherent sequential process")

        # Rule: Overlapping events indicate concurrency
        overlapping_events = self._find_overlapping_events(events)
        if overlapping_events:
            inferences.append(f"Found {len(overlapping_events)} concurrent event pairs")

        return inferences

    def _form_temporal_conclusion(
        self,
        events: List[TemporalEvent],
        patterns: List[TemporalPattern],
        sequences: List[TemporalSequence],
        inferences: List[str],
    ) -> str:
        """Form final temporal reasoning conclusion."""
        if not events:
            return "No temporal events identified for analysis"

        conclusion_parts = []
        conclusion_parts.append(f"Temporal analysis of {len(events)} events")

        if patterns:
            pattern_names = [p.name for p in patterns]
            conclusion_parts.append(f"identified patterns: {', '.join(pattern_names)}")

        if sequences:
            conclusion_parts.append(
                f"organized into {len(sequences)} temporal sequences"
            )

        if inferences:
            conclusion_parts.append(f"yielding {len(inferences)} temporal inferences")

        return "; ".join(conclusion_parts)

    def _calculate_temporal_confidence(
        self,
        events: List[TemporalEvent],
        patterns: List[TemporalPattern],
        sequences: List[TemporalSequence],
    ) -> float:
        """Calculate confidence based on temporal coherence."""
        if not events:
            return 0.0

        # Base confidence from number of events
        base_confidence = min(0.9, 0.3 + 0.1 * len(events))

        # Boost for identified patterns
        pattern_boost = 0.1 * len(patterns)

        # Boost for temporal coherence
        coherence_boost = 0.0
        if sequences:
            # Check if sequences are well-ordered
            for sequence in sequences:
                if len(sequence.ordering) > 0:
                    coherence_boost += 0.1

        total_confidence = base_confidence + pattern_boost + coherence_boost
        return min(1.0, total_confidence)

    def _parse_datetime(self, dt_string: Optional[str]) -> Optional[datetime]:
        """Parse datetime string."""
        if not dt_string:
            return None

        try:
            # Try common formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%H:%M:%S"]:
                try:
                    return datetime.strptime(dt_string, fmt)
                except ValueError:
                    continue
            return None
        except Exception:
            return None

    def _is_sequential_pattern(self, events: List[TemporalEvent]) -> bool:
        """Check if events form a sequential pattern."""
        # Simple heuristic: look for temporal markers
        sequential_markers = ["first", "then", "next", "after", "finally"]
        content_text = " ".join([e.content.lower() for e in events])
        return any(marker in content_text for marker in sequential_markers)

    def _is_causal_chain(self, events: List[TemporalEvent]) -> bool:
        """Check if events form a causal chain."""
        causal_markers = ["because", "caused", "led to", "resulted in", "due to"]
        content_text = " ".join([e.content.lower() for e in events])
        return any(marker in content_text for marker in causal_markers)

    def _sort_events_chronologically(
        self, events: List[TemporalEvent]
    ) -> List[TemporalEvent]:
        """Sort events chronologically."""
        # Sort by start_time if available, otherwise by text position
        events_with_time = [e for e in events if e.start_time is not None]
        events_without_time = [e for e in events if e.start_time is None]

        # Sort events with time
        events_with_time.sort(key=lambda e: e.start_time)

        # Combine with events without time (maintain original order)
        return events_with_time + events_without_time

    def _find_overlapping_events(
        self, events: List[TemporalEvent]
    ) -> List[Tuple[TemporalEvent, TemporalEvent]]:
        """Find overlapping events."""
        overlapping = []

        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                event1, event2 = events[i], events[j]

                # Check for temporal overlap
                if (
                    event1.start_time
                    and event1.end_time
                    and event2.start_time
                    and event2.end_time
                ):

                    # Check if time intervals overlap
                    if (
                        event1.start_time <= event2.end_time
                        and event2.start_time <= event1.end_time
                    ):
                        overlapping.append((event1, event2))

        return overlapping

    def _initialize_temporal_rules(self) -> List[TemporalRule]:
        """Initialize basic temporal reasoning rules."""
        rules = []

        # Transitivity rule
        rules.append(
            TemporalRule(
                name="transitivity",
                conditions=["A before B", "B before C"],
                conclusions=["A before C"],
                temporal_constraints=[
                    ("A", "B", TemporalRelation.BEFORE),
                    ("B", "C", TemporalRelation.BEFORE),
                ],
                confidence=0.9,
            )
        )

        # Causality rule
        rules.append(
            TemporalRule(
                name="causality",
                conditions=["A causes B"],
                conclusions=["A before B"],
                temporal_constraints=[("A", "B", TemporalRelation.BEFORE)],
                confidence=0.8,
            )
        )

        return rules
