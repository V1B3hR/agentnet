"""
Core enums for AgentNet problem-solving modes, styles, and techniques.

This module defines enriched enums for problem-solving modes, cognitive processes,
interaction patterns, and reliability strategies. Each enum member carries both
a value and a human-readable description to enable self-documenting configuration
and dynamic UI generation (e.g., dropdowns, tooltips, docs export).

Enhancements in this revision:
- Unified DescribedEnum with rich helper/inspection utilities
- Safe lookup helpers (from_value / from_description / get)
- Introspection utilities (values(), names(), descriptions(), choices(), mapping())
- Serialization helpers (to_dict(), to_tuple(), as_markdown_table(), json schema-ish export)
- Membership support for raw string values (e.g. `'brainstorm' in Mode`)
- Markdown & structured documentation generation across all enums
- Aggregated metadata export (all_enum_metadata, export_all_enums_markdown)
- Defensive typing & explicit __all__
"""

from __future__ import annotations

from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Optional, Tuple, Type, TypeVar, Union, overload

TEnum = TypeVar("TEnum", bound="DescribedEnum")

# ---------------------------------------------------------------------------
# Base Enum
# ---------------------------------------------------------------------------


class DescribedEnum(str, Enum):
    """
    An Enum class where each member has a value and a description.

    Enhancements:
    - Each member stores a `description` attribute
    - Utility classmethods for choices, lookups, serialization, and docs
    - Graceful membership testing with raw strings (e.g. `'retry' in FailureRecoveryPolicy`)
    """

    # Allow subclasses to define optional synonyms mapping if desired
    _synonyms_: ClassVar[Dict[str, str]] = {}

    def __new__(cls, value: str, description: str = ""):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    # ------------------------------------------------------------------
    # Representation & coercion
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}(value='{self.value}', description='{self.description}')"

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        """Returns a list of (value, description) tuples suitable for form choices."""
        return [(item.value, item.description) for item in cls]

    @classmethod
    def values(cls) -> List[str]:
        return [m.value for m in cls]

    @classmethod
    def names(cls) -> List[str]:
        return [m.name for m in cls]

    @classmethod
    def descriptions(cls) -> List[str]:
        return [m.description for m in cls]

    @classmethod
    def mapping(cls) -> Dict[str, str]:
        """Value -> description mapping."""
        return {m.value: m.description for m in cls}

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    @classmethod
    def get(cls: Type[TEnum], value: Union[str, TEnum]) -> TEnum:
        """
        Strict lookup by value or member. Raises KeyError if not found.
        Synonyms (if defined) are resolved.
        """
        if isinstance(value, cls):
            return value
        if value in cls._synonyms_:
            value = cls._synonyms_[value]
        try:
            return cls(value)  # type: ignore[arg-type]
        except ValueError:
            raise KeyError(f"{value!r} is not a valid {cls.__name__}") from None

    @classmethod
    def from_value(cls: Type[TEnum], value: Union[str, TEnum], default: Optional[TEnum] = None) -> Optional[TEnum]:
        """
        Safe lookup; returns default (None unless specified) if invalid.
        """
        try:
            return cls.get(value)
        except KeyError:
            return default

    @classmethod
    def from_description(cls: Type[TEnum], description_substring: str, case_insensitive: bool = True) -> List[TEnum]:
        """
        Returns members whose description contains the given substring.
        """
        needle = description_substring.lower() if case_insensitive else description_substring
        results: List[TEnum] = []
        for m in cls:
            hay = m.description.lower() if case_insensitive else m.description
            if needle in hay:
                results.append(m)
        return results

    # ------------------------------------------------------------------
    # Introspection & membership
    # ------------------------------------------------------------------
    def to_tuple(self) -> Tuple[str, str]:
        return (self.value, self.description)

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "value": self.value, "description": self.description}

    @classmethod
    def list_dicts(cls) -> List[Dict[str, str]]:
        return [m.to_dict() for m in cls]

    @classmethod
    def as_markdown_table(cls, sort: bool = False) -> str:
        """
        Produce a Markdown table documenting the enum.
        """
        header = f"### {cls.__name__}\n\n| Name | Value | Description |\n|------|-------|-------------|"
        members = list(cls)
        if sort:
            members = sorted(members, key=lambda m: m.value)
        rows = [f"| {m.name} | `{m.value}` | {m.description} |" for m in members]
        return header + "\n" + "\n".join(rows) + "\n"

    @classmethod
    def json_schema_fragment(cls) -> Dict[str, Any]:
        """
        Return a JSON-schema-like fragment for documentation or validation contexts.
        """
        return {
            "title": cls.__name__,
            "type": "string",
            "enum": cls.values(),
            "description": f"Enum: {cls.__name__}",
            "x-enumDescriptions": {m.value: m.description for m in cls},
        }

    # Membership test enhancement: `'value' in EnumClass`
    def __hash__(self) -> int:  # explicit for mypy clarity
        return super().__hash__()

    @classmethod
    def __contains__(cls, item: object) -> bool:  # type: ignore[override]
        if isinstance(item, cls):
            return True
        if isinstance(item, str):
            if item in cls.values():
                return True
            # Accept synonyms
            if item in cls._synonyms_ and cls._synonyms_[item] in cls.values():
                return True
        return False

    # Iterator alias if needed externally
    @classmethod
    def iter(cls) -> Iterator["DescribedEnum"]:
        return iter(cls)


# ---------------------------------------------------------------------------
# Orchestration and Interaction Enums
# ---------------------------------------------------------------------------


class Mode(DescribedEnum):
    """High-level orchestration modes for multi-agent collaboration."""
    BRAINSTORM = ("brainstorm", "Free-form idea generation among multiple agents.")
    DEBATE = ("debate", "Agents take opposing viewpoints to challenge assumptions and find the best solution.")
    CONSENSUS = ("consensus", "Agents work collaboratively to reach a unified agreement or synthesis of ideas.")
    WORKFLOW = ("workflow", "Agents execute a predefined, structured sequence of tasks.")
    DIALOGUE = ("dialogue", "A single agent interacts with a user or another agent in a conversational manner.")
    SIMULATION = ("simulation", "Agents interact within a simulated environment to test hypotheses or predict outcomes.")
    RED_TEAMING = ("red_teaming", "One or more agents act as adversaries to find flaws, biases, or vulnerabilities in a plan.")
    AUCTION = ("auction", "Agents bid to take on tasks or allocate resources, enabling dynamic task distribution.")


class InteractionTopology(DescribedEnum):
    """Defines the communication structure in a multi-agent system."""
    HIERARCHY = ("hierarchy", "Tree-like structure with a clear chain of command and reporting.")
    FLAT_PEER_TO_PEER = ("flat_peer_to_peer", "All agents can communicate directly with all other agents.")
    HUB_AND_SPOKE = ("hub_and_spoke", "A central agent coordinates communication and tasks for peripheral agents.")
    CELLULAR = ("cellular", "Small, independent groups of agents that interact with each other as a single unit.")
    BROADCAST = ("broadcast", "One agent sends information to all others without expecting a direct reply.")


# ---------------------------------------------------------------------------
# Agent Behavior Enums
# ---------------------------------------------------------------------------


class ProblemSolvingStyle(DescribedEnum):
    """Defines the behavioral persona or role an agent adopts."""
    CLARIFIER = ("clarifier", "Focuses on asking questions, defining the problem space, and ensuring clarity.")
    IDEATOR = ("ideator", "Generates a wide range of creative and novel ideas and potential solutions.")
    DEVELOPER = ("developer", "Refines and builds upon existing ideas to make them more practical and robust.")
    IMPLEMENTOR = ("implementor", "Focuses on the practical steps, execution, and resource planning of a solution.")
    EVALUATOR = ("evaluator", "Critically assesses ideas and plans for viability, risks, and unintended consequences.")
    SYNTHESIZER = ("synthesizer", "Combines multiple disparate ideas and perspectives into a coherent, unified whole.")


class CognitiveProcess(DescribedEnum):
    """Defines the primary mode of thinking or reasoning an agent should employ."""
    ANALYTICAL_REASONING = ("analytical_reasoning", "Logical deduction, breaking down components, and structured analysis.")
    CREATIVE_THINKING = ("creative_thinking", "Generating novel, imaginative, and unconventional ideas (lateral thinking).")
    CRITICAL_EVALUATION = ("critical_evaluation", "Objectively assessing information, arguments, and sources for validity and bias.")
    SYSTEMS_THINKING = ("systems_thinking", "Understanding how parts of a complex system interrelate and influence each other.")
    STRATEGIC_FORESIGHT = ("strategic_foresight", "Anticipating future trends, scenarios, and their implications to plan effectively.")
    META_COGNITION = ("meta_cognition", "Reflecting on, monitoring, and controlling one's own thinking and learning processes.")


# ---------------------------------------------------------------------------
# Task and Problem Enums
# ---------------------------------------------------------------------------


class ProblemTechnique(DescribedEnum):
    """Specific techniques or frameworks for approaching a problem."""
    SIMPLE = ("simple", "Direct, single-step problem solving for straightforward issues.")
    COMPLEX = ("complex", "Multi-step, interdependent problem solving for intricate challenges.")
    TROUBLESHOOTING = ("troubleshooting", "Diagnosing and resolving faults or issues in a pre-existing system.")
    GAP_FROM_STANDARD = ("gap_from_standard", "Identifying and closing the gap between a current state and a desired standard.")
    TARGET_STATE = ("target_state", "Defining a future goal state and working backward to create a plan from the present.")
    OPEN_ENDED = ("open_ended", "Exploring a broad problem with no single correct answer, focusing on exploration.")
    ROOT_CAUSE_ANALYSIS = ("root_cause_analysis", "Systematically identifying the fundamental cause of a problem (e.g., 5 Whys).")
    MECE_ANALYSIS = ("mece_analysis", "Breaking a problem into 'Mutually Exclusive, Collectively Exhaustive' components.")
    SCENARIO_PLANNING = ("scenario_planning", "Developing plans and strategies for multiple potential future scenarios.")


class TaskDecompositionStrategy(DescribedEnum):
    """Methods for breaking down a large, complex task into smaller, manageable sub-tasks."""
    SEQUENTIAL = ("sequential", "Tasks are executed one after another in a linear, predefined order.")
    PARALLEL = ("parallel", "Multiple tasks that do not depend on each other are executed simultaneously.")
    HIERARCHICAL = ("hierarchical", "A main task is broken down into sub-tasks, which may be further broken down.")
    RECURSIVE = ("recursive", "A task is broken down into smaller instances of the same task until a base case is met.")
    CONDITION_BASED = ("condition_based", "The next task or workflow branch is determined by the outcome of the previous one.")


# ---------------------------------------------------------------------------
# System and Reliability Enums
# ---------------------------------------------------------------------------


class FailureRecoveryPolicy(DescribedEnum):
    """Defines the strategy for handling failures or errors during task execution."""
    RETRY = ("retry", "Attempt the failed operation again a fixed number of times.")
    RETRY_WITH_BACKOFF = ("retry_with_backoff", "Retry the operation with exponentially increasing delays between attempts.")
    FAILOVER = ("failover", "Switch to a backup or alternative component, agent, or process.")
    HUMAN_IN_THE_LOOP = ("human_in_the_loop", "Escalate the failure to a human operator for intervention and decision-making.")
    IGNORE = ("ignore", "Continue execution, ignoring the failure (use with caution).")
    TERMINATE = ("terminate", "Stop the entire process or workflow upon encountering a failure.")


# ---------------------------------------------------------------------------
# Aggregation / Documentation Utilities
# ---------------------------------------------------------------------------


def _all_described_enum_subclasses() -> List[Type[DescribedEnum]]:
    """
    Collect all direct subclasses of DescribedEnum defined in this module.
    (Does not recursively traverse subclass hierarchies beyond one level;
    adjust if deep subclassing is added in future.)
    """
    enums: List[Type[DescribedEnum]] = []
    for obj in globals().values():
        if isinstance(obj, type) and issubclass(obj, DescribedEnum) and obj is not DescribedEnum:
            enums.append(obj)
    # Stable order: by class name
    return sorted(enums, key=lambda c: c.__name__)


def all_enum_metadata() -> Dict[str, Any]:
    """
    Returns a dictionary containing metadata for every enum:
    {
       "Mode": {
           "members": [{"name": "...", "value": "...", "description": "..."}],
           "schema": {...}
       },
       ...
    }
    """
    meta: Dict[str, Any] = {}
    for enum_cls in _all_described_enum_subclasses():
        meta[enum_cls.__name__] = {
            "members": enum_cls.list_dicts(),
            "schema": enum_cls.json_schema_fragment(),
        }
    return meta


def export_all_enums_markdown(sort_members: bool = False) -> str:
    """
    Returns a combined Markdown documentation string for all enums.
    """
    sections = ["# AgentNet Enum Reference\n"]
    for enum_cls in _all_described_enum_subclasses():
        sections.append(enum_cls.as_markdown_table(sort=sort_members))
    return "\n".join(sections).rstrip() + "\n"


# ---------------------------------------------------------------------------
# __all__ for controlled exports
# ---------------------------------------------------------------------------

__all__ = [
    # Base
    "DescribedEnum",
    # Enums
    "Mode",
    "InteractionTopology",
    "ProblemSolvingStyle",
    "CognitiveProcess",
    "ProblemTechnique",
    "TaskDecompositionStrategy",
    "FailureRecoveryPolicy",
    # Utilities
    "all_enum_metadata",
    "export_all_enums_markdown",
]
