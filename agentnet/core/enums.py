"""
Core enums for AgentNet problem-solving modes, styles, and techniques.

This module defines the enums for problem-solving modes, styles, and techniques
that can be used to adapt AgentNet behavior and AutoConfig parameters.

Enhanced enums include descriptions for clarity and new categories for advanced
agentic system design, such as cognitive processes, interaction topologies,
and failure recovery policies.
"""

from __future__ import annotations

from enum import Enum


class DescribedEnum(str, Enum):
    """
    An Enum class where each member has a value and a description.
    This enhances base enums by making them self-documenting.
    """

    def __new__(cls, value: str, description: str = ""):
        """Create a new instance of the enum member."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    @classmethod
    def choices(cls) -> list[tuple[str, str]]:
        """Returns a list of (value, description) tuples for all members."""
        return [(item.value, item.description) for item in cls]

    def __str__(self) -> str:
        """Return the value of the enum member."""
        return self.value


# --- Orchestration and Interaction Enums ---


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


# --- Agent Behavior Enums ---


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


# --- Task and Problem Enums ---


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


# --- System and Reliability Enums ---


class FailureRecoveryPolicy(DescribedEnum):
    """Defines the strategy for handling failures or errors during task execution."""

    RETRY = ("retry", "Attempt the failed operation again a fixed number of times.")
    RETRY_WITH_BACKOFF = ("retry_with_backoff", "Retry the operation with exponentially increasing delays between attempts.")
    FAILOVER = ("failover", "Switch to a backup or alternative component, agent, or process.")
    HUMAN_IN_THE_LOOP = ("human_in_the_loop", "Escalate the failure to a human operator for intervention and decision-making.")
    IGNORE = ("ignore", "Continue execution, ignoring the failure (use with caution).")
    TERMINATE = ("terminate", "Stop the entire process or workflow upon encountering a failure.")
