"""
Phase 7 AI-Powered Agent Evolution.

Implements self-improving agents through reinforcement learning and dynamic capabilities:
- Self-improving agents through reinforcement learning
- Dynamic skill acquisition and transfer
- Automated agent specialization based on task patterns
- Performance-based agent composition optimization
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

logger = logging.getLogger("agentnet.core.evolution")


class LearningStrategy(str, Enum):
    """Learning strategies for agent evolution."""

    REINFORCEMENT = "reinforcement"
    IMITATION = "imitation"
    TRANSFER = "transfer"
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"


class SpecializationType(str, Enum):
    """Types of agent specialization."""

    TASK_BASED = "task_based"
    DOMAIN_BASED = "domain_based"
    SKILL_BASED = "skill_based"
    PERFORMANCE_BASED = "performance_based"


@dataclass
class Skill:
    """Represents an agent skill."""

    name: str
    description: str
    proficiency: float = 0.0  # 0.0 to 1.0
    usage_count: int = 0
    success_rate: float = 0.0
    last_used: float = field(default_factory=time.time)
    prerequisites: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskPattern:
    """Represents a pattern of tasks."""

    pattern_id: str
    task_type: str
    common_elements: List[str]
    frequency: int = 1
    success_rate: float = 0.0
    optimal_skills: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric for agent evaluation."""

    metric_name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningExperience:
    """A learning experience for reinforcement learning."""

    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillAcquisitionEngine:
    """Handles dynamic skill acquisition and transfer."""

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.skill_library: Dict[str, Skill] = {}
        self.transfer_relationships: Dict[str, List[str]] = {}
        self.acquisition_history: List[Dict[str, Any]] = []

    def acquire_skill(self, skill_name: str, task_context: Dict[str, Any]) -> bool:
        """Acquire a new skill based on task context."""
        if skill_name in self.skill_library:
            return self._improve_existing_skill(skill_name, task_context)

        # Create new skill
        skill = Skill(
            name=skill_name,
            description=task_context.get("description", f"Skill for {skill_name}"),
            proficiency=0.1,  # Start with low proficiency
            metadata={"acquired_from": task_context.get("task_type", "unknown")},
        )

        self.skill_library[skill_name] = skill

        # Record acquisition
        self.acquisition_history.append(
            {
                "skill": skill_name,
                "timestamp": time.time(),
                "context": task_context,
                "action": "acquired",
            }
        )

        logger.info(f"Acquired new skill: {skill_name}")
        return True

    def transfer_skill(
        self, source_skill: str, target_skill: str, similarity: float
    ) -> bool:
        """Transfer knowledge from one skill to another."""
        if source_skill not in self.skill_library:
            return False

        source = self.skill_library[source_skill]

        if target_skill not in self.skill_library:
            # Create target skill with transferred knowledge
            target = Skill(
                name=target_skill,
                description=f"Skill transferred from {source_skill}",
                proficiency=source.proficiency
                * similarity
                * 0.7,  # Reduced proficiency
                metadata={"transferred_from": source_skill, "similarity": similarity},
            )
            self.skill_library[target_skill] = target
        else:
            # Improve existing skill through transfer
            target = self.skill_library[target_skill]
            transfer_boost = source.proficiency * similarity * 0.3
            target.proficiency = min(1.0, target.proficiency + transfer_boost)

        # Record transfer relationship
        if source_skill not in self.transfer_relationships:
            self.transfer_relationships[source_skill] = []
        self.transfer_relationships[source_skill].append(target_skill)

        # Record in history
        self.acquisition_history.append(
            {
                "source_skill": source_skill,
                "target_skill": target_skill,
                "timestamp": time.time(),
                "similarity": similarity,
                "action": "transferred",
            }
        )

        logger.info(f"Transferred skill from {source_skill} to {target_skill}")
        return True

    def update_skill_performance(
        self, skill_name: str, success: bool, task_difficulty: float = 0.5
    ) -> None:
        """Update skill performance based on usage."""
        if skill_name not in self.skill_library:
            return

        skill = self.skill_library[skill_name]
        skill.usage_count += 1
        skill.last_used = time.time()

        # Update success rate
        current_successes = skill.success_rate * (skill.usage_count - 1)
        new_successes = current_successes + (1.0 if success else 0.0)
        skill.success_rate = new_successes / skill.usage_count

        # Update proficiency based on performance and difficulty
        if success:
            # Improve proficiency, more for harder tasks
            improvement = self.learning_rate * task_difficulty
            skill.proficiency = min(1.0, skill.proficiency + improvement)
        else:
            # Slight decrease for failures
            decrease = self.learning_rate * 0.1
            skill.proficiency = max(0.0, skill.proficiency - decrease)

    def identify_transferable_skills(
        self, new_task: str, task_context: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Identify skills that could transfer to a new task."""
        transferable = []

        for skill_name, skill in self.skill_library.items():
            similarity = self._calculate_task_similarity(skill, new_task, task_context)
            if similarity > 0.3:  # Threshold for transferability
                transferable.append((skill_name, similarity))

        # Sort by similarity
        transferable.sort(key=lambda x: x[1], reverse=True)
        return transferable

    def get_skill_recommendations(self, task_patterns: List[TaskPattern]) -> List[str]:
        """Recommend skills to acquire based on task patterns."""
        skill_priorities = {}

        for pattern in task_patterns:
            # High frequency patterns suggest important skills
            weight = pattern.frequency * pattern.success_rate

            for skill in pattern.optimal_skills:
                if skill not in self.skill_library:
                    skill_priorities[skill] = skill_priorities.get(skill, 0) + weight

        # Sort by priority
        recommended = sorted(skill_priorities.items(), key=lambda x: x[1], reverse=True)
        return [skill for skill, priority in recommended[:5]]  # Top 5 recommendations

    def _improve_existing_skill(
        self, skill_name: str, task_context: Dict[str, Any]
    ) -> bool:
        """Improve an existing skill."""
        skill = self.skill_library[skill_name]

        # Small improvement from practice
        improvement = self.learning_rate * 0.5
        skill.proficiency = min(1.0, skill.proficiency + improvement)

        # Record improvement
        self.acquisition_history.append(
            {
                "skill": skill_name,
                "timestamp": time.time(),
                "context": task_context,
                "action": "improved",
                "new_proficiency": skill.proficiency,
            }
        )

        return True

    def _calculate_task_similarity(
        self, skill: Skill, new_task: str, context: Dict[str, Any]
    ) -> float:
        """Calculate similarity between a skill and a new task."""
        # Simplified similarity calculation
        skill_context = skill.metadata.get("acquired_from", "")
        task_type = context.get("task_type", new_task)

        # Check for common words/concepts
        skill_words = set(skill_context.lower().split())
        task_words = set(task_type.lower().split())

        if not skill_words or not task_words:
            return 0.0

        intersection = len(skill_words.intersection(task_words))
        union = len(skill_words.union(task_words))

        return intersection / union if union > 0 else 0.0


class TaskPatternAnalyzer:
    """Analyzes task patterns for agent specialization."""

    def __init__(self, min_pattern_frequency: int = 3):
        self.min_pattern_frequency = min_pattern_frequency
        self.task_history: List[Dict[str, Any]] = []
        self.identified_patterns: Dict[str, TaskPattern] = {}

    def record_task(self, task_info: Dict[str, Any]) -> None:
        """Record a task for pattern analysis."""
        task_record = {
            "task_id": task_info.get("task_id", f"task_{len(self.task_history)}"),
            "task_type": task_info.get("task_type", "unknown"),
            "content": task_info.get("content", ""),
            "success": task_info.get("success", False),
            "duration": task_info.get("duration", 0.0),
            "skills_used": task_info.get("skills_used", []),
            "timestamp": time.time(),
            "metadata": task_info.get("metadata", {}),
        }

        self.task_history.append(task_record)

        # Trigger pattern analysis periodically
        if len(self.task_history) % 10 == 0:
            self._analyze_patterns()

    def get_specialization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for agent specialization."""
        recommendations = []

        for pattern_id, pattern in self.identified_patterns.items():
            if pattern.frequency >= self.min_pattern_frequency:
                recommendation = {
                    "specialization_type": self._determine_specialization_type(pattern),
                    "pattern": pattern,
                    "priority": pattern.frequency * pattern.success_rate,
                    "recommended_skills": pattern.optimal_skills,
                    "rationale": f"Pattern {pattern_id} occurs {pattern.frequency} times with {pattern.success_rate:.2f} success rate",
                }
                recommendations.append(recommendation)

        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        return recommendations

    def identify_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in task execution."""
        bottlenecks = []

        # Analyze recent tasks
        recent_tasks = (
            self.task_history[-50:]
            if len(self.task_history) > 50
            else self.task_history
        )

        # Group by task type
        task_types = {}
        for task in recent_tasks:
            task_type = task["task_type"]
            if task_type not in task_types:
                task_types[task_type] = []
            task_types[task_type].append(task)

        # Identify bottlenecks
        for task_type, tasks in task_types.items():
            if len(tasks) < 3:  # Need sufficient data
                continue

            success_rate = sum(t["success"] for t in tasks) / len(tasks)
            avg_duration = sum(t["duration"] for t in tasks) / len(tasks)

            if success_rate < 0.6:  # Low success rate
                bottlenecks.append(
                    {
                        "type": "low_success_rate",
                        "task_type": task_type,
                        "success_rate": success_rate,
                        "recommendation": "Skill improvement or specialization needed",
                    }
                )

            if avg_duration > 60.0:  # Long duration (in seconds)
                bottlenecks.append(
                    {
                        "type": "slow_execution",
                        "task_type": task_type,
                        "avg_duration": avg_duration,
                        "recommendation": "Process optimization needed",
                    }
                )

        return bottlenecks

    def _analyze_patterns(self) -> None:
        """Analyze task history for patterns."""
        # Group tasks by type
        task_groups = {}
        for task in self.task_history:
            task_type = task["task_type"]
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(task)

        # Identify patterns
        for task_type, tasks in task_groups.items():
            if len(tasks) >= self.min_pattern_frequency:
                pattern = self._create_pattern(task_type, tasks)
                self.identified_patterns[pattern.pattern_id] = pattern

    def _create_pattern(
        self, task_type: str, tasks: List[Dict[str, Any]]
    ) -> TaskPattern:
        """Create a task pattern from a group of tasks."""
        # Find common elements
        common_elements = self._find_common_elements(tasks)

        # Calculate success rate
        successful_tasks = [t for t in tasks if t["success"]]
        success_rate = len(successful_tasks) / len(tasks)

        # Find optimal skills
        optimal_skills = self._find_optimal_skills(successful_tasks)

        pattern = TaskPattern(
            pattern_id=f"pattern_{task_type}_{len(tasks)}",
            task_type=task_type,
            common_elements=common_elements,
            frequency=len(tasks),
            success_rate=success_rate,
            optimal_skills=optimal_skills,
            metadata={
                "avg_duration": sum(t["duration"] for t in tasks) / len(tasks),
                "first_seen": min(t["timestamp"] for t in tasks),
                "last_seen": max(t["timestamp"] for t in tasks),
            },
        )

        return pattern

    def _find_common_elements(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Find common elements across tasks."""
        if not tasks:
            return []

        # Simple approach: find common words in content
        all_words = set()
        word_counts = {}

        for task in tasks:
            words = task["content"].lower().split()
            all_words.update(words)

            for word in set(words):  # Unique words per task
                word_counts[word] = word_counts.get(word, 0) + 1

        # Find words that appear in most tasks
        threshold = max(1, len(tasks) * 0.6)  # 60% of tasks
        common_elements = [
            word for word, count in word_counts.items() if count >= threshold
        ]

        return common_elements[:10]  # Limit to top 10

    def _find_optimal_skills(self, successful_tasks: List[Dict[str, Any]]) -> List[str]:
        """Find skills that lead to successful task completion."""
        skill_counts = {}

        for task in successful_tasks:
            for skill in task.get("skills_used", []):
                skill_counts[skill] = skill_counts.get(skill, 0) + 1

        # Sort by frequency
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        return [skill for skill, count in sorted_skills[:5]]  # Top 5 skills

    def _determine_specialization_type(
        self, pattern: TaskPattern
    ) -> SpecializationType:
        """Determine the type of specialization for a pattern."""
        # Simple heuristics
        if pattern.frequency > 20:
            return SpecializationType.TASK_BASED
        elif len(pattern.optimal_skills) > 3:
            return SpecializationType.SKILL_BASED
        elif pattern.success_rate > 0.8:
            return SpecializationType.PERFORMANCE_BASED
        else:
            return SpecializationType.DOMAIN_BASED


class ReinforcementLearningEngine:
    """Reinforcement learning engine for agent self-improvement."""

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.experiences: List[LearningExperience] = []
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.policy_updates: List[Dict[str, Any]] = []

    def record_experience(self, experience: LearningExperience) -> None:
        """Record a learning experience."""
        self.experiences.append(experience)

        # Update Q-table
        self._update_q_value(experience)

        # Limit experience buffer size
        if len(self.experiences) > 1000:
            self.experiences = self.experiences[-1000:]

    def get_action_recommendation(
        self, state: Dict[str, Any], available_actions: List[str]
    ) -> str:
        """Get action recommendation based on current policy."""
        state_key = self._state_to_key(state)

        if state_key not in self.q_table:
            # Random action for unexplored states
            return available_actions[0] if available_actions else "default"

        # Get Q-values for available actions
        q_values = self.q_table[state_key]
        available_q_values = {
            action: q_values.get(action, 0.0) for action in available_actions
        }

        # Choose action with highest Q-value (with some exploration)
        if len(self.experiences) % 10 == 0:  # 10% exploration
            import random

            return random.choice(available_actions)
        else:
            return max(available_q_values.items(), key=lambda x: x[1])[0]

    def evaluate_policy_performance(self) -> Dict[str, Any]:
        """Evaluate current policy performance."""
        if not self.experiences:
            return {"status": "no_data"}

        recent_experiences = (
            self.experiences[-100:] if len(self.experiences) > 100 else self.experiences
        )

        # Calculate metrics
        total_reward = sum(exp.reward for exp in recent_experiences)
        avg_reward = total_reward / len(recent_experiences)
        success_rate = sum(1 for exp in recent_experiences if exp.reward > 0) / len(
            recent_experiences
        )

        # Analyze state-action performance
        state_action_performance = {}
        for exp in recent_experiences:
            state_key = self._state_to_key(exp.state)
            action = exp.action
            key = f"{state_key}::{action}"

            if key not in state_action_performance:
                state_action_performance[key] = {"rewards": [], "count": 0}

            state_action_performance[key]["rewards"].append(exp.reward)
            state_action_performance[key]["count"] += 1

        # Find best and worst performing state-action pairs
        performance_summary = []
        for key, data in state_action_performance.items():
            avg_reward = sum(data["rewards"]) / len(data["rewards"])
            performance_summary.append(
                {"state_action": key, "avg_reward": avg_reward, "count": data["count"]}
            )

        performance_summary.sort(key=lambda x: x["avg_reward"], reverse=True)

        return {
            "total_experiences": len(self.experiences),
            "recent_experiences": len(recent_experiences),
            "avg_reward": avg_reward,
            "success_rate": success_rate,
            "best_performing": performance_summary[:3],
            "worst_performing": performance_summary[-3:],
            "q_table_size": len(self.q_table),
        }

    def suggest_policy_improvements(self) -> List[Dict[str, Any]]:
        """Suggest improvements to the current policy."""
        suggestions = []

        if not self.experiences:
            return suggestions

        # Analyze Q-table for improvement opportunities
        for state_key, actions in self.q_table.items():
            if len(actions) < 2:
                continue

            # Find actions with low Q-values
            sorted_actions = sorted(actions.items(), key=lambda x: x[1])
            worst_action = sorted_actions[0]
            best_action = sorted_actions[-1]

            if best_action[1] - worst_action[1] > 0.5:  # Significant difference
                suggestions.append(
                    {
                        "type": "avoid_action",
                        "state": state_key,
                        "action_to_avoid": worst_action[0],
                        "preferred_action": best_action[0],
                        "q_value_difference": best_action[1] - worst_action[1],
                    }
                )

        # Find underexplored states
        state_visit_counts = {}
        for exp in self.experiences:
            state_key = self._state_to_key(exp.state)
            state_visit_counts[state_key] = state_visit_counts.get(state_key, 0) + 1

        avg_visits = (
            sum(state_visit_counts.values()) / len(state_visit_counts)
            if state_visit_counts
            else 0
        )
        for state_key, visits in state_visit_counts.items():
            if visits < avg_visits * 0.5:  # Underexplored
                suggestions.append(
                    {
                        "type": "explore_more",
                        "state": state_key,
                        "visit_count": visits,
                        "avg_visits": avg_visits,
                    }
                )

        return suggestions[:10]  # Limit to top 10 suggestions

    def _update_q_value(self, experience: LearningExperience) -> None:
        """Update Q-value based on experience."""
        state_key = self._state_to_key(experience.state)
        next_state_key = self._state_to_key(experience.next_state)

        # Initialize Q-table entries if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if experience.action not in self.q_table[state_key]:
            self.q_table[state_key][experience.action] = 0.0

        # Get max Q-value for next state
        max_next_q = 0.0
        if next_state_key in self.q_table and not experience.done:
            max_next_q = (
                max(self.q_table[next_state_key].values())
                if self.q_table[next_state_key]
                else 0.0
            )

        # Q-learning update
        current_q = self.q_table[state_key][experience.action]
        target_q = experience.reward + self.discount_factor * max_next_q
        new_q = current_q + self.learning_rate * (target_q - current_q)

        self.q_table[state_key][experience.action] = new_q

    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state dictionary to string key."""
        # Simplified state representation
        key_parts = []
        for key, value in sorted(state.items()):
            if isinstance(value, (int, float, bool, str)):
                key_parts.append(f"{key}:{value}")

        return "||".join(key_parts)


class AgentEvolutionManager:
    """Central manager for agent evolution and self-improvement."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.skill_engine = SkillAcquisitionEngine(
            learning_rate=config.get("learning_rate", 0.1)
        )
        self.pattern_analyzer = TaskPatternAnalyzer(
            min_pattern_frequency=config.get("min_pattern_frequency", 3)
        )
        self.rl_engine = ReinforcementLearningEngine(
            learning_rate=config.get("rl_learning_rate", 0.1),
            discount_factor=config.get("discount_factor", 0.9),
        )

        self.evolution_history: List[Dict[str, Any]] = []
        self.performance_metrics: List[PerformanceMetric] = []

    def evolve_agent(
        self, agent_id: str, task_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evolve an agent based on task results."""
        evolution_report = {
            "agent_id": agent_id,
            "timestamp": time.time(),
            "improvements": [],
            "new_skills": [],
            "specialization_changes": [],
        }

        # Record tasks for pattern analysis
        for task_result in task_results:
            self.pattern_analyzer.record_task(task_result)

        # Skill acquisition and improvement
        for task_result in task_results:
            skills_used = task_result.get("skills_used", [])
            task_success = task_result.get("success", False)
            task_difficulty = task_result.get("difficulty", 0.5)

            # Update existing skills
            for skill in skills_used:
                self.skill_engine.update_skill_performance(
                    skill, task_success, task_difficulty
                )
                evolution_report["improvements"].append(f"Updated skill: {skill}")

            # Identify new skills to acquire
            task_type = task_result.get("task_type", "unknown")
            if not task_success and task_type not in skills_used:
                # Failed task might need new skill
                acquired = self.skill_engine.acquire_skill(task_type, task_result)
                if acquired:
                    evolution_report["new_skills"].append(task_type)

        # Reinforcement learning
        for task_result in task_results:
            if "rl_experience" in task_result:
                experience = task_result["rl_experience"]
                self.rl_engine.record_experience(experience)

        # Specialization recommendations
        specialization_recs = self.pattern_analyzer.get_specialization_recommendations()
        if specialization_recs:
            evolution_report["specialization_changes"] = specialization_recs[
                :3
            ]  # Top 3

        # Record evolution
        self.evolution_history.append(evolution_report)

        return evolution_report

    def get_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Get current capabilities of an agent."""
        return {
            "skills": dict(self.skill_engine.skill_library),
            "skill_count": len(self.skill_engine.skill_library),
            "avg_proficiency": (
                sum(s.proficiency for s in self.skill_engine.skill_library.values())
                / len(self.skill_engine.skill_library)
                if self.skill_engine.skill_library
                else 0.0
            ),
            "most_used_skills": sorted(
                self.skill_engine.skill_library.items(),
                key=lambda x: x[1].usage_count,
                reverse=True,
            )[:5],
            "transfer_relationships": self.skill_engine.transfer_relationships,
            "rl_performance": self.rl_engine.evaluate_policy_performance(),
        }

    def recommend_agent_improvements(self, agent_id: str) -> Dict[str, Any]:
        """Recommend improvements for an agent."""
        recommendations = {
            "skill_recommendations": [],
            "specialization_recommendations": [],
            "policy_improvements": [],
            "performance_bottlenecks": [],
        }

        # Skill recommendations
        task_patterns = list(self.pattern_analyzer.identified_patterns.values())
        skill_recs = self.skill_engine.get_skill_recommendations(task_patterns)
        recommendations["skill_recommendations"] = skill_recs

        # Specialization recommendations
        spec_recs = self.pattern_analyzer.get_specialization_recommendations()
        recommendations["specialization_recommendations"] = spec_recs

        # Policy improvements
        policy_improvements = self.rl_engine.suggest_policy_improvements()
        recommendations["policy_improvements"] = policy_improvements

        # Performance bottlenecks
        bottlenecks = self.pattern_analyzer.identify_performance_bottlenecks()
        recommendations["performance_bottlenecks"] = bottlenecks

        return recommendations

    def save_evolution_state(self, filepath: str) -> None:
        """Save the evolution state to file."""
        state = {
            "skill_library": {
                name: {
                    "name": skill.name,
                    "description": skill.description,
                    "proficiency": skill.proficiency,
                    "usage_count": skill.usage_count,
                    "success_rate": skill.success_rate,
                    "last_used": skill.last_used,
                    "prerequisites": skill.prerequisites,
                    "metadata": skill.metadata,
                }
                for name, skill in self.skill_engine.skill_library.items()
            },
            "transfer_relationships": self.skill_engine.transfer_relationships,
            "identified_patterns": {
                pid: {
                    "pattern_id": pattern.pattern_id,
                    "task_type": pattern.task_type,
                    "common_elements": pattern.common_elements,
                    "frequency": pattern.frequency,
                    "success_rate": pattern.success_rate,
                    "optimal_skills": pattern.optimal_skills,
                    "metadata": pattern.metadata,
                }
                for pid, pattern in self.pattern_analyzer.identified_patterns.items()
            },
            "q_table": self.rl_engine.q_table,
            "evolution_history": self.evolution_history[-100:],  # Last 100 entries
            "config": self.config,
            "timestamp": time.time(),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_evolution_state(self, filepath: str) -> bool:
        """Load evolution state from file."""
        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            # Restore skill library
            for name, skill_data in state.get("skill_library", {}).items():
                skill = Skill(
                    name=skill_data["name"],
                    description=skill_data["description"],
                    proficiency=skill_data["proficiency"],
                    usage_count=skill_data["usage_count"],
                    success_rate=skill_data["success_rate"],
                    last_used=skill_data["last_used"],
                    prerequisites=skill_data["prerequisites"],
                    metadata=skill_data["metadata"],
                )
                self.skill_engine.skill_library[name] = skill

            # Restore other components
            self.skill_engine.transfer_relationships = state.get(
                "transfer_relationships", {}
            )

            # Restore patterns
            for pid, pattern_data in state.get("identified_patterns", {}).items():
                pattern = TaskPattern(
                    pattern_id=pattern_data["pattern_id"],
                    task_type=pattern_data["task_type"],
                    common_elements=pattern_data["common_elements"],
                    frequency=pattern_data["frequency"],
                    success_rate=pattern_data["success_rate"],
                    optimal_skills=pattern_data["optimal_skills"],
                    metadata=pattern_data["metadata"],
                )
                self.pattern_analyzer.identified_patterns[pid] = pattern

            # Restore Q-table
            self.rl_engine.q_table = state.get("q_table", {})

            # Restore history
            self.evolution_history = state.get("evolution_history", [])

            logger.info(f"Loaded evolution state from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load evolution state: {e}")
            return False
