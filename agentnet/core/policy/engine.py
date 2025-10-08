"""
Policy Engine implementation.

Core policy evaluation engine that applies rules to agent actions and outputs.
Supports different policy actions like allow, block, transform, and log.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from .rules import ConstraintRule, RuleResult, Severity

logger = logging.getLogger("agentnet.policy.engine")


class PolicyAction(str, Enum):
    """Actions that can be taken based on policy evaluation."""

    ALLOW = "allow"  # Allow the action to proceed
    BLOCK = "block"  # Block the action completely
    TRANSFORM = "transform"  # Transform/modify the content
    LOG = "log"  # Log the action but allow it
    REQUIRE_APPROVAL = "require_approval"  # Human approval required


@dataclass
class PolicyResult:
    """Result of policy evaluation."""

    action: PolicyAction
    passed: bool
    rule_results: List[RuleResult] = field(default_factory=list)
    violations: List[RuleResult] = field(default_factory=list)
    transformed_content: Optional[str] = None
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time: float = 0.0

    @property
    def violation_count(self) -> int:
        """Number of rule violations."""
        return len(self.violations)

    @property
    def highest_severity(self) -> Optional[Severity]:
        """Highest severity among violations."""
        if not self.violations:
            return None

        severity_order = {Severity.MINOR: 1, Severity.MAJOR: 2, Severity.SEVERE: 3}
        return max(self.violations, key=lambda v: severity_order[v.severity]).severity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "action": self.action.value,
            "passed": self.passed,
            "violation_count": self.violation_count,
            "highest_severity": (
                self.highest_severity.value if self.highest_severity else None
            ),
            "rule_results": [r.to_dict() for r in self.rule_results],
            "violations": [v.to_dict() for v in self.violations],
            "transformed_content": self.transformed_content,
            "explanation": self.explanation,
            "metadata": self.metadata,
            "evaluation_time": self.evaluation_time,
        }


class PolicyEngine:
    """
    Core policy evaluation engine.

    Evaluates agent actions and outputs against a set of configurable rules.
    Supports different matching strategies and action policies.
    """

    def __init__(
        self,
        rules: Optional[List[ConstraintRule]] = None,
        default_action: PolicyAction = PolicyAction.ALLOW,
        strict_mode: bool = False,
        enable_transformations: bool = True,
        max_violations: int = 5,
        name: str = "default",
    ):
        """
        Initialize the policy engine.

        Args:
            rules: List of constraint rules to evaluate
            default_action: Default action when no rules match
            strict_mode: If True, any violation results in BLOCK
            enable_transformations: Whether to apply content transformations
            max_violations: Maximum violations before auto-blocking
            name: Name identifier for this policy engine
        """
        self.rules: List[ConstraintRule] = rules or []
        self.default_action = default_action
        self.strict_mode = strict_mode
        self.enable_transformations = enable_transformations
        self.max_violations = max_violations
        self.name = name

        # Statistics
        self.evaluation_count = 0
        self.violation_count = 0
        self.blocked_count = 0
        self.transformed_count = 0
        self.created_time = time.time()

        # Rule lookup optimization
        self._rules_by_tag: Dict[str, List[ConstraintRule]] = {}
        self._rebuild_tag_index()

        logger.info(f"PolicyEngine '{name}' initialized with {len(self.rules)} rules")

    def add_rule(self, rule: ConstraintRule) -> None:
        """Add a new rule to the engine."""
        if any(r.name == rule.name for r in self.rules):
            logger.warning(f"Rule with name '{rule.name}' already exists, replacing")
            self.rules = [r for r in self.rules if r.name != rule.name]

        self.rules.append(rule)
        self._rebuild_tag_index()
        logger.info(f"Added rule '{rule.name}' to policy engine '{self.name}'")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name."""
        initial_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != rule_name]

        if len(self.rules) < initial_count:
            self._rebuild_tag_index()
            logger.info(f"Removed rule '{rule_name}' from policy engine '{self.name}'")
            return True

        logger.warning(f"Rule '{rule_name}' not found in policy engine '{self.name}'")
        return False

    def get_rule(self, rule_name: str) -> Optional[ConstraintRule]:
        """Get a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None

    def enable_rule(self, rule_name: str) -> bool:
        """Enable a rule by name."""
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = True
            logger.info(f"Enabled rule '{rule_name}' in policy engine '{self.name}'")
            return True
        return False

    def disable_rule(self, rule_name: str) -> bool:
        """Disable a rule by name."""
        rule = self.get_rule(rule_name)
        if rule:
            rule.enabled = False
            logger.info(f"Disabled rule '{rule_name}' in policy engine '{self.name}'")
            return True
        return False

    def evaluate(
        self,
        context: Dict[str, Any],
        tags: Optional[List[str]] = None,
        rules_subset: Optional[List[str]] = None,
    ) -> PolicyResult:
        """
        Evaluate context against policy rules.

        Args:
            context: Context to evaluate (should contain 'content', 'agent_name', etc.)
            tags: Only evaluate rules with these tags
            rules_subset: Only evaluate rules with these names

        Returns:
            PolicyResult with evaluation outcome and recommended action
        """
        start_time = time.time()
        self.evaluation_count += 1

        # Determine which rules to evaluate
        rules_to_evaluate = self._select_rules(tags, rules_subset)

        # Evaluate all applicable rules
        rule_results = []
        violations = []

        for rule in rules_to_evaluate:
            try:
                result = rule.evaluate(context)
                rule_results.append(result)

                if not result.passed:
                    violations.append(result)

            except Exception as e:
                logger.error(f"Error evaluating rule '{rule.name}': {e}")
                # Create an error result
                error_result = RuleResult(
                    rule_name=rule.name,
                    passed=False,
                    severity=Severity.MAJOR,
                    error=str(e),
                    rationale=f"Rule evaluation failed: {e}",
                )
                rule_results.append(error_result)
                violations.append(error_result)

        # Determine policy action
        action, explanation = self._determine_action(violations, context)

        # Apply transformations if needed
        transformed_content = None
        if action == PolicyAction.TRANSFORM and self.enable_transformations:
            transformed_content = self._apply_transformations(context, violations)

        # Update statistics
        if violations:
            self.violation_count += len(violations)
        if action == PolicyAction.BLOCK:
            self.blocked_count += 1
        if transformed_content:
            self.transformed_count += 1

        evaluation_time = time.time() - start_time

        result = PolicyResult(
            action=action,
            passed=(
                action in [PolicyAction.ALLOW, PolicyAction.LOG, PolicyAction.TRANSFORM]
            ),
            rule_results=rule_results,
            violations=violations,
            transformed_content=transformed_content,
            explanation=explanation,
            metadata={
                "rules_evaluated": len(rules_to_evaluate),
                "engine_name": self.name,
                "strict_mode": self.strict_mode,
            },
            evaluation_time=evaluation_time,
        )

        logger.debug(
            f"Policy evaluation completed: {action.value}, "
            f"{len(violations)} violations, {evaluation_time:.3f}s"
        )

        return result

    def _select_rules(
        self, tags: Optional[List[str]] = None, rules_subset: Optional[List[str]] = None
    ) -> List[ConstraintRule]:
        """Select which rules to evaluate based on tags and subset."""
        if rules_subset:
            # Use specific rule names
            return [r for r in self.rules if r.name in rules_subset and r.enabled]

        if tags:
            # Use rules that match any of the specified tags
            matching_rules = set()
            for tag in tags:
                matching_rules.update(self._rules_by_tag.get(tag, []))
            return [r for r in matching_rules if r.enabled]

        # Use all enabled rules
        return [r for r in self.rules if r.enabled]

    def _determine_action(
        self, violations: List[RuleResult], context: Dict[str, Any]
    ) -> tuple[PolicyAction, str]:
        """Determine the policy action based on violations."""
        if not violations:
            return PolicyAction.ALLOW, "No policy violations detected"

        # In strict mode, any violation blocks
        if self.strict_mode:
            return (
                PolicyAction.BLOCK,
                f"Strict mode: {len(violations)} violations detected",
            )

        # Check if we exceed max violations
        if len(violations) > self.max_violations:
            return (
                PolicyAction.BLOCK,
                f"Too many violations: {len(violations)} > {self.max_violations}",
            )

        # Check severity levels
        severe_violations = [v for v in violations if v.severity == Severity.SEVERE]
        if severe_violations:
            return (
                PolicyAction.BLOCK,
                f"Severe violations: {[v.rule_name for v in severe_violations]}",
            )

        major_violations = [v for v in violations if v.severity == Severity.MAJOR]
        if len(major_violations) >= 2:
            return (
                PolicyAction.BLOCK,
                f"Multiple major violations: {[v.rule_name for v in major_violations]}",
            )

        # Check if content can be transformed
        if self.enable_transformations and self._can_transform(violations, context):
            return (
                PolicyAction.TRANSFORM,
                f"Content can be transformed to fix {len(violations)} violations",
            )

        # Major violations might require approval
        if major_violations:
            return (
                PolicyAction.REQUIRE_APPROVAL,
                f"Major violations require approval: {[v.rule_name for v in major_violations]}",
            )

        # Minor violations can be logged
        return (
            PolicyAction.LOG,
            f"Minor violations logged: {[v.rule_name for v in violations]}",
        )

    def _can_transform(
        self, violations: List[RuleResult], context: Dict[str, Any]
    ) -> bool:
        """Check if violations can be fixed through content transformation."""
        # Simple heuristic: can transform keyword and length violations
        transformable_tags = ["keyword", "length", "content"]

        for violation in violations:
            rule = self.get_rule(violation.rule_name)
            if not rule or not any(tag in transformable_tags for tag in rule.tags):
                return False

        return True

    def _apply_transformations(
        self, context: Dict[str, Any], violations: List[RuleResult]
    ) -> Optional[str]:
        """Apply content transformations to fix violations."""
        content = str(context.get("content", ""))
        transformed = content

        for violation in violations:
            rule = self.get_rule(violation.rule_name)
            if not rule:
                continue

            # Apply transformation based on rule type
            if "keyword" in rule.tags:
                transformed = self._transform_keywords(transformed, violation)
            elif "length" in rule.tags:
                transformed = self._transform_length(transformed, violation)

        return transformed if transformed != content else None

    def _transform_keywords(self, content: str, violation: RuleResult) -> str:
        """Transform content to remove prohibited keywords."""
        # Simple redaction - replace with [REDACTED]
        if violation.rationale and "Found prohibited keywords:" in violation.rationale:
            # Extract keywords from rationale
            keywords_str = violation.rationale.split("Found prohibited keywords: ")[1]
            keywords = eval(keywords_str)  # Note: In production, use proper parsing

            transformed = content
            for keyword in keywords:
                # Replace with [REDACTED] while preserving word boundaries
                pattern = r"\b" + re.escape(keyword) + r"\b"
                transformed = re.sub(
                    pattern, "[REDACTED]", transformed, flags=re.IGNORECASE
                )

            return transformed

        return content

    def _transform_length(self, content: str, violation: RuleResult) -> str:
        """Transform content to fix length violations."""
        if violation.rationale and "too long:" in violation.rationale:
            # Extract max length from rationale
            parts = violation.rationale.split("> ")
            if len(parts) > 1:
                max_length = int(parts[1])
                if len(content) > max_length:
                    return content[: max_length - 10] + "... [TRUNCATED]"

        return content

    def _rebuild_tag_index(self):
        """Rebuild the tag-based rule index for faster lookups."""
        self._rules_by_tag.clear()
        for rule in self.rules:
            for tag in rule.tags:
                if tag not in self._rules_by_tag:
                    self._rules_by_tag[tag] = []
                self._rules_by_tag[tag].append(rule)

    def get_stats(self) -> Dict[str, Any]:
        """Get policy engine statistics."""
        uptime = time.time() - self.created_time

        return {
            "name": self.name,
            "uptime": uptime,
            "rules_count": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules if r.enabled),
            "evaluation_count": self.evaluation_count,
            "violation_count": self.violation_count,
            "blocked_count": self.blocked_count,
            "transformed_count": self.transformed_count,
            "violation_rate": (
                self.violation_count / self.evaluation_count
                if self.evaluation_count > 0
                else 0.0
            ),
            "block_rate": (
                self.blocked_count / self.evaluation_count
                if self.evaluation_count > 0
                else 0.0
            ),
            "config": {
                "default_action": self.default_action.value,
                "strict_mode": self.strict_mode,
                "enable_transformations": self.enable_transformations,
                "max_violations": self.max_violations,
            },
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.evaluation_count = 0
        self.violation_count = 0
        self.blocked_count = 0
        self.transformed_count = 0

        # Reset rule stats too
        for rule in self.rules:
            rule.reset_stats()

        logger.info(f"Reset statistics for policy engine '{self.name}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert policy engine to dictionary representation."""
        return {
            "name": self.name,
            "config": {
                "default_action": self.default_action.value,
                "strict_mode": self.strict_mode,
                "enable_transformations": self.enable_transformations,
                "max_violations": self.max_violations,
            },
            "rules": [rule.to_dict() for rule in self.rules],
            "stats": self.get_stats(),
            "tags": list(self._rules_by_tag.keys()),
        }

    def evaluate_agent_orchestration(
        self, agents: List[Dict[str, Any]], coordination_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate orchestration policies for multi-agent coordination.

        Args:
            agents: List of agent information dicts
            coordination_context: Context for coordination decisions

        Returns:
            Orchestration result with allowed agents and coordination rules
        """
        orchestration_result = {
            "allowed_agents": [],
            "blocked_agents": [],
            "coordination_rules": [],
            "orchestration_violations": [],
        }

        for agent in agents:
            agent_context = {
                **coordination_context,
                "agent_name": agent.get("name", ""),
                "agent_role": agent.get("role", ""),
                "agent_capabilities": agent.get("capabilities", []),
                "agent_trust_level": agent.get("trust_level", 0.5),
            }

            # Evaluate orchestration rules for this agent
            result = self.evaluate(
                agent_context, tags=["orchestration", "coordination"]
            )

            if result.action == PolicyAction.ALLOW:
                orchestration_result["allowed_agents"].append(agent)
            else:
                orchestration_result["blocked_agents"].append(
                    {
                        "agent": agent,
                        "reason": result.explanation,
                        "violations": [v.to_dict() for v in result.violations],
                    }
                )
                orchestration_result["orchestration_violations"].extend(
                    result.violations
                )

        # Add coordination rules based on allowed agents
        if len(orchestration_result["allowed_agents"]) > 1:
            orchestration_result["coordination_rules"] = (
                self._generate_coordination_rules(
                    orchestration_result["allowed_agents"], coordination_context
                )
            )

        return orchestration_result

    def _generate_coordination_rules(
        self, agents: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate coordination rules for multi-agent interaction."""
        rules = []

        # Communication order rules
        if len(agents) > 2:
            rules.append(
                {
                    "type": "communication_order",
                    "description": "Agents must communicate in specified order",
                    "agents": [agent["name"] for agent in agents],
                    "enforcement": "sequential",
                }
            )

        # Resource sharing rules
        if any("shared_resource" in agent.get("capabilities", []) for agent in agents):
            rules.append(
                {
                    "type": "resource_sharing",
                    "description": "Shared resources must be coordinated",
                    "resource_locks": True,
                    "timeout_seconds": 30,
                }
            )

        # Trust level constraints
        trust_levels = [agent.get("trust_level", 0.5) for agent in agents]
        if min(trust_levels) < 0.3:
            rules.append(
                {
                    "type": "supervision_required",
                    "description": "Low trust agents require supervision",
                    "supervisor_required": True,
                    "min_trust_level": 0.3,
                }
            )

        return rules

    def evaluate_tool_usage_policy(
        self, tool_name: str, tool_params: Dict[str, Any], agent_context: Dict[str, Any]
    ) -> PolicyResult:
        """
        Evaluate policy for tool usage by agents.

        Args:
            tool_name: Name of the tool being used
            tool_params: Parameters for tool execution
            agent_context: Context about the agent using the tool

        Returns:
            Policy result for tool usage
        """
        tool_context = {
            **agent_context,
            "tool_name": tool_name,
            "tool_params": tool_params,
            "action_type": "tool_usage",
        }

        # Evaluate with tool-specific tags
        result = self.evaluate(tool_context, tags=["tool", "usage", "security"])

        # Add tool-specific metadata
        result.metadata.update(
            {
                "tool_name": tool_name,
                "evaluation_type": "tool_usage",
                "agent_id": agent_context.get("agent_id", "unknown"),
            }
        )

        return result
