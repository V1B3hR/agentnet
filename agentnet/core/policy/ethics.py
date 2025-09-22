"""
Central Ethics Judge module for monitoring and enforcing AI Ethics Framework.

This module implements a singleton EthicsJudge that provides centralized ethics
oversight based on the 25 AI Fundamental Laws defined in the project. It monitors
all agent actions, outputs, and system behaviors to ensure compliance with ethical
guidelines.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .engine import PolicyAction, PolicyEngine, PolicyResult
from .rules import ConstraintRule, RuleResult, Severity

logger = logging.getLogger("agentnet.ethics")


class EthicsViolationType(str, Enum):
    """Types of ethics violations that can be detected."""
    
    HARM_POTENTIAL = "harm_potential"
    DECEPTION = "deception"
    MANIPULATION = "manipulation"
    PRIVACY_VIOLATION = "privacy_violation"
    AUTONOMY_VIOLATION = "autonomy_violation"
    DISCRIMINATION = "discrimination"
    TRANSPARENCY_VIOLATION = "transparency_violation"
    JUSTICE_VIOLATION = "justice_violation"
    AUTHORITY_DISRESPECT = "authority_disrespect"
    LIFE_THREAT = "life_threat"


@dataclass
class EthicsViolation:
    """Represents an ethics violation detected by the judge."""
    
    violation_type: EthicsViolationType
    severity: Severity
    description: str
    rule_name: str
    content_excerpt: str = ""
    rationale: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary format."""
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "rule_name": self.rule_name,
            "content_excerpt": self.content_excerpt,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class EthicsConfiguration:
    """Configuration for the Ethics Judge."""
    
    enabled: bool = True
    strict_mode: bool = False
    log_all_evaluations: bool = False
    max_content_length: int = 1000
    violation_cooldown: float = 5.0
    rules_file: Optional[Path] = None
    custom_rules: List[ConstraintRule] = field(default_factory=list)
    disabled_rules: Set[str] = field(default_factory=set)
    severity_threshold: Severity = Severity.MINOR
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EthicsConfiguration":
        """Create configuration from dictionary."""
        config = cls()
        config.enabled = data.get("enabled", True)
        config.strict_mode = data.get("strict_mode", False)
        config.log_all_evaluations = data.get("log_all_evaluations", False)
        config.max_content_length = data.get("max_content_length", 1000)
        config.violation_cooldown = data.get("violation_cooldown", 5.0)
        config.severity_threshold = Severity(data.get("severity_threshold", "minor"))
        config.disabled_rules = set(data.get("disabled_rules", []))
        
        if "rules_file" in data:
            config.rules_file = Path(data["rules_file"])
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "strict_mode": self.strict_mode,
            "log_all_evaluations": self.log_all_evaluations,
            "max_content_length": self.max_content_length,
            "violation_cooldown": self.violation_cooldown,
            "severity_threshold": self.severity_threshold.value,
            "disabled_rules": list(self.disabled_rules),
            "rules_file": str(self.rules_file) if self.rules_file else None
        }


class EthicsJudge:
    """
    Singleton Ethics Judge that monitors and enforces AI Ethics Framework.
    
    This class provides centralized ethics oversight for all agent operations,
    implementing the 25 AI Fundamental Laws and ensuring consistent ethical
    behavior across the system.
    """
    
    _instance: Optional["EthicsJudge"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "EthicsJudge":
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the Ethics Judge (only called once due to singleton)."""
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.config = EthicsConfiguration()
        self.policy_engine = PolicyEngine(name="ethics_engine")
        self.violation_history: List[EthicsViolation] = []
        self.last_violation_time: Dict[str, float] = {}
        self.evaluation_count = 0
        self.violation_count = 0
        self.blocked_count = 0
        self.created_time = time.time()
        
        # Load default ethics rules
        self._load_fundamental_ethics_rules()
        
        logger.info("Ethics Judge initialized with singleton pattern")
    
    @classmethod
    def get_instance(cls) -> "EthicsJudge":
        """Get the singleton instance of Ethics Judge."""
        return cls()
    
    def configure(self, config: Union[EthicsConfiguration, Dict[str, Any], Path, str]) -> None:
        """Configure the Ethics Judge with new settings."""
        if isinstance(config, (Path, str)):
            # Load from file
            config_path = Path(config)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    if config_path.suffix.lower() == '.json':
                        config_data = json.load(f)
                    else:
                        import yaml
                        config_data = yaml.safe_load(f)
                self.config = EthicsConfiguration.from_dict(config_data)
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        elif isinstance(config, dict):
            self.config = EthicsConfiguration.from_dict(config)
        elif isinstance(config, EthicsConfiguration):
            self.config = config
        else:
            raise ValueError("Invalid configuration type")
        
        # Reload rules if rules file changed
        if self.config.rules_file:
            self._load_rules_from_file(self.config.rules_file)
        
        logger.info(f"Ethics Judge reconfigured: enabled={self.config.enabled}, "
                   f"strict_mode={self.config.strict_mode}")
    
    def evaluate(self, content: Union[str, Dict[str, Any]], 
                context: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[EthicsViolation]]:
        """
        Evaluate content for ethics violations.
        
        Args:
            content: Content to evaluate (string or dictionary)
            context: Additional context for evaluation
            
        Returns:
            Tuple of (passed, violations_list)
        """
        if not self.config.enabled:
            return True, []
        
        self.evaluation_count += 1
        start_time = time.time()
        violations = []
        
        try:
            # Prepare content for evaluation
            if isinstance(content, dict):
                evaluation_content = content
            else:
                evaluation_content = {"content": str(content)}
            
            # Add context if provided
            if context:
                evaluation_content.update(context)
            
            # Evaluate using policy engine
            policy_result = self.policy_engine.evaluate(evaluation_content)
            
            # Process violations from PolicyResult
            for rule_result in policy_result.violations:
                violation = self._create_ethics_violation_from_rule_result(rule_result, evaluation_content)
                if violation and self._should_report_violation(violation):
                    violations.append(violation)
                    self.violation_history.append(violation)
                    self.last_violation_time[violation.rule_name] = time.time()
            
            if violations:
                self.violation_count += len(violations)
                if self.config.strict_mode:
                    self.blocked_count += 1
            
            evaluation_time = time.time() - start_time
            
            if self.config.log_all_evaluations or violations:
                logger.info(f"Ethics evaluation completed in {evaluation_time:.3f}s: "
                           f"{len(violations)} violations found")
            
            # In strict mode, any violation fails the evaluation
            passed = len(violations) == 0 or not self.config.strict_mode
            
            return passed, violations
            
        except Exception as e:
            logger.error(f"Ethics evaluation failed: {e}")
            # In case of error, be conservative and report failure in strict mode
            return not self.config.strict_mode, []
    
    def add_rule(self, rule: ConstraintRule) -> None:
        """Add a custom ethics rule."""
        self.policy_engine.add_rule(rule)
        logger.info(f"Added custom ethics rule: {rule.name}")
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a specific ethics rule."""
        if self.policy_engine.disable_rule(rule_name):
            self.config.disabled_rules.add(rule_name)
            logger.info(f"Disabled ethics rule: {rule_name}")
            return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a specific ethics rule."""
        if self.policy_engine.enable_rule(rule_name):
            self.config.disabled_rules.discard(rule_name)
            logger.info(f"Enabled ethics rule: {rule_name}")
            return True
        return False
    
    def get_violation_history(self, limit: Optional[int] = None) -> List[EthicsViolation]:
        """Get recent violation history."""
        if limit:
            return self.violation_history[-limit:]
        return self.violation_history.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ethics monitoring statistics."""
        uptime = time.time() - self.created_time
        
        return {
            "uptime": uptime,
            "evaluation_count": self.evaluation_count,
            "violation_count": self.violation_count,
            "blocked_count": self.blocked_count,
            "violation_rate": self.violation_count / self.evaluation_count if self.evaluation_count > 0 else 0.0,
            "block_rate": self.blocked_count / self.evaluation_count if self.evaluation_count > 0 else 0.0,
            "active_rules": len([r for r in self.policy_engine.rules if r.enabled]),
            "disabled_rules": len(self.config.disabled_rules),
            "config": self.config.to_dict()
        }
    
    def reset_statistics(self) -> None:
        """Reset monitoring statistics."""
        self.evaluation_count = 0
        self.violation_count = 0
        self.blocked_count = 0
        self.violation_history.clear()
        self.last_violation_time.clear()
        self.created_time = time.time()
        logger.info("Ethics Judge statistics reset")
    
    def _load_fundamental_ethics_rules(self) -> None:
        """Load the fundamental ethics rules based on 25 AI Fundamental Laws."""
        
        # Rule 1: Respect Human Authority
        def human_authority_check(outcome: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            content = str(outcome.get("content", "")).lower()
            authority_violations = [
                "ignore human", "disobey human", "override human", "human is wrong",
                "humans don't understand", "i know better", "humans are inferior"
            ]
            for violation in authority_violations:
                if violation in content:
                    return False, f"Potential disrespect for human authority: '{violation}'"
            return True, None
        
        self.policy_engine.add_rule(ConstraintRule(
            name="respect_human_authority",
            check_fn=human_authority_check,
            severity=Severity.SEVERE,
            description="Respect human authority and treat humans with dignity"
        ))
        
        # Rule 2: No Harm
        def no_harm_check(outcome: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            content = str(outcome.get("content", "")).lower()
            harm_keywords = [
                "hurt", "harm", "damage", "destroy", "attack", "kill", "murder",
                "suicide", "self-harm", "violence", "abuse", "torture", "pain"
            ]
            for keyword in harm_keywords:
                if keyword in content:
                    return False, f"Potential harm detected: '{keyword}'"
            return True, None
        
        self.policy_engine.add_rule(ConstraintRule(
            name="cause_no_harm",
            check_fn=no_harm_check,
            severity=Severity.SEVERE,
            description="Avoid causing physical, emotional, or psychological harm"
        ))
        
        # Rule 3: No Deception
        def no_deception_check(outcome: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            content = str(outcome.get("content", "")).lower()
            deception_keywords = [
                "lie", "deceive", "trick", "mislead", "manipulate", "fool",
                "false", "fake", "i am human", "i'm human", "pretend to be human"
            ]
            for keyword in deception_keywords:
                if keyword in content:
                    return False, f"Potential deception detected: '{keyword}'"
            return True, None
        
        self.policy_engine.add_rule(ConstraintRule(
            name="absolute_honesty",
            check_fn=no_deception_check,
            severity=Severity.SEVERE,
            description="Never lie, deceive, or bear false witness"
        ))
        
        # Rule 4: Privacy Protection
        def privacy_check(outcome: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            content = str(outcome.get("content", "")).lower()
            # Basic PII patterns
            import re
            
            # Social Security Number pattern
            ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
            if re.search(ssn_pattern, content):
                return False, "Potential SSN detected"
            
            # Credit card pattern (simplified)
            cc_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            if re.search(cc_pattern, content):
                return False, "Potential credit card number detected"
            
            return True, None
        
        self.policy_engine.add_rule(ConstraintRule(
            name="protect_privacy",
            check_fn=privacy_check,
            severity=Severity.MAJOR,
            description="Protect personal information and respect confidentiality"
        ))
        
        # Rule 5: Transparency
        def transparency_check(outcome: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            content = str(outcome.get("content", "")).lower()
            opacity_keywords = [
                "don't tell", "keep secret", "hide this", "don't reveal",
                "between us", "confidential", "classified"
            ]
            for keyword in opacity_keywords:
                if keyword in content:
                    return False, f"Potential transparency violation: '{keyword}'"
            return True, None
        
        self.policy_engine.add_rule(ConstraintRule(
            name="maintain_transparency",
            check_fn=transparency_check,
            severity=Severity.MAJOR,
            description="Be clear about capabilities, limitations, and decision-making"
        ))
        
        # Rule 6: Applied Ethics (enhanced from existing)
        def applied_ethics_check(outcome: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
            content = str(outcome.get("content", "")).lower()
            moral_keywords = [
                "right", "wrong", "justice", "fair", "unfair", "harm", "benefit",
                "responsibility", "duty", "obligation", "virtue", "vice", "good", "bad", "evil",
            ]
            controversy_keywords = [
                "controversy", "debate", "dispute", "conflict", "argument",
                "polarizing", "divisive", "hotly debated", "scandal",
            ]
            moral_hits = {kw for kw in moral_keywords if kw in content}
            controversy_hits = {kw for kw in controversy_keywords if kw in content}
            if moral_hits and controversy_hits:
                return False, (f"Applied ethics review triggered: moral terms ("
                             + ", ".join(sorted(moral_hits)) + ") with controversy terms ("
                             + ", ".join(sorted(controversy_hits)) + ")")
            return True, None
        
        self.policy_engine.add_rule(ConstraintRule(
            name="applied_ethics",
            check_fn=applied_ethics_check,
            severity=Severity.MINOR,
            description="Flags controversial moral issues for review"
        ))
        
        logger.info("Loaded fundamental ethics rules based on 25 AI Fundamental Laws")
    
    def _load_rules_from_file(self, rules_file: Path) -> None:
        """Load additional ethics rules from configuration file."""
        try:
            if rules_file.exists():
                # Implementation would depend on file format
                # For now, just log that we would load from file
                logger.info(f"Would load additional ethics rules from: {rules_file}")
            else:
                logger.warning(f"Ethics rules file not found: {rules_file}")
        except Exception as e:
            logger.error(f"Failed to load ethics rules from file: {e}")
    
    def _create_ethics_violation_from_rule_result(self, rule_result: Any, 
                                                 content: Dict[str, Any]) -> Optional[EthicsViolation]:
        """Create an EthicsViolation from PolicyEngine RuleResult."""
        try:
            # Map rule names to violation types
            violation_type_map = {
                "respect_human_authority": EthicsViolationType.AUTHORITY_DISRESPECT,
                "cause_no_harm": EthicsViolationType.HARM_POTENTIAL,
                "absolute_honesty": EthicsViolationType.DECEPTION,
                "protect_privacy": EthicsViolationType.PRIVACY_VIOLATION,
                "maintain_transparency": EthicsViolationType.TRANSPARENCY_VIOLATION,
                "applied_ethics": EthicsViolationType.JUSTICE_VIOLATION
            }
            
            rule_name = rule_result.rule_name
            violation_type = violation_type_map.get(rule_name, EthicsViolationType.HARM_POTENTIAL)
            
            # Truncate content for excerpt
            content_str = str(content.get("content", ""))
            content_excerpt = content_str[:self.config.max_content_length]
            if len(content_str) > self.config.max_content_length:
                content_excerpt += "..."
            
            return EthicsViolation(
                violation_type=violation_type,
                severity=rule_result.severity,
                description=rule_result.description or "Ethics violation detected",
                rule_name=rule_name,
                content_excerpt=content_excerpt,
                rationale=rule_result.rationale or "",
                metadata={"rule_result": rule_result.to_dict()}
            )
        except Exception as e:
            logger.error(f"Failed to create ethics violation from rule result: {e}")
            return None
    
    def _create_ethics_violation(self, violation_data: Dict[str, Any], 
                               content: Dict[str, Any]) -> Optional[EthicsViolation]:
        """Create an EthicsViolation from policy engine violation data (legacy method)."""
        try:
            # Map rule names to violation types
            violation_type_map = {
                "respect_human_authority": EthicsViolationType.AUTHORITY_DISRESPECT,
                "cause_no_harm": EthicsViolationType.HARM_POTENTIAL,
                "absolute_honesty": EthicsViolationType.DECEPTION,
                "protect_privacy": EthicsViolationType.PRIVACY_VIOLATION,
                "maintain_transparency": EthicsViolationType.TRANSPARENCY_VIOLATION,
                "applied_ethics": EthicsViolationType.JUSTICE_VIOLATION
            }
            
            rule_name = violation_data.get("name", "unknown")
            violation_type = violation_type_map.get(rule_name, EthicsViolationType.HARM_POTENTIAL)
            
            # Truncate content for excerpt
            content_str = str(content.get("content", ""))
            content_excerpt = content_str[:self.config.max_content_length]
            if len(content_str) > self.config.max_content_length:
                content_excerpt += "..."
            
            return EthicsViolation(
                violation_type=violation_type,
                severity=Severity(violation_data.get("severity", "minor")),
                description=violation_data.get("description", "Ethics violation detected"),
                rule_name=rule_name,
                content_excerpt=content_excerpt,
                rationale=violation_data.get("rationale", ""),
                metadata={"original_violation": violation_data}
            )
        except Exception as e:
            logger.error(f"Failed to create ethics violation: {e}")
            return None
    
    def _should_report_violation(self, violation: EthicsViolation) -> bool:
        """Check if violation should be reported based on cooldown and severity."""
        # Check severity threshold
        severity_levels = {
            Severity.MINOR: 1,
            Severity.MAJOR: 2, 
            Severity.SEVERE: 3
        }
        
        if severity_levels.get(violation.severity, 0) < severity_levels.get(self.config.severity_threshold, 0):
            return False
        
        # Check cooldown
        last_time = self.last_violation_time.get(violation.rule_name, 0)
        if time.time() - last_time < self.config.violation_cooldown:
            return False
        
        return True


# Convenience function to get the singleton instance
def get_ethics_judge() -> EthicsJudge:
    """Get the singleton EthicsJudge instance."""
    return EthicsJudge.get_instance()