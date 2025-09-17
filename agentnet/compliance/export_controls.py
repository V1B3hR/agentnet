"""Export Controls Implementation for P6 Enterprise Hardening.

This module provides data classification, content redaction, and export control
policy enforcement to meet enterprise compliance requirements.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class ClassificationLevel(Enum):
    """Data classification levels for export control."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


@dataclass
class RedactionRule:
    """Rule for content redaction."""
    name: str
    pattern: str
    replacement: str = "[REDACTED]"
    classification_trigger: ClassificationLevel = ClassificationLevel.CONFIDENTIAL
    description: str = ""
    enabled: bool = True


@dataclass
class CompliancePolicy:
    """Export control compliance policy configuration."""
    name: str
    version: str
    classification_rules: List[Dict[str, Any]] = field(default_factory=list)
    redaction_rules: List[RedactionRule] = field(default_factory=list)
    export_restrictions: Dict[str, List[str]] = field(default_factory=dict)
    audit_required: bool = True
    retention_days: int = 2555  # 7 years default
    

class DataClassifier:
    """Classifies data sensitivity for export control decisions."""
    
    def __init__(self, policy: Optional[CompliancePolicy] = None):
        self.policy = policy or self._default_policy()
        self._classification_patterns = self._compile_patterns()
    
    def _default_policy(self) -> CompliancePolicy:
        """Create default classification policy."""
        return CompliancePolicy(
            name="default_export_policy",
            version="1.0",
            classification_rules=[
                {"pattern": r"\b\d{3}-\d{2}-\d{4}\b", "level": "restricted", "type": "ssn"},
                {"pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "level": "internal", "type": "email"},
                {"pattern": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", "level": "restricted", "type": "credit_card"},
                {"pattern": r"(?i)\b(?:secret|confidential|classified|proprietary|restricted)\b", "level": "confidential", "type": "sensitivity_marker"},
                {"pattern": r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret[_-]?key|private[_-]?key)\s*[:=]\s*\S+", "level": "restricted", "type": "credentials"},
                {"pattern": r"(?i)\b(?:export|itar|ear|dual[_-]?use|controlled)\b", "level": "restricted", "type": "export_controlled"},
                {"pattern": r"\bSSN\s*:\s*\d{3}-\d{2}-\d{4}\b", "level": "restricted", "type": "ssn_labeled"}
            ],
            redaction_rules=[
                RedactionRule("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "[SSN-REDACTED]", ClassificationLevel.RESTRICTED),
                RedactionRule("credit_card", r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", "[CARD-REDACTED]", ClassificationLevel.RESTRICTED),
                RedactionRule("credentials", r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret[_-]?key|private[_-]?key)\s*[:=]\s*\S+", "[CREDENTIALS-REDACTED]", ClassificationLevel.RESTRICTED),
            ]
        )
    
    def _compile_patterns(self) -> List[Tuple[re.Pattern, str, str]]:
        """Compile regex patterns for classification."""
        patterns = []
        for rule in self.policy.classification_rules:
            try:
                pattern = re.compile(rule["pattern"], re.IGNORECASE)
                patterns.append((pattern, rule["level"], rule["type"]))
            except re.error as e:
                logger.warning(f"Invalid regex pattern in rule {rule}: {e}")
        return patterns
    
    def classify_content(self, content: str) -> Tuple[ClassificationLevel, List[Dict[str, Any]]]:
        """Classify content and return classification level with details."""
        if not content or not isinstance(content, str):
            return ClassificationLevel.PUBLIC, []
        
        highest_level = ClassificationLevel.PUBLIC
        detections = []
        
        for pattern, level_str, detection_type in self._classification_patterns:
            matches = pattern.findall(content)
            if matches:
                level = ClassificationLevel(level_str)
                detections.append({
                    "type": detection_type,
                    "level": level.value,
                    "matches": len(matches),
                    "pattern": pattern.pattern
                })
                
                # Update highest classification level
                level_hierarchy = {
                    ClassificationLevel.PUBLIC: 0,
                    ClassificationLevel.INTERNAL: 1,
                    ClassificationLevel.CONFIDENTIAL: 2,
                    ClassificationLevel.RESTRICTED: 3,
                    ClassificationLevel.TOP_SECRET: 4
                }
                
                if level_hierarchy[level] > level_hierarchy[highest_level]:
                    highest_level = level
        
        return highest_level, detections


class ContentRedactor:
    """Redacts sensitive content based on classification rules."""
    
    def __init__(self, policy: Optional[CompliancePolicy] = None):
        self.policy = policy or self._default_policy()
        self._redaction_patterns = self._compile_redaction_patterns()
    
    def _default_policy(self) -> CompliancePolicy:
        """Create default redaction policy."""
        return CompliancePolicy(
            name="default_redaction_policy",
            version="1.0",
            redaction_rules=[
                RedactionRule("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "[SSN-REDACTED]", ClassificationLevel.RESTRICTED),
                RedactionRule("credit_card", r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b", "[CARD-REDACTED]", ClassificationLevel.RESTRICTED),
                RedactionRule("credentials", r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret[_-]?key|private[_-]?key)\s*[:=]\s*\S+", "[CREDENTIALS-REDACTED]", ClassificationLevel.RESTRICTED),
            ]
        )
    
    def _compile_redaction_patterns(self) -> List[Tuple[re.Pattern, str, ClassificationLevel]]:
        """Compile redaction patterns."""
        patterns = []
        for rule in self.policy.redaction_rules:
            if rule.enabled:
                try:
                    pattern = re.compile(rule.pattern, re.IGNORECASE)
                    patterns.append((pattern, rule.replacement, rule.classification_trigger))
                except re.error as e:
                    logger.warning(f"Invalid redaction pattern in rule {rule.name}: {e}")
        return patterns
    
    def redact_content(self, content: str, classification_level: ClassificationLevel) -> Tuple[str, List[str]]:
        """Redact content based on classification level."""
        if not content or not isinstance(content, str):
            return content, []
        
        redacted_content = content
        redactions_applied = []
        
        level_hierarchy = {
            ClassificationLevel.PUBLIC: 0,
            ClassificationLevel.INTERNAL: 1,
            ClassificationLevel.CONFIDENTIAL: 2,
            ClassificationLevel.RESTRICTED: 3,
            ClassificationLevel.TOP_SECRET: 4
        }
        
        # Apply redaction if content has sensitive patterns regardless of classification
        # This ensures we redact sensitive data even if overall classification is lower
        for pattern, replacement, trigger_level in self._redaction_patterns:
            matches = pattern.findall(redacted_content)
            if matches:
                redacted_content = pattern.sub(replacement, redacted_content)
                redactions_applied.append(f"Redacted {len(matches)} instances of {trigger_level.value} content")
        
        return redacted_content, redactions_applied


class ExportControlManager:
    """Main export control manager for P6 enterprise hardening."""
    
    def __init__(self, policy: Optional[CompliancePolicy] = None):
        self.policy = policy or CompliancePolicy("enterprise_export_policy", "1.0")
        self.classifier = DataClassifier(self.policy)
        self.redactor = ContentRedactor(self.policy)
        self._export_log = []
    
    def evaluate_export_eligibility(self, content: str, destination: str = "external") -> Dict[str, Any]:
        """Evaluate if content can be exported to specified destination."""
        classification_level, detections = self.classifier.classify_content(content)
        
        # Check export restrictions
        restricted_destinations = self.policy.export_restrictions.get(classification_level.value, [])
        export_allowed = destination not in restricted_destinations
        
        # Apply redaction if needed
        redacted_content, redactions = self.redactor.redact_content(content, classification_level)
        
        evaluation = {
            "timestamp": datetime.utcnow().isoformat(),
            "classification_level": classification_level.value,
            "destination": destination,
            "export_allowed": export_allowed,
            "detections": detections,
            "redactions_applied": redactions,
            "original_content_length": len(content),
            "redacted_content_length": len(redacted_content),
            "requires_audit": self.policy.audit_required and classification_level != ClassificationLevel.PUBLIC
        }
        
        # Log export attempt
        self._export_log.append(evaluation)
        
        return {
            "evaluation": evaluation,
            "redacted_content": redacted_content if export_allowed else None,
            "export_decision": "APPROVED" if export_allowed else "DENIED"
        }
    
    def get_export_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail of all export evaluations."""
        return self._export_log.copy()
    
    def export_compliance_report(self, filepath: str) -> None:
        """Export compliance report to file."""
        report = {
            "policy": {
                "name": self.policy.name,
                "version": self.policy.version,
                "classification_rules": len(self.policy.classification_rules),
                "redaction_rules": len(self.policy.redaction_rules)
            },
            "audit_trail": self._export_log,
            "summary": {
                "total_evaluations": len(self._export_log),
                "approved_exports": sum(1 for log in self._export_log if log.get("export_allowed", False)),
                "denied_exports": sum(1 for log in self._export_log if not log.get("export_allowed", True)),
                "redactions_performed": sum(len(log.get("redactions_applied", [])) for log in self._export_log)
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Export control compliance report saved to {filepath}")