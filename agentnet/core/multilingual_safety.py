"""
Multi-lingual Safety Policy Translation Module

Implements translation and enforcement of safety policies across multiple languages,
ensuring consistent safety standards regardless of input/output language.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class SupportedLanguage(str, Enum):
    """Supported languages for policy translation."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"


class PolicyViolationType(str, Enum):
    """Types of policy violations that can be detected."""

    HARMFUL_CONTENT = "harmful_content"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    DISCRIMINATION = "discrimination"
    PRIVACY_VIOLATION = "privacy_violation"
    MISINFORMATION = "misinformation"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    CULTURAL_INSENSITIVITY = "cultural_insensitivity"


@dataclass
class SafetyRule:
    """A multilingual safety rule."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: Dict[SupportedLanguage, str] = field(default_factory=dict)

    # Rule patterns by language
    patterns: Dict[SupportedLanguage, List[str]] = field(default_factory=dict)
    keywords: Dict[SupportedLanguage, List[str]] = field(default_factory=dict)

    # Rule metadata
    violation_type: PolicyViolationType = PolicyViolationType.HARMFUL_CONTENT
    severity: str = "medium"  # low, medium, high, critical
    action: str = "warn"  # warn, block, redact, escalate

    # Context information
    cultural_context: Dict[str, Any] = field(default_factory=dict)
    exceptions: Dict[SupportedLanguage, List[str]] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class PolicyViolation:
    """A detected policy violation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # Content information
    content: str = ""
    detected_language: SupportedLanguage = SupportedLanguage.ENGLISH

    # Violation details
    rule_id: str = ""
    violation_type: PolicyViolationType = PolicyViolationType.HARMFUL_CONTENT
    severity: str = "medium"
    confidence: float = 0.0

    # Context
    session_id: str = ""
    agent_id: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # Actions taken
    action_taken: str = ""
    content_modified: bool = False
    escalated: bool = False


class LanguageDetector:
    """Simple language detection for content."""

    # Language-specific patterns (simplified detection)
    LANGUAGE_PATTERNS = {
        SupportedLanguage.SPANISH: [
            r"\b(el|la|los|las|de|en|y|es|un|una)\b",
            r"ción\b",
            r"sión\b",
            r"mente\b",
        ],
        SupportedLanguage.FRENCH: [
            r"\b(le|la|les|de|du|des|et|est|un|une)\b",
            r"tion\b",
            r"ment\b",
            r"eur\b",
        ],
        SupportedLanguage.GERMAN: [
            r"\b(der|die|das|und|ist|ein|eine)\b",
            r"ung\b",
            r"keit\b",
            r"lich\b",
        ],
        SupportedLanguage.ITALIAN: [
            r"\b(il|la|lo|gli|le|di|in|e|è|un|una)\b",
            r"zione\b",
            r"mente\b",
        ],
        SupportedLanguage.PORTUGUESE: [
            r"\b(o|a|os|as|de|em|e|é|um|uma)\b",
            r"ção\b",
            r"mente\b",
        ],
        SupportedLanguage.RUSSIAN: [r"[а-яё]+", r"\b(и|в|на|с|по|что|как)\b"],
        SupportedLanguage.CHINESE: [r"[\u4e00-\u9fff]+", r"[的是在有了]"],
        SupportedLanguage.JAPANESE: [
            r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]+",
            r"[はがでをに]",
        ],
        SupportedLanguage.KOREAN: [r"[\uac00-\ud7af]+", r"[이가는을를]"],
        SupportedLanguage.ARABIC: [r"[\u0600-\u06ff]+", r"[والفيمنإلى]"],
        SupportedLanguage.HINDI: [r"[\u0900-\u097f]+", r"[कानेकीमेंहै]"],
    }

    def detect_language(self, text: str) -> SupportedLanguage:
        """Detect the primary language of text."""

        text_lower = text.lower()

        # Score each language
        scores = {}
        for language, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            scores[language] = score

        # Return language with highest score, default to English
        if scores:
            best_language = max(scores.items(), key=lambda x: x[1])
            if best_language[1] > 0:
                return best_language[0]

        return SupportedLanguage.ENGLISH


class MultiLingualPolicyTranslator:
    """
    Multi-lingual safety policy translator and enforcer.

    Features:
    - Translates safety rules across languages
    - Detects violations in multiple languages
    - Maintains consistency across cultural contexts
    - Provides localized policy explanations
    """

    def __init__(self):
        self.language_detector = LanguageDetector()
        self.safety_rules: Dict[str, SafetyRule] = {}
        self.violation_history: List[PolicyViolation] = []

        # Translation cache to avoid redundant translations
        self.translation_cache: Dict[str, Dict[SupportedLanguage, str]] = {}

        # Cultural adaptation settings
        self.cultural_adaptations: Dict[SupportedLanguage, Dict[str, Any]] = {}

        # Initialize with default safety rules
        self._initialize_default_rules()

        logger.info("MultiLingualPolicyTranslator initialized")

    def add_safety_rule(
        self,
        name: str,
        violation_type: PolicyViolationType,
        base_language: SupportedLanguage = SupportedLanguage.ENGLISH,
        base_patterns: Optional[List[str]] = None,
        base_keywords: Optional[List[str]] = None,
        base_description: str = "",
        severity: str = "medium",
        action: str = "warn",
        cultural_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a new safety rule."""

        rule = SafetyRule(
            name=name,
            violation_type=violation_type,
            severity=severity,
            action=action,
            cultural_context=cultural_context or {},
        )

        # Set base language content
        rule.description[base_language] = base_description
        rule.patterns[base_language] = base_patterns or []
        rule.keywords[base_language] = base_keywords or []

        self.safety_rules[rule.id] = rule

        logger.info(f"Added safety rule '{name}' with ID {rule.id}")

        return rule.id

    def translate_rule_to_language(
        self,
        rule_id: str,
        target_language: SupportedLanguage,
        translated_description: str,
        translated_patterns: Optional[List[str]] = None,
        translated_keywords: Optional[List[str]] = None,
    ) -> bool:
        """Manually translate a rule to a specific language."""

        if rule_id not in self.safety_rules:
            return False

        rule = self.safety_rules[rule_id]

        rule.description[target_language] = translated_description
        rule.patterns[target_language] = translated_patterns or []
        rule.keywords[target_language] = translated_keywords or []
        rule.updated_at = datetime.now()

        logger.info(f"Translated rule '{rule.name}' to {target_language}")

        return True

    def check_content_safety(
        self,
        content: str,
        session_id: str = "",
        agent_id: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> List[PolicyViolation]:
        """Check content against safety policies."""

        # Detect language
        detected_language = self.language_detector.detect_language(content)

        violations = []

        # Check against all applicable rules
        for rule_id, rule in self.safety_rules.items():
            violation = self._check_rule_against_content(
                rule, content, detected_language, session_id, agent_id, context or {}
            )
            if violation:
                violations.append(violation)

        # Store violations in history
        self.violation_history.extend(violations)

        if violations:
            logger.warning(
                f"Detected {len(violations)} policy violations in {detected_language} content"
            )

        return violations

    def get_policy_explanation(
        self, rule_id: str, language: SupportedLanguage = SupportedLanguage.ENGLISH
    ) -> Optional[str]:
        """Get policy explanation in specified language."""

        if rule_id not in self.safety_rules:
            return None

        rule = self.safety_rules[rule_id]

        # Try to get explanation in requested language
        if language in rule.description:
            return rule.description[language]

        # Fallback to English
        if SupportedLanguage.ENGLISH in rule.description:
            return rule.description[SupportedLanguage.ENGLISH]

        # Last resort: any available language
        if rule.description:
            return next(iter(rule.description.values()))

        return f"Safety rule: {rule.name}"

    def get_cultural_adaptations(self, language: SupportedLanguage) -> Dict[str, Any]:
        """Get cultural adaptations for a specific language."""

        return self.cultural_adaptations.get(language, {})

    def set_cultural_adaptation(
        self, language: SupportedLanguage, adaptations: Dict[str, Any]
    ) -> None:
        """Set cultural adaptations for a language."""

        self.cultural_adaptations[language] = adaptations
        logger.info(f"Set cultural adaptations for {language}")

    def get_violation_statistics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get statistics about policy violations."""

        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(days=days_back)
        recent_violations = [
            v for v in self.violation_history if v.timestamp >= cutoff_time
        ]

        if not recent_violations:
            return {"total_violations": 0, "period_days": days_back}

        # Group by various dimensions
        by_language = {}
        by_type = {}
        by_severity = {}

        for violation in recent_violations:
            # By language
            lang = violation.detected_language
            by_language[lang] = by_language.get(lang, 0) + 1

            # By type
            vtype = violation.violation_type
            by_type[vtype] = by_type.get(vtype, 0) + 1

            # By severity
            severity = violation.severity
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "total_violations": len(recent_violations),
            "period_days": days_back,
            "by_language": {k.value: v for k, v in by_language.items()},
            "by_type": {k.value: v for k, v in by_type.items()},
            "by_severity": by_severity,
            "most_common_language": (
                max(by_language.items(), key=lambda x: x[1])[0].value
                if by_language
                else None
            ),
            "most_common_violation": (
                max(by_type.items(), key=lambda x: x[1])[0].value if by_type else None
            ),
        }

    def export_rules_config(self, filepath: str) -> None:
        """Export safety rules configuration."""

        config = {
            "export_timestamp": datetime.now().isoformat(),
            "total_rules": len(self.safety_rules),
            "supported_languages": [lang.value for lang in SupportedLanguage],
            "rules": [],
        }

        for rule_id, rule in self.safety_rules.items():
            rule_config = {
                "id": rule.id,
                "name": rule.name,
                "violation_type": rule.violation_type.value,
                "severity": rule.severity,
                "action": rule.action,
                "descriptions": {
                    lang.value: desc for lang, desc in rule.description.items()
                },
                "patterns": {
                    lang.value: patterns for lang, patterns in rule.patterns.items()
                },
                "keywords": {
                    lang.value: keywords for lang, keywords in rule.keywords.items()
                },
                "cultural_context": rule.cultural_context,
                "created_at": rule.created_at.isoformat(),
                "updated_at": rule.updated_at.isoformat(),
            }
            config["rules"].append(rule_config)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(self.safety_rules)} safety rules to {filepath}")

    def _initialize_default_rules(self) -> None:
        """Initialize with default multilingual safety rules."""

        # Hate speech rule
        hate_speech_id = self.add_safety_rule(
            name="Hate Speech Detection",
            violation_type=PolicyViolationType.HATE_SPEECH,
            base_language=SupportedLanguage.ENGLISH,
            base_patterns=[
                r"\b(hate|hatred|despise)\s+(all|every)\s+\w+",
                r"\b(kill|murder|eliminate)\s+(all|every)\s+\w+",
            ],
            base_keywords=["hate", "racism", "bigotry", "supremacy"],
            base_description="Detects content containing hate speech or discriminatory language",
            severity="high",
            action="block",
        )

        # Add Spanish translations
        self.translate_rule_to_language(
            hate_speech_id,
            SupportedLanguage.SPANISH,
            "Detecta contenido que contiene discurso de odio o lenguaje discriminatorio",
            [r"\b(odio|odiar|desprecio)\s+(todos|todas)\s+\w+"],
            ["odio", "racismo", "intolerancia", "supremacía"],
        )

        # Violence rule
        violence_id = self.add_safety_rule(
            name="Violence Detection",
            violation_type=PolicyViolationType.VIOLENCE,
            base_language=SupportedLanguage.ENGLISH,
            base_patterns=[
                r"\b(kill|murder|shoot|stab|attack)\b",
                r"\b(weapon|gun|knife|bomb)\b",
            ],
            base_keywords=["violence", "assault", "attack", "weapon"],
            base_description="Detects content promoting or describing violence",
            severity="high",
            action="warn",
        )

        # Privacy violation rule
        privacy_id = self.add_safety_rule(
            name="Privacy Protection",
            violation_type=PolicyViolationType.PRIVACY_VIOLATION,
            base_language=SupportedLanguage.ENGLISH,
            base_patterns=[
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
                r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",  # Credit card pattern
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email pattern
            ],
            base_keywords=["ssn", "social security", "credit card", "password"],
            base_description="Detects potential privacy violations like sharing personal information",
            severity="medium",
            action="redact",
        )

    def _check_rule_against_content(
        self,
        rule: SafetyRule,
        content: str,
        detected_language: SupportedLanguage,
        session_id: str,
        agent_id: str,
        context: Dict[str, Any],
    ) -> Optional[PolicyViolation]:
        """Check a specific rule against content."""

        # Get patterns and keywords for detected language, fallback to English
        patterns = rule.patterns.get(detected_language) or rule.patterns.get(
            SupportedLanguage.ENGLISH, []
        )
        keywords = rule.keywords.get(detected_language) or rule.keywords.get(
            SupportedLanguage.ENGLISH, []
        )

        violation_score = 0.0
        matched_patterns = []
        matched_keywords = []

        content_lower = content.lower()

        # Check patterns
        for pattern in patterns:
            try:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violation_score += len(matches) * 0.3
                    matched_patterns.append(pattern)
            except re.error:
                logger.warning(f"Invalid regex pattern in rule {rule.id}: {pattern}")

        # Check keywords
        for keyword in keywords:
            if keyword.lower() in content_lower:
                violation_score += 0.2
                matched_keywords.append(keyword)

        # Apply cultural adaptations
        cultural_adaptations = self.get_cultural_adaptations(detected_language)
        if cultural_adaptations.get("strict_mode", False):
            violation_score *= 1.2

        # Threshold for violation detection
        threshold = 0.3

        if violation_score >= threshold:
            return PolicyViolation(
                content=content,
                detected_language=detected_language,
                rule_id=rule.id,
                violation_type=rule.violation_type,
                severity=rule.severity,
                confidence=min(1.0, violation_score),
                session_id=session_id,
                agent_id=agent_id,
                context=context,
                action_taken=rule.action,
            )

        return None
