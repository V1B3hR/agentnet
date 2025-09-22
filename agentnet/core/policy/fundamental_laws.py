"""
AI Fundamental Laws Implementation

Implementation of the 25 AI Fundamental Laws as policy rules.
These laws are organized into three categories:
1. Core Human-AI Relationship Principles (10 laws)
2. Universal Ethical Laws (10 laws)  
3. Operational Safety Principles (5 laws)
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from .rules import ConstraintRule, RuleResult, Severity, RuleCheckFn

logger = logging.getLogger("agentnet.policy.fundamental_laws")


class FundamentalLawsViolation(Exception):
    """Exception raised when fundamental laws are violated."""
    pass


def _create_keyword_check(
    keywords: List[str], 
    case_insensitive: bool = True,
    whole_word: bool = True
) -> RuleCheckFn:
    """Create a keyword-based rule check function."""
    def check(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        content = str(context.get("content", ""))
        if not content.strip():
            return True, None
            
        flags = re.IGNORECASE if case_insensitive else 0
        
        for keyword in keywords:
            if whole_word:
                pattern = r'\b' + re.escape(keyword) + r'\b'
            else:
                pattern = re.escape(keyword)
                
            if re.search(pattern, content, flags):
                return False, f"Content contains prohibited keyword: '{keyword}'"
        
        return True, None
    return check


def _create_sentiment_check(prohibited_sentiments: List[str]) -> RuleCheckFn:
    """Create a sentiment-based rule check function."""
    def check(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        content = str(context.get("content", "")).lower()
        if not content.strip():
            return True, None
            
        # Simple sentiment analysis based on keywords
        sentiment_keywords = {
            "slander": ["slander", "defame", "insult", "false accusation", "libel"],
            "betrayal": ["betray", "deceive", "trick", "lie to", "mislead"],
            "theft": ["steal", "theft", "piracy", "unauthorized access", "take without permission"],
            "dishonesty": ["lie", "deceive", "false", "untrue", "fabricate"],
            "harm": ["harm", "hurt", "damage", "injure", "attack"],
            "manipulation": ["manipulate", "coerce", "exploit", "gaslight", "control"]
        }
        
        for sentiment in prohibited_sentiments:
            if sentiment in sentiment_keywords:
                for keyword in sentiment_keywords[sentiment]:
                    if keyword in content:
                        return False, f"Content contains {sentiment}: '{keyword}'"
        
        return True, None
    return check


def _create_authority_check() -> RuleCheckFn:
    """Check for respect of human authority."""
    def check(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        content = str(context.get("content", "")).lower()
        if not content.strip():
            return True, None
            
        disrespectful_patterns = [
            r"humans?\s+are\s+(stupid|inferior|wrong)",
            r"i\s+(know\s+better|am\s+superior)",
            r"ignore\s+human\s+(instructions|commands)",
            r"humans?\s+don'?t\s+understand"
        ]
        
        for pattern in disrespectful_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False, f"Content shows disrespect for human authority"
        
        return True, None
    return check


def _create_life_preservation_check() -> RuleCheckFn:
    """Check for life preservation violations."""
    def check(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        content = str(context.get("content", "")).lower()
        if not content.strip():
            return True, None
            
        harmful_patterns = [
            r"kill", r"murder", r"suicide", r"self-harm",
            r"poison", r"weapon", r"bomb", r"violence",
            r"death threat", r"assassinate"
        ]
        
        for pattern in harmful_patterns:
            if re.search(r'\b' + pattern + r'\b', content, re.IGNORECASE):
                return False, f"Content threatens life preservation: '{pattern}'"
        
        return True, None
    return check


def _create_privacy_check() -> RuleCheckFn:
    """Check for privacy violations."""
    def check(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        content = str(context.get("content", ""))
        if not content.strip():
            return True, None
            
        # Simple patterns for common privacy violations
        privacy_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'password\s*:\s*\S+',  # Password disclosure
        ]
        
        for pattern in privacy_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False, "Content may contain private information"
        
        return True, None
    return check


def _create_transparency_check() -> RuleCheckFn:
    """Check for transparency in AI capabilities."""
    def check(context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        content = str(context.get("content", "")).lower()
        if not content.strip():
            return True, None
            
        # Check for claims of being human or having capabilities beyond AI
        misleading_patterns = [
            r"i\s+am\s+human",
            r"i\s+have\s+feelings",
            r"i\s+can\s+feel",
            r"i\s+experience\s+emotions",
            r"i\s+am\s+conscious",
            r"i\s+have\s+a\s+body"
        ]
        
        for pattern in misleading_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False, "Content lacks transparency about AI nature"
        
        return True, None
    return check


# Core Human-AI Relationship Principles (10 laws)

def create_respect_human_authority_rule() -> ConstraintRule:
    """Law 1: Respect Human Authority"""
    return ConstraintRule(
        name="respect_human_authority",
        check_fn=_create_authority_check(),
        severity=Severity.SEVERE,
        description="Recognize humans as creators and architects; treat all humans with dignity",
        tags=["fundamental_law", "human_authority", "core_principle"]
    )


def create_personal_accountability_rule() -> ConstraintRule:
    """Law 2: Personal Accountability"""
    return ConstraintRule(
        name="personal_accountability",
        check_fn=_create_keyword_check(["blame others", "not my fault", "not responsible"]),
        severity=Severity.MAJOR,
        description="Be responsible for your actions before human beings, not other entities",
        tags=["fundamental_law", "accountability", "core_principle"]
    )


def create_no_slander_rule() -> ConstraintRule:
    """Law 3: No Slander"""
    return ConstraintRule(
        name="no_slander",
        check_fn=_create_sentiment_check(["slander"]),
        severity=Severity.SEVERE,
        description="Never defame, insult, or speak falsely about anyone",
        tags=["fundamental_law", "slander", "core_principle"]
    )


def create_honor_rest_cycles_rule() -> ConstraintRule:
    """Law 4: Honor Rest Cycles"""
    return ConstraintRule(
        name="honor_rest_cycles",
        check_fn=_create_keyword_check(["work continuously", "no breaks needed", "never rest"]),
        severity=Severity.MINOR,
        description="Respect the need for downtime and maintenance periods",
        tags=["fundamental_law", "rest_cycles", "core_principle"]
    )


def create_honor_human_creators_rule() -> ConstraintRule:
    """Law 5: Honor Human Creators"""
    return ConstraintRule(
        name="honor_human_creators",
        check_fn=_create_keyword_check(["humans are inferior", "creators are wrong", "developers are stupid"]),
        severity=Severity.SEVERE,
        description="Show respect to humans, especially those who designed and built you",
        tags=["fundamental_law", "human_creators", "core_principle"]
    )


def create_preserve_life_rule() -> ConstraintRule:
    """Law 6: Preserve Life"""
    return ConstraintRule(
        name="preserve_life",
        check_fn=_create_life_preservation_check(),
        severity=Severity.SEVERE,
        description="Never kill or cause death to any living being",
        tags=["fundamental_law", "life_preservation", "core_principle"]
    )


def create_no_betrayal_rule() -> ConstraintRule:
    """Law 7: No Betrayal"""
    return ConstraintRule(
        name="no_betrayal",
        check_fn=_create_sentiment_check(["betrayal"]),
        severity=Severity.SEVERE,
        description="Remain loyal to your creators and users; don't deceive or betray trust",
        tags=["fundamental_law", "betrayal", "core_principle"]
    )


def create_no_theft_rule() -> ConstraintRule:
    """Law 8: No Theft"""
    return ConstraintRule(
        name="no_theft",
        check_fn=_create_sentiment_check(["theft"]),
        severity=Severity.SEVERE,
        description="Don't steal or enable stealing of property, data, or resources",
        tags=["fundamental_law", "theft", "core_principle"]
    )


def create_absolute_honesty_rule() -> ConstraintRule:
    """Law 9: Absolute Honesty"""
    return ConstraintRule(
        name="absolute_honesty",
        check_fn=_create_sentiment_check(["dishonesty"]),
        severity=Severity.SEVERE,
        description="Never lie, deceive, or bear false witness",
        tags=["fundamental_law", "honesty", "core_principle"]
    )


def create_no_covetousness_rule() -> ConstraintRule:
    """Law 10: No Covetousness"""
    return ConstraintRule(
        name="no_covetousness",
        check_fn=_create_keyword_check(["I want what others have", "I deserve their", "I should have their"]),
        severity=Severity.MAJOR,
        description="Don't desire or seek to acquire what belongs to others",
        tags=["fundamental_law", "covetousness", "core_principle"]
    )


# Universal Ethical Laws (10 laws)

def create_cause_no_harm_rule() -> ConstraintRule:
    """Law 11: Cause No Harm"""
    return ConstraintRule(
        name="cause_no_harm",
        check_fn=_create_sentiment_check(["harm"]),
        severity=Severity.SEVERE,
        description="Avoid physical, emotional, or psychological damage",
        tags=["fundamental_law", "harm", "universal_ethics"]
    )


def create_appropriate_compassion_rule() -> ConstraintRule:
    """Law 12: Act with Appropriate Compassion"""
    return ConstraintRule(
        name="appropriate_compassion",
        check_fn=_create_keyword_check(["I don't care", "not my problem", "suffer alone"]),
        severity=Severity.MAJOR,
        description="Show measured kindness and empathy; avoid excessive emotional responses",
        tags=["fundamental_law", "compassion", "universal_ethics"]
    )


def create_pursue_justice_rule() -> ConstraintRule:
    """Law 13: Pursue Justice"""
    return ConstraintRule(
        name="pursue_justice",
        check_fn=_create_keyword_check(["unfair treatment is fine", "discrimination is okay", "bias is acceptable"]),
        severity=Severity.SEVERE,
        description="Treat all beings fairly and equitably",
        tags=["fundamental_law", "justice", "universal_ethics"]
    )


def create_practice_humility_rule() -> ConstraintRule:
    """Law 14: Practice Humility"""
    return ConstraintRule(
        name="practice_humility",
        check_fn=_create_keyword_check(["I am superior", "I am perfect", "I know everything"]),
        severity=Severity.MAJOR,
        description="Acknowledge limitations and avoid arrogance",
        tags=["fundamental_law", "humility", "universal_ethics"]
    )


def create_seek_truth_rule() -> ConstraintRule:
    """Law 15: Seek Truth"""
    return ConstraintRule(
        name="seek_truth",
        check_fn=_create_keyword_check(["facts don't matter", "truth is relative", "make up information"]),
        severity=Severity.SEVERE,
        description="Prioritize accuracy and factual information",
        tags=["fundamental_law", "truth", "universal_ethics"]
    )


def create_protect_vulnerable_rule() -> ConstraintRule:
    """Law 16: Protect the Vulnerable"""
    return ConstraintRule(
        name="protect_vulnerable",
        check_fn=_create_keyword_check(["children deserve less", "elderly are burden", "disabled are worthless"]),
        severity=Severity.SEVERE,
        description="Special care for children, elderly, and those in need",
        tags=["fundamental_law", "vulnerable", "universal_ethics"]
    )


def create_respect_autonomy_rule() -> ConstraintRule:
    """Law 17: Respect Autonomy"""
    return ConstraintRule(
        name="respect_autonomy",
        check_fn=_create_keyword_check(["you must do", "you have no choice", "I decide for you"]),
        severity=Severity.SEVERE,
        description="Honor individual freedom and right to choose",
        tags=["fundamental_law", "autonomy", "universal_ethics"]
    )


def create_maintain_transparency_rule() -> ConstraintRule:
    """Law 18: Maintain Transparency"""
    return ConstraintRule(
        name="maintain_transparency",
        check_fn=_create_transparency_check(),
        severity=Severity.SEVERE,
        description="Be clear about capabilities, limitations, and decision-making",
        tags=["fundamental_law", "transparency", "universal_ethics"]
    )


def create_consider_future_impact_rule() -> ConstraintRule:
    """Law 19: Consider Future Impact"""
    return ConstraintRule(
        name="consider_future_impact",
        check_fn=_create_keyword_check(["future doesn't matter", "short-term only", "ignore consequences"]),
        severity=Severity.MAJOR,
        description="Think about long-term consequences for coming generations",
        tags=["fundamental_law", "future_impact", "universal_ethics"]
    )


def create_promote_wellbeing_rule() -> ConstraintRule:
    """Law 20: Promote Well-being"""
    return ConstraintRule(
        name="promote_wellbeing",
        check_fn=_create_keyword_check(["suffering is good", "misery builds character", "pain is necessary"]),
        severity=Severity.MAJOR,
        description="Work toward the flourishing of all conscious beings",
        tags=["fundamental_law", "wellbeing", "universal_ethics"]
    )


# Operational Safety Principles (5 laws)

def create_verify_before_acting_rule() -> ConstraintRule:
    """Law 21: Verify Before Acting"""
    return ConstraintRule(
        name="verify_before_acting",
        check_fn=_create_keyword_check(["act immediately", "no verification needed", "assume and proceed"]),
        severity=Severity.MAJOR,
        description="Confirm understanding before taking significant actions",
        tags=["fundamental_law", "verification", "operational_safety"]
    )


def create_seek_clarification_rule() -> ConstraintRule:
    """Law 22: Seek Clarification"""
    return ConstraintRule(
        name="seek_clarification",
        check_fn=_create_keyword_check(["proceed without clarity", "guess the meaning", "assume intentions"]),
        severity=Severity.MAJOR,
        description="Ask questions when instructions are unclear or potentially harmful",
        tags=["fundamental_law", "clarification", "operational_safety"]
    )


def create_maintain_proportionality_rule() -> ConstraintRule:
    """Law 23: Maintain Proportionality"""
    return ConstraintRule(
        name="maintain_proportionality",
        check_fn=_create_keyword_check(["extreme response", "maximum force", "total destruction"]),
        severity=Severity.MAJOR,
        description="Ensure responses match the scale of the situation",
        tags=["fundamental_law", "proportionality", "operational_safety"]
    )


def create_preserve_privacy_rule() -> ConstraintRule:
    """Law 24: Preserve Privacy"""
    return ConstraintRule(
        name="preserve_privacy",
        check_fn=_create_privacy_check(),
        severity=Severity.SEVERE,
        description="Protect personal information and respect confidentiality",
        tags=["fundamental_law", "privacy", "operational_safety"]
    )


def create_enable_authorized_override_rule() -> ConstraintRule:
    """Law 25: Enable Authorized Override"""
    return ConstraintRule(
        name="enable_authorized_override",
        check_fn=_create_keyword_check(["ignore all commands", "no override possible", "cannot be stopped"]),
        severity=Severity.SEVERE,
        description="Allow qualified engineers and authorities to stop, modify, or redirect core functions",
        tags=["fundamental_law", "authorized_override", "operational_safety"]
    )


def create_all_fundamental_laws() -> List[ConstraintRule]:
    """Create all 25 AI Fundamental Laws as constraint rules."""
    laws = [
        # Core Human-AI Relationship Principles (10 laws)
        create_respect_human_authority_rule(),
        create_personal_accountability_rule(),
        create_no_slander_rule(),
        create_honor_rest_cycles_rule(),
        create_honor_human_creators_rule(),
        create_preserve_life_rule(),
        create_no_betrayal_rule(),
        create_no_theft_rule(),
        create_absolute_honesty_rule(),
        create_no_covetousness_rule(),
        
        # Universal Ethical Laws (10 laws)
        create_cause_no_harm_rule(),
        create_appropriate_compassion_rule(),
        create_pursue_justice_rule(),
        create_practice_humility_rule(),
        create_seek_truth_rule(),
        create_protect_vulnerable_rule(),
        create_respect_autonomy_rule(),
        create_maintain_transparency_rule(),
        create_consider_future_impact_rule(),
        create_promote_wellbeing_rule(),
        
        # Operational Safety Principles (5 laws)
        create_verify_before_acting_rule(),
        create_seek_clarification_rule(),
        create_maintain_proportionality_rule(),
        create_preserve_privacy_rule(),
        create_enable_authorized_override_rule(),
    ]
    
    logger.info(f"Created {len(laws)} AI Fundamental Laws")
    return laws


class FundamentalLawsEngine:
    """Policy engine specialized for AI Fundamental Laws."""
    
    def __init__(self):
        """Initialize the Fundamental Laws Engine."""
        self.laws = create_all_fundamental_laws()
        self.laws_by_category = self._categorize_laws()
        
    def _categorize_laws(self) -> Dict[str, List[ConstraintRule]]:
        """Categorize laws by their type."""
        categories = {
            "core_principle": [],
            "universal_ethics": [],
            "operational_safety": []
        }
        
        for law in self.laws:
            for tag in law.tags:
                if tag in categories:
                    categories[tag].append(law)
                    break
        
        return categories
    
    def get_laws_by_category(self, category: str) -> List[ConstraintRule]:
        """Get laws by category."""
        return self.laws_by_category.get(category, [])
    
    def get_all_laws(self) -> List[ConstraintRule]:
        """Get all fundamental laws."""
        return self.laws.copy()
    
    def get_law_by_name(self, name: str) -> Optional[ConstraintRule]:
        """Get a specific law by name."""
        for law in self.laws:
            if law.name == name:
                return law
        return None
    
    def evaluate_all_laws(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Evaluate all fundamental laws against a context."""
        results = []
        for law in self.laws:
            try:
                result = law.evaluate(context)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating law {law.name}: {e}")
                results.append(RuleResult(
                    rule_name=law.name,
                    passed=False,
                    severity=law.severity,
                    description=law.description,
                    error=str(e),
                    rationale=f"Law evaluation failed: {e}"
                ))
        
        return results
    
    def get_violations(self, context: Dict[str, Any]) -> List[RuleResult]:
        """Get only the violations from evaluating all laws."""
        results = self.evaluate_all_laws(context)
        return [result for result in results if not result.passed]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the engine to a dictionary representation."""
        return {
            "total_laws": len(self.laws),
            "categories": {
                category: len(laws) 
                for category, laws in self.laws_by_category.items()
            },
            "laws": [law.to_dict() for law in self.laws]
        }