"""Risk registry and core risk data structures."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("agentnet.risk")


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class RiskStatus(Enum):
    """Risk status tracking."""
    IDENTIFIED = "identified"
    ASSESSING = "assessing"
    MITIGATING = "mitigating"
    MONITORING = "monitoring"
    CLOSED = "closed"
    ACCEPTED = "accepted"


class RiskCategory(Enum):
    """Risk categories based on AgentNet architecture."""
    PROVIDER_OUTAGE = "provider_outage"
    POLICY_FALSE_POSITIVE = "policy_false_positive" 
    COST_SPIKE = "cost_spike"
    MEMORY_BLOAT = "memory_bloat"
    TOOL_INJECTION = "tool_injection"
    CONVERGENCE_STALL = "convergence_stall"
    PROMPT_LEAKAGE = "prompt_leakage"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"


@dataclass
class Risk:
    """Core risk data structure."""
    
    risk_id: str
    title: str
    description: str
    category: RiskCategory
    level: RiskLevel
    status: RiskStatus
    probability: float  # 0.0 to 1.0
    impact: float  # 0.0 to 1.0 
    risk_score: float  # probability * impact
    
    # Tracking
    identified_date: datetime
    identified_by: str
    last_updated: datetime
    updated_by: str
    target_resolution_date: Optional[datetime] = None
    actual_resolution_date: Optional[datetime] = None
    
    # Context
    affected_components: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Other risk IDs
    
    # Mitigation
    mitigation_strategies: List[str] = field(default_factory=list)
    current_controls: List[str] = field(default_factory=list)
    residual_probability: Optional[float] = None
    residual_impact: Optional[float] = None
    residual_risk_score: Optional[float] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived fields."""
        self.risk_score = self.probability * self.impact
        
        if self.residual_probability is not None and self.residual_impact is not None:
            self.residual_risk_score = self.residual_probability * self.residual_impact


class RiskRegistry:
    """Central registry for managing risks."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or "risk_registry.json")
        self.risks: Dict[str, Risk] = {}
        self._load_from_storage()
        self._initialize_default_risks()
    
    def _initialize_default_risks(self):
        """Initialize with known AgentNet risks from documentation."""
        
        default_risks = [
            {
                "risk_id": "RISK-001",
                "title": "Provider Outage",
                "description": "External AI provider experiences outages causing service degradation",
                "category": RiskCategory.PROVIDER_OUTAGE,
                "level": RiskLevel.HIGH,
                "probability": 0.3,
                "impact": 0.8,
                "affected_components": ["providers", "inference", "api"],
                "triggers": ["Network connectivity issues", "Provider maintenance", "Regional failures"],
                "current_controls": ["Health checks", "Retry logic"],
                "mitigation_strategies": ["Fallback providers", "Circuit breaker pattern"],
            },
            {
                "risk_id": "RISK-002", 
                "title": "Policy False Positives",
                "description": "Governance policies incorrectly flag legitimate operations causing user frustration",
                "category": RiskCategory.POLICY_FALSE_POSITIVE,
                "level": RiskLevel.MEDIUM,
                "probability": 0.4,
                "impact": 0.6,
                "affected_components": ["governance", "policy", "monitors"],
                "triggers": ["Over-restrictive policies", "Context misinterpretation", "Rule conflicts"],
                "current_controls": ["Policy testing", "Severity tiers"],
                "mitigation_strategies": ["Override tokens", "Policy tuning", "Human review"],
            },
            {
                "risk_id": "RISK-003",
                "title": "Token Cost Spike", 
                "description": "Unexpected increase in token usage leading to budget overruns",
                "category": RiskCategory.COST_SPIKE,
                "level": RiskLevel.HIGH,
                "probability": 0.5,
                "impact": 0.7,
                "affected_components": ["cost_tracking", "agents", "inference"],
                "triggers": ["Complex queries", "Model changes", "Increased usage"],
                "current_controls": ["Cost monitoring", "Usage tracking"],
                "mitigation_strategies": ["Spend alerts", "Model downgrade", "Usage limits"],
            },
            {
                "risk_id": "RISK-004",
                "title": "Memory Bloat",
                "description": "Memory usage grows unbounded causing increased latency and failures",
                "category": RiskCategory.MEMORY_BLOAT,
                "level": RiskLevel.MEDIUM,
                "probability": 0.3,
                "impact": 0.7,
                "affected_components": ["memory", "sessions", "agents"],
                "triggers": ["Long conversations", "Memory leaks", "Large context windows"],
                "current_controls": ["Memory monitoring", "Session limits"],
                "mitigation_strategies": ["Memory summaries", "Context pruning", "Session rotation"],
            },
            {
                "risk_id": "RISK-005",
                "title": "Tool Injection",
                "description": "Malicious input causes unauthorized tool execution leading to data exfiltration",
                "category": RiskCategory.TOOL_INJECTION,
                "level": RiskLevel.CRITICAL,
                "probability": 0.2,
                "impact": 0.9,
                "affected_components": ["tools", "security", "execution"],
                "triggers": ["Crafted prompts", "Input validation bypass", "Privilege escalation"],
                "current_controls": ["Input validation", "Tool sandboxing"],
                "mitigation_strategies": ["Schema validation", "Execution limits", "Audit logging"],
            },
            {
                "risk_id": "RISK-006",
                "title": "Convergence Stall",
                "description": "Agent reasoning fails to converge leading to indefinitely long sessions",
                "category": RiskCategory.CONVERGENCE_STALL,
                "level": RiskLevel.MEDIUM,
                "probability": 0.4,
                "impact": 0.5,
                "affected_components": ["reasoning", "agents", "sessions"],
                "triggers": ["Complex problems", "Circular logic", "Insufficient information"],
                "current_controls": ["Iteration limits", "Timeout enforcement"],
                "mitigation_strategies": ["Stagnation detection", "Hard caps", "Progress monitoring"],
            },
            {
                "risk_id": "RISK-007",
                "title": "Prompt Leakage",
                "description": "Sensitive information in prompts or responses leads to compliance breaches", 
                "category": RiskCategory.PROMPT_LEAKAGE,
                "level": RiskLevel.HIGH,
                "probability": 0.3,
                "impact": 0.8,
                "affected_components": ["compliance", "data_handling", "audit"],
                "triggers": ["PII in prompts", "Sensitive responses", "Logging exposure"],
                "current_controls": ["Data classification", "Access controls"],
                "mitigation_strategies": ["Secret scanning", "Data redaction", "Encrypted storage"],
            },
        ]
        
        # Only add defaults if registry is empty
        if not self.risks:
            for risk_data in default_risks:
                risk = Risk(
                    risk_id=risk_data["risk_id"],
                    title=risk_data["title"],
                    description=risk_data["description"],
                    category=risk_data["category"],
                    level=risk_data["level"],
                    status=RiskStatus.IDENTIFIED,
                    probability=risk_data["probability"],
                    impact=risk_data["impact"],
                    risk_score=risk_data["probability"] * risk_data["impact"],
                    identified_date=datetime.now(),
                    identified_by="system_default",
                    last_updated=datetime.now(),
                    updated_by="system_default",
                    affected_components=risk_data.get("affected_components", []),
                    triggers=risk_data.get("triggers", []),
                    current_controls=risk_data.get("current_controls", []),
                    mitigation_strategies=risk_data.get("mitigation_strategies", []),
                )
                self.risks[risk.risk_id] = risk
            
            self._save_to_storage()
            logger.info(f"Initialized risk registry with {len(default_risks)} default risks")
    
    def register_risk(self, risk: Risk) -> None:
        """Register a new risk in the registry."""
        self.risks[risk.risk_id] = risk
        self._save_to_storage()
        logger.info(f"Registered risk {risk.risk_id}: {risk.title}")
    
    def update_risk(self, risk_id: str, **updates) -> Risk:
        """Update an existing risk."""
        if risk_id not in self.risks:
            raise ValueError(f"Risk {risk_id} not found")
        
        risk = self.risks[risk_id]
        
        # Update fields
        for field_name, value in updates.items():
            if hasattr(risk, field_name):
                setattr(risk, field_name, value)
        
        # Always update tracking fields
        risk.last_updated = datetime.now()
        risk.updated_by = updates.get("updated_by", "system")
        
        # Recalculate scores if probability or impact changed
        if "probability" in updates or "impact" in updates:
            risk.risk_score = risk.probability * risk.impact
        
        if "residual_probability" in updates or "residual_impact" in updates:
            if risk.residual_probability is not None and risk.residual_impact is not None:
                risk.residual_risk_score = risk.residual_probability * risk.residual_impact
        
        self._save_to_storage()
        logger.info(f"Updated risk {risk_id}")
        
        return risk
    
    def get_risk(self, risk_id: str) -> Optional[Risk]:
        """Get a specific risk by ID."""
        return self.risks.get(risk_id)
    
    def get_risks_by_category(self, category: RiskCategory) -> List[Risk]:
        """Get all risks in a specific category."""
        return [risk for risk in self.risks.values() if risk.category == category]
    
    def get_risks_by_level(self, level: RiskLevel) -> List[Risk]:
        """Get all risks at a specific level."""
        return [risk for risk in self.risks.values() if risk.level == level]
    
    def get_risks_by_status(self, status: RiskStatus) -> List[Risk]:
        """Get all risks with a specific status."""
        return [risk for risk in self.risks.values() if risk.status == status]
    
    def get_active_risks(self) -> List[Risk]:
        """Get all active (non-closed, non-accepted) risks."""
        active_statuses = {RiskStatus.IDENTIFIED, RiskStatus.ASSESSING, 
                          RiskStatus.MITIGATING, RiskStatus.MONITORING}
        return [risk for risk in self.risks.values() if risk.status in active_statuses]
    
    def get_high_priority_risks(self) -> List[Risk]:
        """Get high priority risks (high/critical level or high risk score)."""
        return [risk for risk in self.risks.values() 
                if risk.level in {RiskLevel.HIGH, RiskLevel.CRITICAL} or risk.risk_score > 0.6]
    
    def get_overdue_risks(self) -> List[Risk]:
        """Get risks that are overdue for resolution."""
        now = datetime.now()
        return [risk for risk in self.risks.values() 
                if risk.target_resolution_date and risk.target_resolution_date < now 
                and risk.status not in {RiskStatus.CLOSED, RiskStatus.ACCEPTED}]
    
    def search_risks(self, query: str) -> List[Risk]:
        """Search risks by title, description, or tags."""
        query_lower = query.lower()
        matches = []
        
        for risk in self.risks.values():
            if (query_lower in risk.title.lower() or 
                query_lower in risk.description.lower() or
                any(query_lower in tag.lower() for tag in risk.tags)):
                matches.append(risk)
        
        return matches
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get statistical overview of the risk registry."""
        total_risks = len(self.risks)
        if total_risks == 0:
            return {"total_risks": 0}
        
        # Status distribution
        status_counts = {}
        for status in RiskStatus:
            status_counts[status.value] = len(self.get_risks_by_status(status))
        
        # Level distribution  
        level_counts = {}
        for level in RiskLevel:
            level_counts[level.value] = len(self.get_risks_by_level(level))
        
        # Category distribution
        category_counts = {}
        for category in RiskCategory:
            category_counts[category.value] = len(self.get_risks_by_category(category))
        
        # Risk scores
        risk_scores = [risk.risk_score for risk in self.risks.values()]
        avg_risk_score = sum(risk_scores) / len(risk_scores)
        max_risk_score = max(risk_scores)
        
        # Active risks
        active_count = len(self.get_active_risks())
        high_priority_count = len(self.get_high_priority_risks())
        overdue_count = len(self.get_overdue_risks())
        
        return {
            "total_risks": total_risks,
            "active_risks": active_count,
            "high_priority_risks": high_priority_count,
            "overdue_risks": overdue_count,
            "avg_risk_score": round(avg_risk_score, 3),
            "max_risk_score": round(max_risk_score, 3),
            "status_distribution": status_counts,
            "level_distribution": level_counts,
            "category_distribution": category_counts,
        }
    
    def _load_from_storage(self):
        """Load risks from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            for risk_data in data.get("risks", []):
                # Convert datetime strings back to datetime objects
                risk_data["identified_date"] = datetime.fromisoformat(risk_data["identified_date"])
                risk_data["last_updated"] = datetime.fromisoformat(risk_data["last_updated"])
                
                if risk_data.get("target_resolution_date"):
                    risk_data["target_resolution_date"] = datetime.fromisoformat(risk_data["target_resolution_date"])
                if risk_data.get("actual_resolution_date"):
                    risk_data["actual_resolution_date"] = datetime.fromisoformat(risk_data["actual_resolution_date"])
                
                # Convert enums
                risk_data["category"] = RiskCategory(risk_data["category"])
                risk_data["level"] = RiskLevel(risk_data["level"])
                risk_data["status"] = RiskStatus(risk_data["status"])
                
                # Convert tags back to set
                risk_data["tags"] = set(risk_data.get("tags", []))
                
                risk = Risk(**risk_data)
                self.risks[risk.risk_id] = risk
            
            logger.info(f"Loaded {len(self.risks)} risks from storage")
            
        except Exception as e:
            logger.error(f"Error loading risks from storage: {e}")
    
    def _save_to_storage(self):
        """Save risks to storage."""
        try:
            # Prepare data for JSON serialization
            risks_data = []
            for risk in self.risks.values():
                risk_dict = asdict(risk)
                
                # Convert datetime objects to strings
                risk_dict["identified_date"] = risk.identified_date.isoformat()
                risk_dict["last_updated"] = risk.last_updated.isoformat()
                
                if risk.target_resolution_date:
                    risk_dict["target_resolution_date"] = risk.target_resolution_date.isoformat()
                if risk.actual_resolution_date:
                    risk_dict["actual_resolution_date"] = risk.actual_resolution_date.isoformat()
                
                # Convert enums to strings
                risk_dict["category"] = risk.category.value
                risk_dict["level"] = risk.level.value
                risk_dict["status"] = risk.status.value
                
                # Convert set to list
                risk_dict["tags"] = list(risk.tags)
                
                risks_data.append(risk_dict)
            
            data = {
                "metadata": {
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat(),
                    "total_risks": len(risks_data),
                },
                "risks": risks_data,
            }
            
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving risks to storage: {e}")
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export all risks to a dictionary for reporting."""
        return {
            "registry_metadata": {
                "total_risks": len(self.risks),
                "exported_at": datetime.now().isoformat(),
            },
            "risks": [asdict(risk) for risk in self.risks.values()],
            "statistics": self.get_risk_statistics(),
        }