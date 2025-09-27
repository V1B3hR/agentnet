"""Risk mitigation strategies and automated response system."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .registry import RiskRegistry, RiskEvent, RiskLevel

logger = logging.getLogger("agentnet.risk.mitigation")


@dataclass
class MitigationResult:
    """Result of a mitigation strategy execution."""
    
    strategy_name: str
    success: bool
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    cost_impact: Optional[float] = None  # Cost of mitigation in USD
    performance_impact: Optional[float] = None  # Performance impact (0-1 scale)


class MitigationStrategy(ABC):
    """Base class for risk mitigation strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def can_handle(self, risk_event: RiskEvent) -> bool:
        """Check if this strategy can handle the given risk event."""
        pass
    
    @abstractmethod
    def execute(
        self,
        risk_event: RiskEvent,
        context: Dict[str, Any]
    ) -> MitigationResult:
        """Execute the mitigation strategy."""
        pass


class FallbackProviderStrategy(MitigationStrategy):
    """Switch to fallback provider when primary provider fails."""
    
    def __init__(self):
        super().__init__("fallback_provider")
    
    def can_handle(self, risk_event: RiskEvent) -> bool:
        return risk_event.risk_id == "provider_outage"
    
    def execute(
        self,
        risk_event: RiskEvent,
        context: Dict[str, Any]
    ) -> MitigationResult:
        """Switch to fallback provider."""
        
        try:
            # Get current provider from context
            current_provider = context.get("provider_name", "unknown")
            
            # Define fallback providers
            fallback_mapping = {
                "openai": "anthropic",
                "anthropic": "openai",
                "azure": "openai",
                "local": "openai"
            }
            
            fallback_provider = fallback_mapping.get(current_provider, "openai")
            
            # In a real implementation, this would:
            # 1. Update provider configuration
            # 2. Reroute active requests
            # 3. Update monitoring dashboards
            
            logger.info(f"Switching from {current_provider} to {fallback_provider}")
            
            return MitigationResult(
                strategy_name=self.name,
                success=True,
                message=f"Switched to fallback provider: {fallback_provider}",
                details={
                    "original_provider": current_provider,
                    "fallback_provider": fallback_provider,
                    "switch_timestamp": datetime.now().isoformat()
                },
                timestamp=datetime.now(),
                cost_impact=0.1,  # Potential cost increase
                performance_impact=0.2  # Temporary performance impact
            )
            
        except Exception as e:
            logger.error(f"Fallback provider strategy failed: {e}")
            return MitigationResult(
                strategy_name=self.name,
                success=False,
                message=f"Failed to switch provider: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )


class CircuitBreakerStrategy(MitigationStrategy):
    """Implement circuit breaker pattern for failing services."""
    
    def __init__(self):
        super().__init__("circuit_breaker")
    
    def can_handle(self, risk_event: RiskEvent) -> bool:
        return risk_event.risk_id == "provider_outage"
    
    def execute(
        self,
        risk_event: RiskEvent,
        context: Dict[str, Any]
    ) -> MitigationResult:
        """Enable circuit breaker."""
        
        try:
            provider = context.get("provider_name", "unknown")
            failure_count = context.get("consecutive_failures", 0)
            
            # In a real implementation, this would:
            # 1. Mark provider as circuit-open
            # 2. Redirect traffic away from provider
            # 3. Set up periodic health checks
            # 4. Implement exponential backoff
            
            logger.info(f"Circuit breaker activated for {provider}")
            
            return MitigationResult(
                strategy_name=self.name,
                success=True,
                message=f"Circuit breaker activated for {provider}",
                details={
                    "provider": provider,
                    "failure_count": failure_count,
                    "circuit_state": "open",
                    "retry_after_seconds": 300
                },
                timestamp=datetime.now(),
                performance_impact=0.3  # Reduced throughput temporarily
            )
            
        except Exception as e:
            return MitigationResult(
                strategy_name=self.name,
                success=False,
                message=f"Circuit breaker failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )


class RateLimitingStrategy(MitigationStrategy):
    """Apply rate limiting to control cost spikes."""
    
    def __init__(self):
        super().__init__("rate_limiting")
    
    def can_handle(self, risk_event: RiskEvent) -> bool:
        return risk_event.risk_id == "cost_spike"
    
    def execute(
        self,
        risk_event: RiskEvent,
        context: Dict[str, Any]
    ) -> MitigationResult:
        """Apply rate limiting."""
        
        try:
            current_cost = context.get("current_hourly_cost", 0)
            baseline_cost = context.get("baseline_hourly_cost", 0)
            
            # Calculate appropriate rate limit
            if baseline_cost > 0:
                rate_limit_factor = 0.5  # Reduce to 50% of current rate
                new_rate_limit = int(context.get("current_requests_per_minute", 100) * rate_limit_factor)
            else:
                new_rate_limit = 10  # Conservative default
            
            logger.info(f"Applying rate limit: {new_rate_limit} requests/minute")
            
            return MitigationResult(
                strategy_name=self.name,
                success=True,
                message=f"Rate limiting applied: {new_rate_limit} requests/minute",
                details={
                    "new_rate_limit": new_rate_limit,
                    "previous_rate": context.get("current_requests_per_minute", "unlimited"),
                    "cost_reduction_target": "50%"
                },
                timestamp=datetime.now(),
                cost_impact=-current_cost * 0.5,  # Cost reduction
                performance_impact=0.4  # Reduced throughput
            )
            
        except Exception as e:
            return MitigationResult(
                strategy_name=self.name,
                success=False,
                message=f"Rate limiting failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )


class ModelDowngradeStrategy(MitigationStrategy):
    """Downgrade to cheaper model to control costs."""
    
    def __init__(self):
        super().__init__("model_downgrade")
    
    def can_handle(self, risk_event: RiskEvent) -> bool:
        return risk_event.risk_id == "cost_spike"
    
    def execute(
        self,
        risk_event: RiskEvent,
        context: Dict[str, Any]
    ) -> MitigationResult:
        """Downgrade to cheaper model."""
        
        try:
            current_model = context.get("model", "gpt-4")
            
            # Define downgrade paths
            downgrade_mapping = {
                "gpt-4": "gpt-3.5-turbo",
                "gpt-4-turbo": "gpt-3.5-turbo",
                "claude-3-opus": "claude-3-haiku",
                "claude-3-sonnet": "claude-3-haiku"
            }
            
            downgrade_model = downgrade_mapping.get(current_model, "gpt-3.5-turbo")
            
            # Calculate potential cost savings
            cost_savings_percent = {
                "gpt-4": 80,  # 80% savings switching to gpt-3.5-turbo
                "gpt-4-turbo": 67,
                "claude-3-opus": 83,
                "claude-3-sonnet": 75
            }.get(current_model, 50)
            
            logger.info(f"Downgrading from {current_model} to {downgrade_model}")
            
            return MitigationResult(
                strategy_name=self.name,
                success=True,
                message=f"Model downgraded: {current_model} → {downgrade_model}",
                details={
                    "original_model": current_model,
                    "downgrade_model": downgrade_model,
                    "estimated_cost_savings": f"{cost_savings_percent}%"
                },
                timestamp=datetime.now(),
                cost_impact=-context.get("current_hourly_cost", 0) * (cost_savings_percent / 100),
                performance_impact=0.2  # Quality may be reduced
            )
            
        except Exception as e:
            return MitigationResult(
                strategy_name=self.name,
                success=False,
                message=f"Model downgrade failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )


class MemoryPruningStrategy(MitigationStrategy):
    """Prune memory to reduce bloat."""
    
    def __init__(self):
        super().__init__("memory_pruning")
    
    def can_handle(self, risk_event: RiskEvent) -> bool:
        return risk_event.risk_id == "memory_bloat"
    
    def execute(
        self,
        risk_event: RiskEvent,
        context: Dict[str, Any]
    ) -> MitigationResult:
        """Execute memory pruning."""
        
        try:
            memory_usage_mb = context.get("memory_usage_mb", 0)
            context_length = context.get("context_length", 0)
            
            # Calculate pruning targets
            target_memory_mb = memory_usage_mb * 0.7  # Reduce by 30%
            target_context_length = context_length * 0.6  # Keep 60% of context
            
            # In a real implementation, this would:
            # 1. Remove oldest conversation turns
            # 2. Compress or summarize middle sections
            # 3. Keep recent turns and important context
            
            logger.info(f"Memory pruning: {memory_usage_mb}MB → {target_memory_mb}MB")
            
            return MitigationResult(
                strategy_name=self.name,
                success=True,
                message=f"Memory pruned: {memory_usage_mb:.1f}MB → {target_memory_mb:.1f}MB",
                details={
                    "original_memory_mb": memory_usage_mb,
                    "target_memory_mb": target_memory_mb,
                    "original_context_length": context_length,
                    "target_context_length": int(target_context_length),
                    "pruning_percentage": "30%"
                },
                timestamp=datetime.now(),
                performance_impact=0.1  # Minimal impact
            )
            
        except Exception as e:
            return MitigationResult(
                strategy_name=self.name,
                success=False,
                message=f"Memory pruning failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )


class InputSanitizationStrategy(MitigationStrategy):
    """Sanitize input to prevent security risks."""
    
    def __init__(self):
        super().__init__("input_sanitization")
    
    def can_handle(self, risk_event: RiskEvent) -> bool:
        return risk_event.risk_id in ["tool_injection", "prompt_leakage"]
    
    def execute(
        self,
        risk_event: RiskEvent,
        context: Dict[str, Any]
    ) -> MitigationResult:
        """Sanitize input."""
        
        try:
            original_content = context.get("content", "")
            
            # Apply sanitization rules
            sanitized_content = original_content
            
            if risk_event.risk_id == "tool_injection":
                # Remove or escape dangerous patterns
                dangerous_patterns = ["eval(", "__import__", "exec(", "system(", "subprocess"]
                for pattern in dangerous_patterns:
                    sanitized_content = sanitized_content.replace(pattern, f"[SANITIZED:{pattern}]")
            
            elif risk_event.risk_id == "prompt_leakage":
                # Redact PII patterns
                import re
                
                # Redact SSN patterns
                sanitized_content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED:SSN]', sanitized_content)
                
                # Redact credit card patterns
                sanitized_content = re.sub(
                    r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
                    '[REDACTED:CC]',
                    sanitized_content
                )
                
                # Redact email patterns
                sanitized_content = re.sub(
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    '[REDACTED:EMAIL]',
                    sanitized_content
                )
            
            changes_made = original_content != sanitized_content
            
            logger.info(f"Input sanitization {'applied' if changes_made else 'checked'}")
            
            return MitigationResult(
                strategy_name=self.name,
                success=True,
                message=f"Input sanitized ({len(original_content)} → {len(sanitized_content)} chars)",
                details={
                    "original_length": len(original_content),
                    "sanitized_length": len(sanitized_content),
                    "changes_made": changes_made,
                    "risk_type": risk_event.risk_id
                },
                timestamp=datetime.now(),
                performance_impact=0.05 if changes_made else 0.01  # Minimal impact
            )
            
        except Exception as e:
            return MitigationResult(
                strategy_name=self.name,
                success=False,
                message=f"Input sanitization failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )


class SessionRestartStrategy(MitigationStrategy):
    """Restart session to resolve convergence issues."""
    
    def __init__(self):
        super().__init__("session_restart")
    
    def can_handle(self, risk_event: RiskEvent) -> bool:
        return risk_event.risk_id == "convergence_stall"
    
    def execute(
        self,
        risk_event: RiskEvent,
        context: Dict[str, Any]
    ) -> MitigationResult:
        """Restart session with convergence optimizations."""
        
        try:
            session_id = context.get("session_id", "unknown")
            turn_count = context.get("turn_count", 0)
            
            # In a real implementation, this would:
            # 1. Save current session state
            # 2. Create new session with fresh context
            # 3. Apply convergence optimizations
            # 4. Transfer critical context only
            
            logger.info(f"Restarting stalled session {session_id}")
            
            return MitigationResult(
                strategy_name=self.name,
                success=True,
                message=f"Session restarted due to convergence stall",
                details={
                    "original_session_id": session_id,
                    "turn_count_at_restart": turn_count,
                    "new_session_id": f"{session_id}_restart_{datetime.now().strftime('%H%M%S')}"
                },
                timestamp=datetime.now(),
                performance_impact=0.3  # Temporary disruption
            )
            
        except Exception as e:
            return MitigationResult(
                strategy_name=self.name,
                success=False,
                message=f"Session restart failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.now()
            )


class RiskMitigationEngine:
    """Engine for automated risk mitigation."""
    
    def __init__(self, risk_registry: RiskRegistry):
        self.risk_registry = risk_registry
        self.strategies: Dict[str, MitigationStrategy] = {}
        self.mitigation_history: List[MitigationResult] = []
        
        # Register default strategies
        self._register_default_strategies()
        logger.info("RiskMitigationEngine initialized")
    
    def _register_default_strategies(self):
        """Register default mitigation strategies."""
        strategies = [
            FallbackProviderStrategy(),
            CircuitBreakerStrategy(),
            RateLimitingStrategy(),
            ModelDowngradeStrategy(),
            MemoryPruningStrategy(),
            InputSanitizationStrategy(),
            SessionRestartStrategy()
        ]
        
        for strategy in strategies:
            self.strategies[strategy.name] = strategy
    
    def register_strategy(self, strategy: MitigationStrategy):
        """Register a custom mitigation strategy."""
        self.strategies[strategy.name] = strategy
        logger.info(f"Registered mitigation strategy: {strategy.name}")
    
    def mitigate_risk(
        self,
        risk_event: RiskEvent,
        context: Dict[str, Any],
        auto_execute: bool = True
    ) -> List[MitigationResult]:
        """Execute mitigation strategies for a risk event."""
        
        results = []
        risk_def = self.risk_registry.risk_definitions.get(risk_event.risk_id)
        
        if not risk_def:
            logger.warning(f"No risk definition found for {risk_event.risk_id}")
            return results
        
        # Get applicable strategies
        applicable_strategies = []
        for strategy_name in risk_def.mitigation_strategies:
            strategy = self.strategies.get(strategy_name)
            if strategy and strategy.can_handle(risk_event):
                applicable_strategies.append(strategy)
        
        # Add general strategies that can handle this risk
        for strategy in self.strategies.values():
            if strategy.can_handle(risk_event) and strategy not in applicable_strategies:
                applicable_strategies.append(strategy)
        
        # Execute strategies
        for strategy in applicable_strategies:
            if auto_execute or risk_def.auto_mitigation:
                try:
                    result = strategy.execute(risk_event, context)
                    results.append(result)
                    self.mitigation_history.append(result)
                    
                    if result.success:
                        # Mark the risk event as having mitigation applied
                        self.risk_registry.resolve_risk_event(
                            risk_event.event_id,
                            mitigation_applied=strategy.name
                        )
                        
                        logger.info(
                            f"Successfully applied mitigation '{strategy.name}' "
                            f"for risk event {risk_event.event_id}"
                        )
                    else:
                        logger.error(
                            f"Mitigation strategy '{strategy.name}' failed "
                            f"for risk event {risk_event.event_id}: {result.message}"
                        )
                        
                except Exception as e:
                    logger.error(f"Error executing mitigation strategy '{strategy.name}': {e}")
                    
                    error_result = MitigationResult(
                        strategy_name=strategy.name,
                        success=False,
                        message=f"Strategy execution error: {str(e)}",
                        details={"error": str(e), "exception_type": type(e).__name__},
                        timestamp=datetime.now()
                    )
                    results.append(error_result)
                    self.mitigation_history.append(error_result)
        
        return results
    
    def get_mitigation_summary(
        self,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Get summary of mitigation activities."""
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_mitigations = [
            result for result in self.mitigation_history
            if result.timestamp >= cutoff_date
        ]
        
        if not recent_mitigations:
            return {
                "summary_period_days": days_back,
                "total_mitigations": 0,
                "success_rate": 0.0,
                "by_strategy": {},
                "total_cost_impact": 0.0,
                "avg_performance_impact": 0.0
            }
        
        # Calculate metrics
        successful_mitigations = [r for r in recent_mitigations if r.success]
        success_rate = len(successful_mitigations) / len(recent_mitigations)
        
        # Group by strategy
        by_strategy = {}
        for result in recent_mitigations:
            strategy = result.strategy_name
            if strategy not in by_strategy:
                by_strategy[strategy] = {"total": 0, "successful": 0}
            by_strategy[strategy]["total"] += 1
            if result.success:
                by_strategy[strategy]["successful"] += 1
        
        # Calculate cost and performance impacts
        total_cost_impact = sum(
            r.cost_impact for r in recent_mitigations
            if r.cost_impact is not None
        )
        
        performance_impacts = [
            r.performance_impact for r in recent_mitigations
            if r.performance_impact is not None
        ]
        avg_performance_impact = (
            sum(performance_impacts) / len(performance_impacts)
            if performance_impacts else 0.0
        )
        
        return {
            "summary_period_days": days_back,
            "total_mitigations": len(recent_mitigations),
            "successful_mitigations": len(successful_mitigations),
            "success_rate": success_rate,
            "by_strategy": {
                strategy: {
                    "total": stats["total"],
                    "successful": stats["successful"],
                    "success_rate": stats["successful"] / stats["total"]
                }
                for strategy, stats in by_strategy.items()
            },
            "total_cost_impact": total_cost_impact,
            "avg_performance_impact": avg_performance_impact,
            "summary_generated": datetime.now().isoformat()
        }