"""
Cost-Efficient Token Utilization Tracking for AgentNet

Provides highly accurate, memory-efficient token usage analysis, tracks direct
cost savings from caching, and offers advanced optimization insights.
"""

import time
import logging
import statistics
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TokenType(str, Enum):
    """Types of tokens tracked in the system."""
    INPUT = "input"
    OUTPUT = "output"


class TokenCategory(str, Enum):
    """Categories of token usage for analysis."""
    REASONING = "reasoning"
    TOOL_CALLS = "tool_calls"
    MEMORY = "memory"
    POLICY = "policy"
    DIALOGUE = "dialogue"
    SYSTEM = "system"


@dataclass
class TokenMetrics:
    """Comprehensive token usage metrics for a single agent turn."""
    agent_id: str
    turn_id: str
    timestamp: float
    model_name: str

    # Basic token counts
    input_tokens: int = 0
    output_tokens: int = 0

    # Cost metrics based on dual-rate model
    turn_cost_usd: float = 0.0

    # Context and efficiency metrics
    context_length: int = 0
    io_redundancy_ratio: float = 0.0  # Ratio of input repeated in output

    # Performance metrics
    processing_time_seconds: float = 0.0

    # Quality metrics
    output_quality_score: float = 0.0  # From 0.0 to 1.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / self.processing_time_seconds if self.processing_time_seconds > 0 else 0.0

    @property
    def efficiency_score(self) -> float:
        """Overall efficiency score combining cost, redundancy, and quality."""
        # Lower cost is better (invert score)
        cost_score = 1.0 - min(self.turn_cost_usd / 0.10, 1.0)  # Normalize against a $0.10 turn
        # Lower redundancy is better
        redundancy_score = 1.0 - self.io_redundancy_ratio
        
        # Weighted average
        return self.output_quality_score * 0.5 + redundancy_score * 0.3 + cost_score * 0.2


class TokenUtilizationTracker:
    """
    Tracks, analyzes, and provides cost-saving insights on token utilization.
    Implemented as a singleton to ensure a single source of truth.
    """
    _instance: Optional["TokenUtilizationTracker"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TokenUtilizationTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_history: int = 10000, cost_models: Optional[Dict] = None):
        # Prevent re-initialization
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._metrics: deque[TokenMetrics] = deque(maxlen=max_history)
        self._cost_saved_by_caching_usd: float = 0.0
        
        default_cost_models = {
            "default": {"input": 0.001, "output": 0.002},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        }
        self._cost_models = cost_models or default_cost_models
        self._initialized = True

    def _get_cost_rates(self, model_name: str) -> Dict[str, float]:
        """Safely retrieves the cost rates for a given model."""
        return self._cost_models.get(model_name, self._cost_models["default"])

    def _calculate_cost(self, input_tokens: int, output_tokens: int, model_name: str) -> float:
        """Calculates cost using the accurate dual-rate model."""
        rates = self._get_cost_rates(model_name)
        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]
        return input_cost + output_cost

    def _calculate_io_redundancy(self, prompt: str, response: str) -> float:
        """Estimates the ratio of the prompt repeated verbatim in the response."""
        if not prompt or not response or len(prompt) > 5000: # Safety guard
            return 0.0
        
        # Use a simplified approach for performance: check for long common substrings
        # A more complex approach could use diffing libraries
        escaped_prompt = re.escape(prompt[:200]) # Check first 200 chars
        if re.search(escaped_prompt, response, re.IGNORECASE):
            return len(prompt) / len(response) if len(response) > len(prompt) else 1.0
        return 0.0

    def record_token_usage(
        self,
        agent_id: str,
        turn_id: str,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        context_length: int = 0,
        processing_time_seconds: float = 0.0,
        output_quality_score: float = 0.8,
        prompt_text: Optional[str] = None,
        response_text: Optional[str] = None,
    ) -> TokenMetrics:
        """Record comprehensive token usage for a turn with accurate cost calculation."""
        turn_cost = self._calculate_cost(input_tokens, output_tokens, model_name)
        
        redundancy_ratio = 0.0
        if prompt_text and response_text:
            redundancy_ratio = self._calculate_io_redundancy(prompt_text, response_text)

        metrics = TokenMetrics(
            agent_id=agent_id,
            turn_id=turn_id,
            timestamp=time.time(),
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            turn_cost_usd=turn_cost,
            context_length=context_length,
            io_redundancy_ratio=redundancy_ratio,
            processing_time_seconds=processing_time_seconds,
            output_quality_score=output_quality_score,
        )
        self._metrics.append(metrics)
        return metrics

    def record_cache_hit(
        self, agent_id: str, turn_id: str, cached_input_tokens: int, cached_output_tokens: int, model_name: str
    ):
        """Record a cache hit to explicitly track cost savings."""
        cost_saved = self._calculate_cost(cached_input_tokens, cached_output_tokens, model_name)
        self._cost_saved_by_caching_usd += cost_saved
        logger.info(
            f"Cache hit for agent '{agent_id}' (turn: {turn_id}). "
            f"Saved ~${cost_saved:.6f} and {cached_input_tokens + cached_output_tokens} tokens."
        )

    def get_system_overview(self) -> Dict[str, Any]:
        """Get a system-wide token utilization and cost-efficiency overview."""
        if not self._metrics:
            return {"message": "No metrics recorded yet."}

        total_cost = sum(m.turn_cost_usd for m in self._metrics)
        
        # Trend analysis: compare last 10% of activity to the overall average
        trend_slice_index = -max(1, len(self._metrics) // 10)
        recent_metrics = list(self._metrics)[trend_slice_index:]
        
        avg_cost_per_turn = total_cost / len(self._metrics)
        avg_cost_per_turn_recent = sum(m.turn_cost_usd for m in recent_metrics) / len(recent_metrics)

        return {
            "overview": {
                "total_turns_tracked": len(self._metrics),
                "total_tokens_processed": sum(m.total_tokens for m in self._metrics),
                "total_cost_usd": total_cost,
                "total_cost_saved_by_caching_usd": self._cost_saved_by_caching_usd,
                "avg_efficiency_score": statistics.mean(m.efficiency_score for m in self._metrics),
                "avg_io_redundancy_ratio": statistics.mean(m.io_redundancy_ratio for m in self._metrics),
            },
            "cost_trends": {
                "avg_cost_per_turn_usd": avg_cost_per_turn,
                "avg_cost_per_turn_recent_usd": avg_cost_per_turn_recent,
                "trend": "rising" if avg_cost_per_turn_recent > avg_cost_per_turn else "falling",
            },
            "recommendations": self.generate_optimization_recommendations(),
        }

    def generate_optimization_recommendations(self) -> List[str]:
        """Generate actionable optimization recommendations based on collected data."""
        if not self._metrics:
            return ["Not enough data to generate recommendations."]
            
        recommendations = set()
        avg_redundancy = statistics.mean(m.io_redundancy_ratio for m in self._metrics)
        avg_efficiency = statistics.mean(m.efficiency_score for m in self._metrics)
        
        high_cost_turns = sorted(self._metrics, key=lambda m: m.turn_cost_usd, reverse=True)[:5]
        
        if avg_redundancy > 0.15:
            recommendations.add(
                f"High I/O Redundancy ({avg_redundancy:.1%}): Review prompts to avoid repeating input in the output. "
                "Use instructions like 'Do not repeat the question in your answer.'"
            )
        
        if avg_efficiency < 0.6:
            recommendations.add(
                f"Low Efficiency Score ({avg_efficiency:.2f}): Prompts may be unfocused. "
                "Improve prompt clarity and quality scoring to enhance relevance."
            )
            
        if high_cost_turns and high_cost_turns[0].turn_cost_usd > 0.05:
            expensive_models = {m.model_name for m in high_cost_turns}
            recommendations.add(
                f"High-Cost Turns Detected: Review tasks using models like {', '.join(expensive_models)}. "
                "Consider if a cheaper model could suffice for these tasks."
            )
            
        if self._cost_saved_by_caching_usd == 0:
            recommendations.add(
                "No Cache Savings: Implement response caching for identical or highly similar prompts to significantly reduce costs."
            )

        return list(recommendations) if recommendations else ["System appears to be running efficiently."]

# --- Global Singleton Management ---
_global_tracker_instance: Optional[TokenUtilizationTracker] = None

def get_token_tracker(**kwargs) -> TokenUtilizationTracker:
    """
    Get the configured global token tracker instance.
    Initializes it on first call.
    """
    global _global_tracker_instance
    if _global_tracker_instance is None:
        _global_tracker_instance = TokenUtilizationTracker(**kwargs)
    return _global_tracker_instance
