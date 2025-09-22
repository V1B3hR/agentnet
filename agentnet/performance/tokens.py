"""
Token Utilization Tracking for AgentNet

Provides comprehensive token usage analysis, optimization insights,
and efficiency metrics as specified in Phase 5 requirements.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import statistics
import logging

logger = logging.getLogger(__name__)


class TokenType(str, Enum):
    """Types of tokens tracked in the system."""
    INPUT = "input"
    OUTPUT = "output"
    TOTAL = "total"


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
    """Comprehensive token usage metrics for analysis."""
    
    agent_id: str
    turn_id: str
    timestamp: float
    
    # Basic token counts
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Token breakdown by category
    category_breakdown: Dict[TokenCategory, int] = field(default_factory=dict)
    
    # Context and efficiency metrics
    context_length: int = 0
    effective_tokens: int = 0  # Tokens that contributed to final output
    redundant_tokens: int = 0  # Tokens that were repeated/unnecessary
    
    # Performance metrics
    tokens_per_second: float = 0.0
    cost_per_token: float = 0.0
    
    # Quality metrics
    output_quality_score: float = 0.0  # From 0.0 to 1.0
    token_efficiency_ratio: float = 0.0  # effective_tokens / total_tokens
    
    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.input_tokens + self.output_tokens
    
    @property
    def efficiency_score(self) -> float:
        """Overall efficiency score combining multiple factors."""
        if self.total_tokens == 0:
            return 0.0
        
        # Base efficiency from token ratio
        base_efficiency = self.token_efficiency_ratio
        
        # Quality adjustment
        quality_factor = self.output_quality_score
        
        # Speed factor (normalized)
        speed_factor = min(self.tokens_per_second / 100.0, 1.0)  # Cap at 100 tokens/sec
        
        # Combined score
        return (base_efficiency * 0.5 + quality_factor * 0.3 + speed_factor * 0.2)


class TokenUtilizationTracker:
    """
    Tracks and analyzes token utilization across agent operations.
    
    Provides insights into token efficiency, optimization opportunities,
    and cost analysis for AgentNet systems.
    """
    
    def __init__(self):
        self._metrics: List[TokenMetrics] = []
        self._category_tracking: Dict[str, Dict[TokenCategory, int]] = {}
        self._cost_models: Dict[str, float] = {
            'default': 0.002,  # $0.002 per 1K tokens (rough estimate)
            'gpt-3.5-turbo': 0.002,
            'gpt-4': 0.03,
            'claude-3': 0.008
        }
    
    def record_token_usage(
        self,
        agent_id: str,
        turn_id: str,
        input_tokens: int,
        output_tokens: int,
        category_breakdown: Optional[Dict[TokenCategory, int]] = None,
        context_length: int = 0,
        processing_time: float = 0.0,
        model_name: str = "default",
        output_quality_score: float = 0.8
    ) -> TokenMetrics:
        """Record comprehensive token usage for a turn."""
        
        timestamp = time.time()
        
        # Calculate derived metrics
        total_tokens = input_tokens + output_tokens
        tokens_per_second = total_tokens / processing_time if processing_time > 0 else 0.0
        cost_per_token = self._cost_models.get(model_name, self._cost_models['default'])
        
        # Estimate token efficiency (simplified heuristic)
        effective_tokens = self._estimate_effective_tokens(output_tokens, output_quality_score)
        redundant_tokens = total_tokens - effective_tokens
        token_efficiency_ratio = effective_tokens / total_tokens if total_tokens > 0 else 0.0
        
        metrics = TokenMetrics(
            agent_id=agent_id,
            turn_id=turn_id,
            timestamp=timestamp,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            category_breakdown=category_breakdown or {},
            context_length=context_length,
            effective_tokens=effective_tokens,
            redundant_tokens=redundant_tokens,
            tokens_per_second=tokens_per_second,
            cost_per_token=cost_per_token,
            output_quality_score=output_quality_score,
            token_efficiency_ratio=token_efficiency_ratio
        )
        
        self._metrics.append(metrics)
        self._update_category_tracking(agent_id, category_breakdown or {})
        
        logger.debug(
            f"Recorded token usage for {agent_id}/{turn_id}: "
            f"{total_tokens} tokens, efficiency: {metrics.efficiency_score:.3f}"
        )
        
        return metrics
    
    def _estimate_effective_tokens(self, output_tokens: int, quality_score: float) -> int:
        """Estimate effective tokens based on output quality and heuristics."""
        # Simple heuristic: effective tokens = output_tokens * quality_score
        # In a real implementation, this would use more sophisticated analysis
        return int(output_tokens * quality_score)
    
    def _update_category_tracking(
        self, 
        agent_id: str, 
        category_breakdown: Dict[TokenCategory, int]
    ) -> None:
        """Update per-agent category tracking."""
        if agent_id not in self._category_tracking:
            self._category_tracking[agent_id] = {}
        
        for category, tokens in category_breakdown.items():
            if category not in self._category_tracking[agent_id]:
                self._category_tracking[agent_id][category] = 0
            self._category_tracking[agent_id][category] += tokens
    
    def get_agent_token_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive token usage summary for an agent."""
        agent_metrics = [m for m in self._metrics if m.agent_id == agent_id]
        
        if not agent_metrics:
            return {}
        
        # Aggregate statistics
        total_input = sum(m.input_tokens for m in agent_metrics)
        total_output = sum(m.output_tokens for m in agent_metrics)
        total_tokens = total_input + total_output
        
        # Efficiency metrics
        avg_efficiency = statistics.mean([m.efficiency_score for m in agent_metrics])
        avg_quality = statistics.mean([m.output_quality_score for m in agent_metrics])
        avg_speed = statistics.mean([m.tokens_per_second for m in agent_metrics])
        
        # Cost analysis
        total_cost = sum(m.total_tokens * m.cost_per_token / 1000 for m in agent_metrics)
        
        # Category breakdown
        category_totals = self._category_tracking.get(agent_id, {})
        
        return {
            'agent_id': agent_id,
            'total_turns': len(agent_metrics),
            'tokens': {
                'input': total_input,
                'output': total_output,
                'total': total_tokens,
                'avg_per_turn': total_tokens / len(agent_metrics)
            },
            'efficiency': {
                'avg_efficiency_score': avg_efficiency,
                'avg_quality_score': avg_quality,
                'avg_tokens_per_second': avg_speed
            },
            'cost': {
                'total_usd': total_cost,
                'avg_per_turn_usd': total_cost / len(agent_metrics)
            },
            'category_breakdown': dict(category_totals),
            'latest_metrics': agent_metrics[-1] if agent_metrics else None
        }
    
    def get_system_token_overview(self) -> Dict[str, Any]:
        """Get system-wide token utilization overview."""
        if not self._metrics:
            return {}
        
        # Aggregate across all agents
        total_input = sum(m.input_tokens for m in self._metrics)
        total_output = sum(m.output_tokens for m in self._metrics)
        total_tokens = total_input + total_output
        
        # System efficiency
        avg_efficiency = statistics.mean([m.efficiency_score for m in self._metrics])
        
        # Cost totals
        total_cost = sum(m.total_tokens * m.cost_per_token / 1000 for m in self._metrics)
        
        # Agent breakdown
        agent_ids = list(set(m.agent_id for m in self._metrics))
        agent_summaries = {
            agent_id: self.get_agent_token_summary(agent_id)
            for agent_id in agent_ids
        }
        
        # Time-based analysis (last 24h, 1h, etc.)
        now = time.time()
        recent_metrics = [m for m in self._metrics if (now - m.timestamp) < 3600]  # Last hour
        
        return {
            'overview': {
                'total_tokens': total_tokens,
                'total_turns': len(self._metrics),
                'unique_agents': len(agent_ids),
                'avg_efficiency': avg_efficiency,
                'total_cost_usd': total_cost
            },
            'recent_activity': {
                'last_hour_tokens': sum(m.total_tokens for m in recent_metrics),
                'last_hour_turns': len(recent_metrics),
                'last_hour_cost_usd': sum(
                    m.total_tokens * m.cost_per_token / 1000 for m in recent_metrics
                )
            },
            'agents': agent_summaries
        }
    
    def identify_optimization_opportunities(
        self, 
        agent_id: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Identify token optimization opportunities."""
        metrics = self._metrics
        if agent_id:
            metrics = [m for m in metrics if m.agent_id == agent_id]
        
        opportunities = {
            'high_redundancy': [],
            'low_efficiency': [],
            'expensive_turns': [],
            'context_bloat': [],
            'category_imbalance': []
        }
        
        for metric in metrics:
            # High redundancy
            redundancy_ratio = metric.redundant_tokens / metric.total_tokens if metric.total_tokens > 0 else 0
            if redundancy_ratio > 0.3:  # More than 30% redundant
                opportunities['high_redundancy'].append(
                    f"{metric.agent_id}/{metric.turn_id}: {redundancy_ratio:.1%} redundancy"
                )
            
            # Low efficiency
            if metric.efficiency_score < 0.5:
                opportunities['low_efficiency'].append(
                    f"{metric.agent_id}/{metric.turn_id}: {metric.efficiency_score:.3f} efficiency"
                )
            
            # Expensive turns
            turn_cost = metric.total_tokens * metric.cost_per_token / 1000
            if turn_cost > 0.05:  # More than $0.05 per turn
                opportunities['expensive_turns'].append(
                    f"{metric.agent_id}/{metric.turn_id}: ${turn_cost:.3f}"
                )
            
            # Context bloat
            if metric.context_length > 8000:  # Arbitrary threshold
                opportunities['context_bloat'].append(
                    f"{metric.agent_id}/{metric.turn_id}: {metric.context_length} context tokens"
                )
        
        # Category imbalance analysis (system-wide)
        if not agent_id:
            total_by_category = {}
            for agent_data in self._category_tracking.values():
                for category, tokens in agent_data.items():
                    total_by_category.setdefault(category, 0)
                    total_by_category[category] += tokens
            
            if total_by_category:
                total_tokens = sum(total_by_category.values())
                for category, tokens in total_by_category.items():
                    ratio = tokens / total_tokens
                    if ratio > 0.6:  # More than 60% of tokens in one category
                        opportunities['category_imbalance'].append(
                            f"{category.value}: {ratio:.1%} of total tokens"
                        )
        
        return opportunities
    
    def generate_optimization_recommendations(
        self, 
        agent_id: Optional[str] = None
    ) -> List[str]:
        """Generate actionable optimization recommendations."""
        opportunities = self.identify_optimization_opportunities(agent_id)
        recommendations = []
        
        if opportunities['high_redundancy']:
            recommendations.append(
                "Consider implementing response caching to reduce redundant token usage"
            )
            recommendations.append(
                "Review prompt templates to eliminate repetitive content"
            )
        
        if opportunities['low_efficiency']:
            recommendations.append(
                "Optimize prompts for more focused, relevant responses"
            )
            recommendations.append(
                "Consider fine-tuning or using more efficient models"
            )
        
        if opportunities['expensive_turns']:
            recommendations.append(
                "Implement cost monitoring alerts for high-cost operations"
            )
            recommendations.append(
                "Consider switching to more cost-effective models for routine tasks"
            )
        
        if opportunities['context_bloat']:
            recommendations.append(
                "Implement context pruning strategies to manage context size"
            )
            recommendations.append(
                "Use summarization techniques for long conversation histories"
            )
        
        if opportunities['category_imbalance']:
            recommendations.append(
                "Review token allocation across different operation types"
            )
            recommendations.append(
                "Consider optimizing the dominant token category for efficiency"
            )
        
        return recommendations
    
    def export_metrics(
        self, 
        format: str = "json",
        agent_id: Optional[str] = None
    ) -> Any:
        """Export token metrics in various formats."""
        metrics = self._metrics
        if agent_id:
            metrics = [m for m in metrics if m.agent_id == agent_id]
        
        if format == "json":
            return [
                {
                    'agent_id': m.agent_id,
                    'turn_id': m.turn_id,
                    'timestamp': m.timestamp,
                    'input_tokens': m.input_tokens,
                    'output_tokens': m.output_tokens,
                    'total_tokens': m.total_tokens,
                    'efficiency_score': m.efficiency_score,
                    'tokens_per_second': m.tokens_per_second,
                    'category_breakdown': {k.value: v for k, v in m.category_breakdown.items()}
                }
                for m in metrics
            ]
        elif format == "csv":
            # Return CSV-style data structure
            return {
                'headers': [
                    'agent_id', 'turn_id', 'timestamp', 'input_tokens', 'output_tokens',
                    'total_tokens', 'efficiency_score', 'tokens_per_second'
                ],
                'rows': [
                    [
                        m.agent_id, m.turn_id, m.timestamp, m.input_tokens, m.output_tokens,
                        m.total_tokens, m.efficiency_score, m.tokens_per_second
                    ]
                    for m in metrics
                ]
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_metrics(self, agent_id: Optional[str] = None) -> None:
        """Clear stored metrics, optionally for a specific agent."""
        if agent_id:
            self._metrics = [m for m in self._metrics if m.agent_id != agent_id]
            if agent_id in self._category_tracking:
                del self._category_tracking[agent_id]
        else:
            self._metrics.clear()
            self._category_tracking.clear()
        
        logger.info(f"Cleared token metrics{f' for {agent_id}' if agent_id else ''}")
    
    def set_cost_model(self, model_name: str, cost_per_1k_tokens: float) -> None:
        """Set cost model for a specific LLM model."""
        self._cost_models[model_name] = cost_per_1k_tokens
        logger.info(f"Set cost model for {model_name}: ${cost_per_1k_tokens:.4f}/1K tokens")