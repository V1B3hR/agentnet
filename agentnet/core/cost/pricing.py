"""Cost pricing engine for different providers."""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger("agentnet.cost")


class ProviderType(Enum):
    """Supported provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    EXAMPLE = "example"
    LOCAL = "local"


@dataclass
class CostRecord:
    """Record of a cost event."""
    provider: str
    model: str
    tokens_input: int
    tokens_output: int
    cost_input: float
    cost_output: float
    total_cost: float
    timestamp: datetime
    agent_name: str
    task_id: str
    session_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderPricing:
    """Pricing configuration for a provider."""
    provider_type: ProviderType
    model_pricing: Dict[str, Dict[str, float]]  # model -> {input_per_1k_tokens, output_per_1k_tokens}
    default_model: str
    currency: str = "USD"


class PricingEngine:
    """Engine for calculating costs across different providers."""
    
    def __init__(self):
        self.provider_configs: Dict[ProviderType, ProviderPricing] = {}
        self._setup_default_pricing()
    
    def _setup_default_pricing(self):
        """Setup default pricing for known providers."""
        # OpenAI pricing (as of 2024)
        openai_pricing = ProviderPricing(
            provider_type=ProviderType.OPENAI,
            model_pricing={
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
                "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            },
            default_model="gpt-3.5-turbo"
        )
        
        # Anthropic pricing (as of 2024)
        anthropic_pricing = ProviderPricing(
            provider_type=ProviderType.ANTHROPIC,
            model_pricing={
                "claude-3-opus": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            },
            default_model="claude-3-haiku"
        )
        
        # Example provider (for testing)
        example_pricing = ProviderPricing(
            provider_type=ProviderType.EXAMPLE,
            model_pricing={
                "example-model": {"input": 0.001, "output": 0.001},
            },
            default_model="example-model"
        )
        
        # Local provider (free)
        local_pricing = ProviderPricing(
            provider_type=ProviderType.LOCAL,
            model_pricing={
                "local-model": {"input": 0.0, "output": 0.0},
            },
            default_model="local-model"
        )
        
        self.provider_configs[ProviderType.OPENAI] = openai_pricing
        self.provider_configs[ProviderType.ANTHROPIC] = anthropic_pricing
        self.provider_configs[ProviderType.EXAMPLE] = example_pricing
        self.provider_configs[ProviderType.LOCAL] = local_pricing
    
    def add_provider_config(self, config: ProviderPricing):
        """Add or update pricing configuration for a provider."""
        self.provider_configs[config.provider_type] = config
        logger.info(f"Updated pricing config for {config.provider_type.value}")
    
    def calculate_cost(
        self,
        provider: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        agent_name: str,
        task_id: str,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CostRecord:
        """Calculate cost for a provider operation."""
        
        # Find provider config
        provider_type = None
        for ptype in ProviderType:
            if ptype.value == provider.lower():
                provider_type = ptype
                break
        
        if provider_type is None or provider_type not in self.provider_configs:
            logger.warning(f"Unknown provider {provider}, using zero cost")
            return CostRecord(
                provider=provider,
                model=model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_input=0.0,
                cost_output=0.0,
                total_cost=0.0,
                timestamp=datetime.now(),
                agent_name=agent_name,
                task_id=task_id,
                session_id=session_id,
                tenant_id=tenant_id,
                metadata=metadata or {}
            )
        
        config = self.provider_configs[provider_type]
        
        # Get model pricing or use default
        if model in config.model_pricing:
            pricing = config.model_pricing[model]
        elif config.default_model in config.model_pricing:
            pricing = config.model_pricing[config.default_model]
            logger.warning(f"Model {model} not found, using default {config.default_model} pricing")
        else:
            logger.error(f"No pricing available for provider {provider}")
            pricing = {"input": 0.0, "output": 0.0}
        
        # Calculate costs (pricing is per 1K tokens)
        cost_input = (tokens_input / 1000.0) * pricing["input"]
        cost_output = (tokens_output / 1000.0) * pricing["output"]
        total_cost = cost_input + cost_output
        
        return CostRecord(
            provider=provider,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_input=cost_input,
            cost_output=cost_output,
            total_cost=total_cost,
            timestamp=datetime.now(),
            agent_name=agent_name,
            task_id=task_id,
            session_id=session_id,
            tenant_id=tenant_id,
            metadata=metadata or {}
        )
    
    def get_provider_models(self, provider: str) -> Dict[str, Dict[str, float]]:
        """Get available models and pricing for a provider."""
        provider_type = None
        for ptype in ProviderType:
            if ptype.value == provider.lower():
                provider_type = ptype
                break
        
        if provider_type and provider_type in self.provider_configs:
            return self.provider_configs[provider_type].model_pricing
        
        return {}
    
    def estimate_cost(
        self,
        provider: str,
        model: str,
        estimated_tokens: int,
        input_output_ratio: float = 0.7
    ) -> Dict[str, float]:
        """Estimate cost for a given token count."""
        input_tokens = int(estimated_tokens * input_output_ratio)
        output_tokens = estimated_tokens - input_tokens
        
        record = self.calculate_cost(
            provider=provider,
            model=model,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            agent_name="estimate",
            task_id="estimate"
        )
        
        return {
            "input_cost": record.cost_input,
            "output_cost": record.cost_output,
            "total_cost": record.total_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }