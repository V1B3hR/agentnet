"""Enhanced cost tracking with predictive modeling and advanced reporting."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .recorder import CostRecorder

logger = logging.getLogger("agentnet.cost.analytics")


@dataclass
class CostPrediction:
    """Prediction result for future costs."""
    
    predicted_cost: float
    confidence_interval: Tuple[float, float]
    prediction_horizon_days: int
    model_accuracy: float
    trend: str  # "increasing", "decreasing", "stable"
    factors: List[str]  # Contributing factors
    timestamp: datetime


@dataclass
class CostOptimizationRecommendation:
    """Recommendation for cost optimization."""
    
    category: str  # "model_selection", "usage_pattern", "budget_allocation"
    description: str
    estimated_savings: float
    estimated_savings_percentage: float
    implementation_effort: str  # "low", "medium", "high"
    priority: str  # "high", "medium", "low"
    actions: List[str]


class CostPredictor:
    """Predictive modeling for cost estimation and forecasting."""
    
    def __init__(self, recorder: CostRecorder):
        self.recorder = recorder
        self.prediction_cache = {}
        self.cache_ttl_minutes = 30
    
    def predict_monthly_cost(
        self,
        tenant_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        confidence_level: float = 0.95,
    ) -> CostPrediction:
        """Predict monthly cost based on historical data."""
        
        # Get historical data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        records = self.recorder.get_records(
            start_date=start_date,
            end_date=end_date,
            tenant_id=tenant_id,
            agent_name=agent_name,
        )
        
        if len(records) < 7:  # Need at least a week of data
            return CostPrediction(
                predicted_cost=0.0,
                confidence_interval=(0.0, 0.0),
                prediction_horizon_days=30,
                model_accuracy=0.0,
                trend="insufficient_data",
                factors=["Insufficient historical data"],
                timestamp=datetime.now(),
            )
        
        # Calculate daily costs
        daily_costs = self._group_by_day(records)
        
        # Simple linear trend analysis
        days = list(range(len(daily_costs)))
        costs = [daily_costs[day] for day in sorted(daily_costs.keys())]
        
        # Calculate trend using numpy
        if len(costs) >= 2:
            # Linear regression for trend
            coefficients = np.polyfit(days, costs, 1)
            trend_slope = coefficients[0]
            
            # Predict next 30 days
            current_daily_average = np.mean(costs[-7:])  # Last week average
            projected_monthly_cost = current_daily_average * 30
            
            # Apply trend adjustment
            trend_adjustment = trend_slope * 15  # Mid-month adjustment
            predicted_cost = projected_monthly_cost + (trend_adjustment * 30)
            
            # Calculate confidence interval (simplified)
            std_dev = np.std(costs)
            z_score = 1.96 if confidence_level >= 0.95 else 1.64
            margin = z_score * std_dev * np.sqrt(30)  # For 30 days
            
            confidence_interval = (
                max(0, predicted_cost - margin),
                predicted_cost + margin,
            )
            
            # Determine trend direction
            if trend_slope > current_daily_average * 0.1:
                trend = "increasing"
            elif trend_slope < -current_daily_average * 0.1:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Model accuracy (simplified R-squared approximation)
            if len(costs) > 2:
                predicted_values = [coefficients[1] + coefficients[0] * x for x in days]
                ss_res = sum((actual - pred) ** 2 for actual, pred in zip(costs, predicted_values))
                ss_tot = sum((actual - np.mean(costs)) ** 2 for actual in costs)
                accuracy = max(0, 1 - (ss_res / ss_tot)) if ss_tot > 0 else 0
            else:
                accuracy = 0.5
            
        else:
            predicted_cost = costs[0] * 30
            confidence_interval = (predicted_cost * 0.5, predicted_cost * 1.5)
            trend = "insufficient_data"
            accuracy = 0.3
        
        # Identify contributing factors
        factors = self._identify_cost_factors(records)
        
        return CostPrediction(
            predicted_cost=max(0, predicted_cost),
            confidence_interval=confidence_interval,
            prediction_horizon_days=30,
            model_accuracy=accuracy,
            trend=trend,
            factors=factors,
            timestamp=datetime.now(),
        )
    
    def predict_session_cost(
        self,
        estimated_interactions: int,
        agent_name: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        provider: str = "openai",
    ) -> Dict[str, float]:
        """Predict cost for a future session."""
        
        # Get historical data for the agent
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        records = self.recorder.get_records(
            start_date=start_date,
            end_date=end_date,
            agent_name=agent_name,
        )
        
        if records:
            # Calculate average cost per interaction
            avg_cost_per_interaction = sum(r.total_cost for r in records) / len(records)
            avg_tokens_per_interaction = sum(r.tokens_input + r.tokens_output for r in records) / len(records)
        else:
            # Use default estimates based on model
            from .pricing import PricingEngine
            pricing_engine = PricingEngine()
            
            estimate = pricing_engine.estimate_cost(
                provider=provider,
                model=model_name,
                estimated_tokens=500,  # Default estimate
            )
            avg_cost_per_interaction = estimate["total_cost"]
            avg_tokens_per_interaction = 500
        
        # Predict session cost
        predicted_cost = avg_cost_per_interaction * estimated_interactions
        predicted_tokens = avg_tokens_per_interaction * estimated_interactions
        
        # Add confidence bounds
        uncertainty_factor = 0.3 if records else 0.5
        lower_bound = predicted_cost * (1 - uncertainty_factor)
        upper_bound = predicted_cost * (1 + uncertainty_factor)
        
        return {
            "predicted_cost": predicted_cost,
            "predicted_tokens": predicted_tokens,
            "cost_range": (lower_bound, upper_bound),
            "avg_cost_per_interaction": avg_cost_per_interaction,
            "confidence": "high" if records and len(records) > 10 else "medium" if records else "low",
        }
    
    def _group_by_day(self, records: List) -> Dict[str, float]:
        """Group cost records by day."""
        daily_costs = defaultdict(float)
        
        for record in records:
            day_key = record.timestamp.strftime("%Y-%m-%d")
            daily_costs[day_key] += record.total_cost
        
        return dict(daily_costs)
    
    def _identify_cost_factors(self, records: List) -> List[str]:
        """Identify factors contributing to cost patterns."""
        factors = []
        
        if not records:
            return ["No historical data"]
        
        # Analyze provider distribution
        provider_costs = defaultdict(float)
        model_costs = defaultdict(float)
        agent_costs = defaultdict(float)
        
        for record in records:
            provider_costs[record.provider] += record.total_cost
            model_costs[record.model] += record.total_cost
            agent_costs[record.agent_name] += record.total_cost
        
        # Top contributing factors
        total_cost = sum(provider_costs.values())
        
        if total_cost > 0:
            # Top provider
            top_provider = max(provider_costs.items(), key=lambda x: x[1])
            if top_provider[1] / total_cost > 0.5:
                factors.append(f"Primary provider: {top_provider[0]} ({top_provider[1]/total_cost:.1%})")
            
            # Expensive models
            expensive_models = [(model, cost) for model, cost in model_costs.items() 
                              if cost / total_cost > 0.3]
            for model, cost in expensive_models:
                factors.append(f"High-cost model: {model} ({cost/total_cost:.1%})")
            
            # High-usage agents
            top_agents = sorted(agent_costs.items(), key=lambda x: x[1], reverse=True)[:2]
            for agent, cost in top_agents:
                if cost / total_cost > 0.2:
                    factors.append(f"High-usage agent: {agent} ({cost/total_cost:.1%})")
        
        return factors if factors else ["Standard usage pattern"]


class CostOptimizer:
    """Generates cost optimization recommendations."""
    
    def __init__(self, recorder: CostRecorder, predictor: CostPredictor):
        self.recorder = recorder
        self.predictor = predictor
    
    def generate_recommendations(
        self,
        tenant_id: Optional[str] = None,
        analysis_days: int = 30,
    ) -> List[CostOptimizationRecommendation]:
        """Generate cost optimization recommendations."""
        
        recommendations = []
        
        # Get recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=analysis_days)
        
        records = self.recorder.get_records(
            start_date=start_date,
            end_date=end_date,
            tenant_id=tenant_id,
        )
        
        if not records:
            return recommendations
        
        # Analyze model usage patterns
        recommendations.extend(self._analyze_model_usage(records))
        
        # Analyze usage timing patterns
        recommendations.extend(self._analyze_usage_patterns(records))
        
        # Analyze agent efficiency
        recommendations.extend(self._analyze_agent_efficiency(records))
        
        # Budget allocation recommendations
        recommendations.extend(self._analyze_budget_allocation(records))
        
        return sorted(recommendations, key=lambda x: (
            {"high": 0, "medium": 1, "low": 2}[x.priority],
            -x.estimated_savings
        ))
    
    def _analyze_model_usage(self, records: List) -> List[CostOptimizationRecommendation]:
        """Analyze model usage for optimization opportunities."""
        recommendations = []
        
        model_stats = defaultdict(lambda: {
            "total_cost": 0.0,
            "usage_count": 0,
            "avg_tokens": 0,
            "total_tokens": 0,
        })
        
        for record in records:
            stats = model_stats[record.model]
            stats["total_cost"] += record.total_cost
            stats["usage_count"] += 1
            stats["total_tokens"] += record.tokens_input + record.tokens_output
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats["usage_count"] > 0:
                stats["avg_tokens"] = stats["total_tokens"] / stats["usage_count"]
                stats["avg_cost"] = stats["total_cost"] / stats["usage_count"]
        
        total_cost = sum(record.total_cost for record in records)
        
        # Check for expensive model overuse
        for model, stats in model_stats.items():
            if stats["total_cost"] / total_cost > 0.5 and "gpt-4" in model.lower():
                # Recommend switching some usage to cheaper models
                estimated_savings = stats["total_cost"] * 0.3  # Assume 30% savings
                
                recommendations.append(CostOptimizationRecommendation(
                    category="model_selection",
                    description=f"Consider using GPT-3.5-turbo for simpler tasks instead of {model}",
                    estimated_savings=estimated_savings,
                    estimated_savings_percentage=30.0,
                    implementation_effort="low",
                    priority="high",
                    actions=[
                        f"Identify tasks using {model} that could use cheaper alternatives",
                        "Implement model selection logic based on task complexity",
                        "Monitor performance impact of model changes",
                    ],
                ))
        
        return recommendations
    
    def _analyze_usage_patterns(self, records: List) -> List[CostOptimizationRecommendation]:
        """Analyze usage timing patterns."""
        recommendations = []
        
        # Analyze peak usage times
        hourly_usage = defaultdict(lambda: {"cost": 0.0, "count": 0})
        
        for record in records:
            hour = record.timestamp.hour
            hourly_usage[hour]["cost"] += record.total_cost
            hourly_usage[hour]["count"] += 1
        
        # Find peak hours
        if hourly_usage:
            avg_hourly_cost = sum(data["cost"] for data in hourly_usage.values()) / len(hourly_usage)
            peak_hours = [hour for hour, data in hourly_usage.items() 
                         if data["cost"] > avg_hourly_cost * 1.5]
            
            if peak_hours and len(peak_hours) < 8:  # If usage is concentrated
                recommendations.append(CostOptimizationRecommendation(
                    category="usage_pattern",
                    description="Consider distributing workload to reduce peak usage concentration",
                    estimated_savings=avg_hourly_cost * len(peak_hours) * 0.1,
                    estimated_savings_percentage=10.0,
                    implementation_effort="medium",
                    priority="medium",
                    actions=[
                        "Identify non-urgent tasks that can be shifted to off-peak hours",
                        "Implement queuing system for batch processing",
                        "Consider caching for repeated queries",
                    ],
                ))
        
        return recommendations
    
    def _analyze_agent_efficiency(self, records: List) -> List[CostOptimizationRecommendation]:
        """Analyze agent cost efficiency."""
        recommendations = []
        
        agent_stats = defaultdict(lambda: {
            "total_cost": 0.0,
            "usage_count": 0,
            "avg_cost": 0.0,
        })
        
        for record in records:
            stats = agent_stats[record.agent_name]
            stats["total_cost"] += record.total_cost
            stats["usage_count"] += 1
        
        # Calculate averages
        for agent, stats in agent_stats.items():
            if stats["usage_count"] > 0:
                stats["avg_cost"] = stats["total_cost"] / stats["usage_count"]
        
        if len(agent_stats) > 1:
            # Find agents with significantly higher average costs
            avg_costs = [stats["avg_cost"] for stats in agent_stats.values()]
            overall_avg = sum(avg_costs) / len(avg_costs)
            
            expensive_agents = [
                (agent, stats) for agent, stats in agent_stats.items()
                if stats["avg_cost"] > overall_avg * 1.5 and stats["usage_count"] > 5
            ]
            
            for agent, stats in expensive_agents:
                recommendations.append(CostOptimizationRecommendation(
                    category="agent_efficiency",
                    description=f"Agent '{agent}' has higher than average cost per interaction",
                    estimated_savings=stats["total_cost"] * 0.2,
                    estimated_savings_percentage=20.0,
                    implementation_effort="medium",
                    priority="medium",
                    actions=[
                        f"Review {agent} configuration and prompt efficiency",
                        "Consider adjusting agent style parameters",
                        "Analyze task complexity assigned to this agent",
                    ],
                ))
        
        return recommendations
    
    def _analyze_budget_allocation(self, records: List) -> List[CostOptimizationRecommendation]:
        """Analyze budget allocation opportunities."""
        recommendations = []
        
        # Simple budget allocation recommendation
        total_cost = sum(record.total_cost for record in records)
        
        if total_cost > 50.0:  # If significant cost
            recommendations.append(CostOptimizationRecommendation(
                category="budget_allocation",
                description="Consider implementing cost alerts and budget limits",
                estimated_savings=total_cost * 0.1,
                estimated_savings_percentage=10.0,
                implementation_effort="low",
                priority="high",
                actions=[
                    "Set up cost monitoring dashboards",
                    "Implement budget alerts at 75% and 90% thresholds",
                    "Review and approve high-cost operations",
                ],
            ))
        
        return recommendations


class CostReporter:
    """Advanced cost reporting and analytics."""
    
    def __init__(self, recorder: CostRecorder, predictor: CostPredictor, optimizer: CostOptimizer):
        self.recorder = recorder
        self.predictor = predictor
        self.optimizer = optimizer
    
    def generate_comprehensive_report(
        self,
        tenant_id: Optional[str] = None,
        report_days: int = 30,
    ) -> Dict[str, Any]:
        """Generate a comprehensive cost report."""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=report_days)
        
        # Get cost data
        from .recorder import CostAggregator
        aggregator = CostAggregator(self.recorder)
        
        summary = aggregator.get_cost_summary(
            start_date=start_date,
            end_date=end_date,
            tenant_id=tenant_id,
        )
        
        trends = aggregator.get_cost_trends(days=report_days, tenant_id=tenant_id)
        top_agents = aggregator.get_top_cost_agents(limit=10, tenant_id=tenant_id)
        
        # Get predictions
        prediction = self.predictor.predict_monthly_cost(tenant_id=tenant_id)
        
        # Get recommendations
        recommendations = self.optimizer.generate_recommendations(
            tenant_id=tenant_id,
            analysis_days=report_days,
        )
        
        # Compile report
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "period_days": report_days,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "tenant_id": tenant_id,
            },
            "cost_summary": summary,
            "cost_trends": trends,
            "top_agents": top_agents,
            "predictions": {
                "monthly_forecast": asdict(prediction),
            },
            "optimization_recommendations": [asdict(rec) for rec in recommendations],
            "key_insights": self._generate_key_insights(summary, trends, prediction, recommendations),
        }
        
        return report
    
    def _generate_key_insights(
        self,
        summary: Dict[str, Any],
        trends: Dict[str, Any],
        prediction: CostPrediction,
        recommendations: List[CostOptimizationRecommendation],
    ) -> List[str]:
        """Generate key insights from the data."""
        
        insights = []
        
        # Cost insights
        if summary["total_cost"] > 0:
            insights.append(f"Total cost for period: ${summary['total_cost']:.2f}")
            
            # Provider analysis
            if summary["provider_breakdown"]:
                top_provider = max(summary["provider_breakdown"].items(), key=lambda x: x[1]["cost"])
                provider_percentage = (top_provider[1]["cost"] / summary["total_cost"]) * 100
                insights.append(f"Primary provider: {top_provider[0]} ({provider_percentage:.1f}% of costs)")
        
        # Prediction insights
        if prediction.trend != "insufficient_data":
            insights.append(f"Cost trend: {prediction.trend}")
            insights.append(f"Predicted monthly cost: ${prediction.predicted_cost:.2f} (±${prediction.confidence_interval[1] - prediction.predicted_cost:.2f})")
        
        # Recommendation insights
        if recommendations:
            total_potential_savings = sum(rec.estimated_savings for rec in recommendations)
            high_priority_count = sum(1 for rec in recommendations if rec.priority == "high")
            insights.append(f"Potential savings identified: ${total_potential_savings:.2f}")
            if high_priority_count > 0:
                insights.append(f"{high_priority_count} high-priority optimization opportunities")
        
        return insights