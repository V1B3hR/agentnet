"""Advanced cost analytics and predictive modeling."""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .pricing import CostRecord
from .recorder import CostRecorder, CostAggregator

logger = logging.getLogger("agentnet.cost.analytics")


@dataclass
class CostPrediction:
    """Cost prediction result."""
    
    predicted_cost: float
    confidence_interval: Tuple[float, float]
    prediction_horizon_days: int
    model_accuracy: float
    factors: Dict[str, float]  # Contributing factors
    timestamp: datetime


@dataclass
class SpendAlert:
    """Cost spend alert."""
    
    alert_type: str  # 'velocity', 'threshold', 'anomaly'
    severity: str    # 'low', 'medium', 'high', 'critical'
    message: str
    current_value: float
    threshold_value: float
    tenant_id: Optional[str]
    agent_name: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]


class CostPredictor:
    """Predictive modeling for cost forecasting."""
    
    def __init__(self, recorder: CostRecorder):
        self.recorder = recorder
        self.aggregator = CostAggregator(recorder)
    
    def predict_monthly_cost(
        self,
        tenant_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        confidence_level: float = 0.95
    ) -> CostPrediction:
        """Predict monthly cost based on historical trends."""
        
        # Get historical data (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        records = self.recorder.get_records(
            start_date=start_date,
            end_date=end_date,
            tenant_id=tenant_id,
            agent_name=agent_name
        )
        
        if not records:
            return CostPrediction(
                predicted_cost=0.0,
                confidence_interval=(0.0, 0.0),
                prediction_horizon_days=30,
                model_accuracy=0.0,
                factors={},
                timestamp=datetime.now()
            )
        
        # Calculate daily costs
        daily_costs = defaultdict(float)
        for record in records:
            day_key = record.timestamp.date()
            daily_costs[day_key] += record.total_cost
        
        costs = list(daily_costs.values())
        
        if len(costs) < 3:
            # Not enough data for prediction
            avg_daily = sum(costs) / len(costs) if costs else 0
            monthly_prediction = avg_daily * 30
            return CostPrediction(
                predicted_cost=monthly_prediction,
                confidence_interval=(monthly_prediction * 0.8, monthly_prediction * 1.2),
                prediction_horizon_days=30,
                model_accuracy=0.5,
                factors={"insufficient_data": 1.0},
                timestamp=datetime.now()
            )
        
        # Simple linear trend analysis
        avg_daily = statistics.mean(costs)
        recent_avg = statistics.mean(costs[-7:])  # Last week average
        trend_factor = recent_avg / avg_daily if avg_daily > 0 else 1.0
        
        # Calculate variance for confidence interval
        daily_std = statistics.stdev(costs) if len(costs) > 1 else avg_daily * 0.2
        
        # Predict monthly cost with trend adjustment
        predicted_daily = avg_daily * trend_factor
        monthly_prediction = predicted_daily * 30
        
        # Confidence interval based on standard deviation
        margin = daily_std * 30 * (2.0 if confidence_level >= 0.95 else 1.0)
        confidence_interval = (
            max(0, monthly_prediction - margin),
            monthly_prediction + margin
        )
        
        # Calculate model accuracy based on recent predictions vs actual
        accuracy = min(0.9, 0.5 + (len(costs) / 30) * 0.4)  # More data = higher accuracy
        
        # Analyze contributing factors
        factors = {
            "historical_average": avg_daily / monthly_prediction if monthly_prediction > 0 else 0,
            "trend_factor": trend_factor,
            "usage_consistency": 1.0 - (daily_std / avg_daily if avg_daily > 0 else 0),
            "data_points": len(costs) / 30
        }
        
        return CostPrediction(
            predicted_cost=monthly_prediction,
            confidence_interval=confidence_interval,
            prediction_horizon_days=30,
            model_accuracy=accuracy,
            factors=factors,
            timestamp=datetime.now()
        )
    
    def predict_session_cost(
        self,
        agent_name: str,
        estimated_turns: int,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo"
    ) -> CostPrediction:
        """Predict cost for a session based on historical patterns."""
        
        # Get historical data for this agent
        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)
        
        records = self.recorder.get_records(
            start_date=start_date,
            end_date=end_date,
            agent_name=agent_name
        )
        
        if not records:
            # Use provider pricing for basic estimate
            from .pricing import PricingEngine
            pricing_engine = PricingEngine()
            estimate = pricing_engine.estimate_cost(
                provider=provider,
                model=model,
                estimated_tokens=estimated_turns * 150  # Rough estimate
            )
            
            return CostPrediction(
                predicted_cost=estimate["total_cost"],
                confidence_interval=(estimate["total_cost"] * 0.5, estimate["total_cost"] * 2.0),
                prediction_horizon_days=0,
                model_accuracy=0.3,
                factors={"no_history": 1.0},
                timestamp=datetime.now()
            )
        
        # Calculate average cost per turn for this agent
        total_cost = sum(r.total_cost for r in records)
        total_records = len(records)
        avg_cost_per_turn = total_cost / total_records if total_records > 0 else 0
        
        predicted_cost = avg_cost_per_turn * estimated_turns
        
        # Calculate confidence based on data variance
        costs = [r.total_cost for r in records]
        cost_std = statistics.stdev(costs) if len(costs) > 1 else avg_cost_per_turn * 0.3
        
        margin = cost_std * (estimated_turns ** 0.5)  # Margin grows with session length
        confidence_interval = (
            max(0, predicted_cost - margin),
            predicted_cost + margin
        )
        
        accuracy = min(0.8, 0.4 + (len(records) / 100) * 0.4)
        
        factors = {
            "agent_history": len(records) / 100,
            "cost_consistency": 1.0 - (cost_std / avg_cost_per_turn if avg_cost_per_turn > 0 else 0),
            "session_length_factor": min(1.0, estimated_turns / 10)
        }
        
        return CostPrediction(
            predicted_cost=predicted_cost,
            confidence_interval=confidence_interval,
            prediction_horizon_days=0,
            model_accuracy=accuracy,
            factors=factors,
            timestamp=datetime.now()
        )


class SpendAlertEngine:
    """Engine for detecting spend anomalies and generating alerts."""
    
    def __init__(self, recorder: CostRecorder):
        self.recorder = recorder
        self.aggregator = CostAggregator(recorder)
        self.alert_thresholds = {
            "velocity_multiplier": 3.0,  # Alert if spending rate > 3x normal
            "daily_spike_threshold": 5.0,  # Alert if daily spend > $5
            "hourly_spike_threshold": 1.0,  # Alert if hourly spend > $1
        }
    
    def check_spend_velocity(
        self,
        tenant_id: Optional[str] = None,
        lookback_hours: int = 24
    ) -> List[SpendAlert]:
        """Check for unusual spend velocity."""
        alerts = []
        
        end_date = datetime.now()
        current_period_start = end_date - timedelta(hours=lookback_hours)
        baseline_period_start = current_period_start - timedelta(hours=lookback_hours)
        
        # Get current period records
        current_records = self.recorder.get_records(
            start_date=current_period_start,
            end_date=end_date,
            tenant_id=tenant_id
        )
        
        # Get baseline period records
        baseline_records = self.recorder.get_records(
            start_date=baseline_period_start,
            end_date=current_period_start,
            tenant_id=tenant_id
        )
        
        current_spend = sum(r.total_cost for r in current_records)
        baseline_spend = sum(r.total_cost for r in baseline_records)
        
        if baseline_spend > 0:
            velocity_ratio = current_spend / baseline_spend
            
            if velocity_ratio >= self.alert_thresholds["velocity_multiplier"]:
                severity = "critical" if velocity_ratio >= 5.0 else "high"
                
                alerts.append(SpendAlert(
                    alert_type="velocity",
                    severity=severity,
                    message=f"Spend velocity {velocity_ratio:.1f}x higher than baseline",
                    current_value=current_spend,
                    threshold_value=baseline_spend * self.alert_thresholds["velocity_multiplier"],
                    tenant_id=tenant_id,
                    agent_name=None,
                    timestamp=datetime.now(),
                    metadata={
                        "velocity_ratio": velocity_ratio,
                        "baseline_spend": baseline_spend,
                        "lookback_hours": lookback_hours
                    }
                ))
        
        return alerts
    
    def check_daily_spend_spikes(
        self,
        tenant_id: Optional[str] = None,
        days_to_check: int = 7
    ) -> List[SpendAlert]:
        """Check for daily spend spikes."""
        alerts = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_to_check)
        
        records = self.recorder.get_records(
            start_date=start_date,
            end_date=end_date,
            tenant_id=tenant_id
        )
        
        # Group by day
        daily_costs = defaultdict(float)
        for record in records:
            day_key = record.timestamp.date()
            daily_costs[day_key] += record.total_cost
        
        # Check each day against threshold
        for day, cost in daily_costs.items():
            if cost >= self.alert_thresholds["daily_spike_threshold"]:
                severity = "critical" if cost >= 20.0 else "high" if cost >= 10.0 else "medium"
                
                alerts.append(SpendAlert(
                    alert_type="threshold",
                    severity=severity,
                    message=f"Daily spend spike: ${cost:.2f} on {day}",
                    current_value=cost,
                    threshold_value=self.alert_thresholds["daily_spike_threshold"],
                    tenant_id=tenant_id,
                    agent_name=None,
                    timestamp=datetime.now(),
                    metadata={
                        "spike_date": day.isoformat(),
                        "threshold_exceeded_by": cost - self.alert_thresholds["daily_spike_threshold"]
                    }
                ))
        
        return alerts
    
    def check_agent_anomalies(
        self,
        tenant_id: Optional[str] = None,
        lookback_hours: int = 24
    ) -> List[SpendAlert]:
        """Check for agent-specific spend anomalies."""
        alerts = []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=lookback_hours)
        
        records = self.recorder.get_records(
            start_date=start_date,
            end_date=end_date,
            tenant_id=tenant_id
        )
        
        # Group by agent
        agent_costs = defaultdict(float)
        agent_counts = defaultdict(int)
        
        for record in records:
            agent_costs[record.agent_name] += record.total_cost
            agent_counts[record.agent_name] += 1
        
        # Check for agents with unusually high costs
        if agent_costs:
            cost_values = list(agent_costs.values())
            if len(cost_values) > 1:
                avg_cost = statistics.mean(cost_values)
                cost_std = statistics.stdev(cost_values)
                
                for agent_name, cost in agent_costs.items():
                    # Check if cost is more than 2 standard deviations above mean
                    if cost > avg_cost + 2 * cost_std and cost > 0.5:
                        severity = "high" if cost > avg_cost + 3 * cost_std else "medium"
                        
                        alerts.append(SpendAlert(
                            alert_type="anomaly",
                            severity=severity,
                            message=f"Agent {agent_name} showing unusual spend pattern",
                            current_value=cost,
                            threshold_value=avg_cost + 2 * cost_std,
                            tenant_id=tenant_id,
                            agent_name=agent_name,
                            timestamp=datetime.now(),
                            metadata={
                                "inference_count": agent_counts[agent_name],
                                "cost_per_inference": cost / agent_counts[agent_name],
                                "deviation_from_mean": (cost - avg_cost) / cost_std if cost_std > 0 else 0
                            }
                        ))
        
        return alerts
    
    def get_all_alerts(
        self,
        tenant_id: Optional[str] = None,
        lookback_hours: int = 24
    ) -> Dict[str, List[SpendAlert]]:
        """Get all types of spend alerts."""
        return {
            "velocity_alerts": self.check_spend_velocity(tenant_id, lookback_hours),
            "spike_alerts": self.check_daily_spend_spikes(tenant_id, lookback_hours // 24 or 1),
            "anomaly_alerts": self.check_agent_anomalies(tenant_id, lookback_hours)
        }


class CostReportGenerator:
    """Generate comprehensive cost reports."""
    
    def __init__(self, recorder: CostRecorder):
        self.recorder = recorder
        self.aggregator = CostAggregator(recorder)
        self.predictor = CostPredictor(recorder)
        self.alert_engine = SpendAlertEngine(recorder)
    
    def generate_executive_summary(
        self,
        tenant_id: Optional[str] = None,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """Generate executive cost summary report."""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Get basic summary
        summary = self.aggregator.get_cost_summary(
            start_date=start_date,
            end_date=end_date,
            tenant_id=tenant_id
        )
        
        # Get trends
        trends = self.aggregator.get_cost_trends(days=period_days, tenant_id=tenant_id)
        
        # Get prediction
        prediction = self.predictor.predict_monthly_cost(tenant_id=tenant_id)
        
        # Get alerts
        alerts = self.alert_engine.get_all_alerts(tenant_id=tenant_id)
        alert_count = sum(len(alert_list) for alert_list in alerts.values())
        
        # Calculate period-over-period comparison
        prev_start = start_date - timedelta(days=period_days)
        prev_summary = self.aggregator.get_cost_summary(
            start_date=prev_start,
            end_date=start_date,
            tenant_id=tenant_id
        )
        
        cost_change = 0.0
        if prev_summary["total_cost"] > 0:
            cost_change = (summary["total_cost"] - prev_summary["total_cost"]) / prev_summary["total_cost"]
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": period_days
            },
            "cost_summary": {
                "total_cost": summary["total_cost"],
                "total_tokens": summary["total_tokens_input"] + summary["total_tokens_output"],
                "average_daily_cost": summary["total_cost"] / period_days,
                "cost_change_percentage": cost_change * 100,
                "inference_count": summary["record_count"]
            },
            "predictions": {
                "monthly_forecast": prediction.predicted_cost,
                "confidence_range": prediction.confidence_interval,
                "model_accuracy": prediction.model_accuracy
            },
            "alerts": {
                "total_alerts": alert_count,
                "critical_alerts": sum(1 for alert_list in alerts.values() 
                                     for alert in alert_list if alert.severity == "critical"),
                "summary_by_type": {k: len(v) for k, v in alerts.items()}
            },
            "top_providers": [
                {"provider": provider, "cost": data["cost"], "percentage": data["cost"] / summary["total_cost"] * 100}
                for provider, data in summary["provider_breakdown"].items()
            ][:5],
            "top_agents": [
                {"agent": agent, "cost": data["cost"], "inference_count": summary["record_count"]}
                for agent, data in summary["agent_breakdown"].items()
            ][:10],
            "report_generated": datetime.now().isoformat()
        }