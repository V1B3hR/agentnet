"""Predictive cost modeling and reporting for AgentNet."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .pricing import CostRecord
from .recorder import CostRecorder

logger = logging.getLogger("agentnet.cost.predictions")


class CostPredictor:
    """Predictive cost modeling and trend analysis."""
    
    def __init__(self, recorder: CostRecorder):
        self.recorder = recorder
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._last_cache_update = datetime.min
    
    def predict_monthly_cost(
        self, 
        tenant_id: Optional[str] = None,
        days_history: int = 30
    ) -> Dict[str, Union[float, str]]:
        """Predict monthly cost based on historical trends."""
        try:
            # Get historical cost data
            historical_data = self._get_historical_costs(tenant_id, days_history)
            
            if len(historical_data) < 3:
                return {
                    "predicted_monthly_cost": 0.0,
                    "confidence": "low",
                    "trend": "insufficient_data",
                    "message": "Need more historical data for accurate prediction"
                }
            
            # Calculate daily averages
            daily_costs = self._aggregate_daily_costs(historical_data)
            
            if not daily_costs:
                return {
                    "predicted_monthly_cost": 0.0,
                    "confidence": "low", 
                    "trend": "no_data"
                }
            
            # Simple linear regression for trend analysis
            x_values = list(range(len(daily_costs)))
            y_values = list(daily_costs.values())
            
            # Calculate trend
            slope, intercept = self._linear_regression(x_values, y_values)
            
            # Predict next 30 days
            current_day = len(daily_costs)
            predicted_costs = []
            
            for i in range(30):
                day_prediction = max(0, slope * (current_day + i) + intercept)
                predicted_costs.append(day_prediction)
            
            monthly_prediction = sum(predicted_costs)
            
            # Determine confidence based on data variance
            variance = np.var(y_values) if len(y_values) > 1 else 0
            avg_cost = np.mean(y_values)
            cv = math.sqrt(variance) / avg_cost if avg_cost > 0 else float('inf')
            
            confidence = "high" if cv < 0.3 else "medium" if cv < 0.7 else "low"
            trend = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
            
            return {
                "predicted_monthly_cost": round(monthly_prediction, 4),
                "current_daily_avg": round(avg_cost, 4),
                "trend": trend,
                "trend_slope": round(slope, 6),
                "confidence": confidence,
                "data_points": len(daily_costs),
                "variance_coefficient": round(cv, 3) if cv != float('inf') else None
            }
            
        except Exception as e:
            logger.error(f"Error predicting monthly cost: {e}")
            return {
                "predicted_monthly_cost": 0.0,
                "confidence": "low",
                "trend": "error",
                "message": f"Prediction error: {str(e)}"
            }
    
    def analyze_cost_patterns(
        self, 
        tenant_id: Optional[str] = None,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Analyze cost patterns and identify anomalies."""
        try:
            historical_data = self._get_historical_costs(tenant_id, days_back)
            
            if not historical_data:
                return {"patterns": [], "anomalies": [], "message": "No data available"}
            
            # Group by provider, model, and agent
            patterns = {
                "by_provider": defaultdict(float),
                "by_model": defaultdict(float),
                "by_agent": defaultdict(float),
                "by_hour": defaultdict(float)
            }
            
            for record in historical_data:
                patterns["by_provider"][record.provider] += record.total_cost
                patterns["by_model"][record.model] += record.total_cost
                patterns["by_agent"][record.agent_name] += record.total_cost
                patterns["by_hour"][record.timestamp.hour] += record.total_cost
            
            # Detect anomalies (simple threshold-based)
            anomalies = []
            daily_costs = self._aggregate_daily_costs(historical_data)
            
            if len(daily_costs) > 1:
                values = list(daily_costs.values())
                mean_cost = np.mean(values)
                std_cost = np.std(values)
                threshold = mean_cost + 2 * std_cost
                
                for date, cost in daily_costs.items():
                    if cost > threshold:
                        anomalies.append({
                            "date": date,
                            "cost": round(cost, 4),
                            "threshold": round(threshold, 4),
                            "type": "cost_spike"
                        })
            
            return {
                "patterns": {
                    "top_providers": sorted(
                        [(k, round(v, 4)) for k, v in patterns["by_provider"].items()],
                        key=lambda x: x[1], reverse=True
                    )[:5],
                    "top_models": sorted(
                        [(k, round(v, 4)) for k, v in patterns["by_model"].items()],
                        key=lambda x: x[1], reverse=True
                    )[:5],
                    "top_agents": sorted(
                        [(k, round(v, 4)) for k, v in patterns["by_agent"].items()],
                        key=lambda x: x[1], reverse=True
                    )[:5],
                    "peak_hours": sorted(
                        [(k, round(v, 4)) for k, v in patterns["by_hour"].items()],
                        key=lambda x: x[1], reverse=True
                    )[:3]
                },
                "anomalies": anomalies,
                "summary": {
                    "total_records": len(historical_data),
                    "date_range": f"{min(r.timestamp.date() for r in historical_data)} to {max(r.timestamp.date() for r in historical_data)}",
                    "total_cost": round(sum(r.total_cost for r in historical_data), 4)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cost patterns: {e}")
            return {"patterns": [], "anomalies": [], "error": str(e)}
    
    def generate_cost_report(
        self, 
        tenant_id: Optional[str] = None,
        report_type: str = "summary"
    ) -> Dict[str, Any]:
        """Generate comprehensive cost reports."""
        try:
            if report_type == "summary":
                return self._generate_summary_report(tenant_id)
            elif report_type == "detailed":
                return self._generate_detailed_report(tenant_id)
            elif report_type == "predictive":
                return self._generate_predictive_report(tenant_id)
            else:
                raise ValueError(f"Unknown report type: {report_type}")
                
        except Exception as e:
            logger.error(f"Error generating cost report: {e}")
            return {"error": str(e), "report_type": report_type}
    
    def _get_historical_costs(
        self, 
        tenant_id: Optional[str] = None,
        days_back: int = 30
    ) -> List[CostRecord]:
        """Retrieve historical cost records."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        records = []
        for file_path in self.recorder.storage_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    record = CostRecord(**data)
                    
                    if record.timestamp >= cutoff_date:
                        if tenant_id is None or record.tenant_id == tenant_id:
                            records.append(record)
                            
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse cost record {file_path}: {e}")
                continue
        
        return sorted(records, key=lambda r: r.timestamp)
    
    def _aggregate_daily_costs(self, records: List[CostRecord]) -> Dict[str, float]:
        """Aggregate costs by day."""
        daily_costs = defaultdict(float)
        
        for record in records:
            date_key = record.timestamp.date().isoformat()
            daily_costs[date_key] += record.total_cost
            
        return dict(daily_costs)
    
    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        """Simple linear regression."""
        n = len(x)
        if n == 0:
            return 0.0, 0.0
            
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0.0
        intercept = y_mean - slope * x_mean
        
        return slope, intercept
    
    def _generate_summary_report(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate summary cost report."""
        prediction = self.predict_monthly_cost(tenant_id, days_history=30)
        patterns = self.analyze_cost_patterns(tenant_id, days_back=30)
        
        # Handle case where patterns might not have the expected structure
        patterns_dict = patterns.get("patterns", {})
        if isinstance(patterns_dict, list):
            # Handle case where patterns is a list instead of dict
            patterns_summary = {
                "total_providers": 0,
                "total_models": 0,
                "total_agents": 0,
                "anomalies_detected": len(patterns.get("anomalies", []))
            }
        else:
            patterns_summary = {
                "total_providers": len(patterns_dict.get("top_providers", [])),
                "total_models": len(patterns_dict.get("top_models", [])),
                "total_agents": len(patterns_dict.get("top_agents", [])),
                "anomalies_detected": len(patterns.get("anomalies", []))
            }
        
        return {
            "report_type": "summary",
            "generated_at": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "prediction": prediction,
            "patterns_summary": patterns_summary
        }
    
    def _generate_detailed_report(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate detailed cost report."""
        prediction = self.predict_monthly_cost(tenant_id, days_history=30)
        patterns = self.analyze_cost_patterns(tenant_id, days_back=30)
        
        return {
            "report_type": "detailed",
            "generated_at": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "prediction": prediction,
            "patterns": patterns
        }
    
    def _generate_predictive_report(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate predictive cost report with multiple scenarios."""
        base_prediction = self.predict_monthly_cost(tenant_id, days_history=30)
        
        # Generate different scenarios
        scenarios = {
            "conservative": base_prediction["predicted_monthly_cost"] * 0.8,
            "expected": base_prediction["predicted_monthly_cost"],
            "optimistic": base_prediction["predicted_monthly_cost"] * 1.2,
            "pessimistic": base_prediction["predicted_monthly_cost"] * 1.5
        }
        
        return {
            "report_type": "predictive",
            "generated_at": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "base_prediction": base_prediction,
            "scenarios": {k: round(v, 4) for k, v in scenarios.items()},
            "recommendations": self._generate_cost_recommendations(base_prediction)
        }
    
    def _generate_cost_recommendations(self, prediction: Dict[str, Any]) -> List[str]:
        """Generate cost optimization recommendations."""
        recommendations = []
        
        trend = prediction.get("trend", "stable")
        confidence = prediction.get("confidence", "medium")
        monthly_cost = prediction.get("predicted_monthly_cost", 0)
        
        if trend == "increasing":
            recommendations.append("Consider implementing cost controls due to increasing trend")
        
        if confidence == "low":
            recommendations.append("Gather more usage data for better cost predictions")
        
        if monthly_cost > 1000:  # Arbitrary threshold
            recommendations.append("High predicted costs - review usage patterns and optimize")
        
        if not recommendations:
            recommendations.append("Cost patterns look stable - continue monitoring")
        
        return recommendations