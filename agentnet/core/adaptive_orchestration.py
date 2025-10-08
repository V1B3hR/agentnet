"""
Adaptive Orchestration via Performance Feedback Module

Implements dynamic orchestration that adapts based on performance metrics,
automatically tuning agent interactions, roles, and strategies for optimal outcomes.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PerformanceMetric(str, Enum):
    """Types of performance metrics to track."""

    LATENCY = "latency"
    ACCURACY = "accuracy"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CONSENSUS_RATE = "consensus_rate"
    RESOURCE_USAGE = "resource_usage"
    USER_SATISFACTION = "user_satisfaction"
    COST_EFFICIENCY = "cost_efficiency"


class OrchestrationStrategy(str, Enum):
    """Available orchestration strategies."""

    DEBATE = "debate"
    BRAINSTORM = "brainstorm"
    CONSENSUS = "consensus"
    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class OptimizationObjective(str, Enum):
    """Optimization objectives for adaptive orchestration."""

    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    OPTIMIZE_COST = "optimize_cost"
    BALANCE_PERFORMANCE = "balance_performance"
    MINIMIZE_ERRORS = "minimize_errors"


@dataclass
class PerformanceSnapshot:
    """A snapshot of performance metrics at a point in time."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""
    strategy: OrchestrationStrategy = OrchestrationStrategy.DEBATE

    # Core metrics
    metrics: Dict[PerformanceMetric, float] = field(default_factory=dict)

    # Context
    agent_count: int = 0
    task_complexity: float = 0.5
    round_count: int = 0

    # Outcomes
    success: bool = True
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformanceProfile:
    """Performance profile for a specific orchestration strategy."""

    strategy: OrchestrationStrategy
    total_runs: int = 0
    success_rate: float = 0.0
    avg_metrics: Dict[PerformanceMetric, float] = field(default_factory=dict)
    std_metrics: Dict[PerformanceMetric, float] = field(default_factory=dict)
    best_contexts: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class PerformanceFeedbackCollector:
    """
    Collects and analyzes performance feedback for adaptive orchestration.

    Features:
    - Tracks multiple performance metrics
    - Builds performance profiles for strategies
    - Provides recommendations for optimization
    - Supports real-time adaptation
    """

    def __init__(
        self,
        history_window: timedelta = timedelta(hours=24),
        min_samples_for_analysis: int = 5,
        adaptation_threshold: float = 0.1,
    ):
        self.history_window = history_window
        self.min_samples_for_analysis = min_samples_for_analysis
        self.adaptation_threshold = adaptation_threshold

        self.performance_history: List[PerformanceSnapshot] = []
        self.strategy_profiles: Dict[
            OrchestrationStrategy, StrategyPerformanceProfile
        ] = {}

        # Callbacks
        self.on_performance_recorded: Optional[
            Callable[[PerformanceSnapshot], None]
        ] = None
        self.on_strategy_adapted: Optional[
            Callable[[OrchestrationStrategy, OrchestrationStrategy], None]
        ] = None

        logger.info("PerformanceFeedbackCollector initialized")

    def record_performance(
        self,
        session_id: str,
        strategy: OrchestrationStrategy,
        metrics: Dict[PerformanceMetric, float],
        agent_count: int = 0,
        task_complexity: float = 0.5,
        round_count: int = 0,
        success: bool = True,
        error_messages: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a performance snapshot."""

        snapshot = PerformanceSnapshot(
            session_id=session_id,
            strategy=strategy,
            metrics=metrics,
            agent_count=agent_count,
            task_complexity=task_complexity,
            round_count=round_count,
            success=success,
            error_messages=error_messages or [],
            metadata=metadata or {},
        )

        self.performance_history.append(snapshot)

        # Update strategy profile
        self._update_strategy_profile(snapshot)

        # Trigger callback
        if self.on_performance_recorded:
            self.on_performance_recorded(snapshot)

        logger.debug(f"Recorded performance for strategy {strategy}: success={success}")

        return snapshot.id

    def get_strategy_recommendation(
        self,
        context: Dict[str, Any],
        objective: OptimizationObjective = OptimizationObjective.BALANCE_PERFORMANCE,
    ) -> Tuple[OrchestrationStrategy, float]:
        """Get recommended strategy based on context and objective."""

        # Extract context features
        task_complexity = context.get("task_complexity", 0.5)
        agent_count = context.get("agent_count", 2)
        latency_constraint = context.get("max_latency", float("inf"))
        accuracy_requirement = context.get("min_accuracy", 0.0)

        best_strategy = OrchestrationStrategy.DEBATE
        best_score = 0.0

        # Evaluate each strategy
        for strategy, profile in self.strategy_profiles.items():
            if profile.total_runs < self.min_samples_for_analysis:
                continue

            score = self._calculate_strategy_score(
                profile,
                context,
                objective,
                task_complexity,
                agent_count,
                latency_constraint,
                accuracy_requirement,
            )

            if score > best_score:
                best_score = score
                best_strategy = strategy

        # Fallback strategy selection if no profiles available
        if best_score == 0.0:
            best_strategy = self._fallback_strategy_selection(context)
            best_score = 0.5

        logger.info(f"Recommended strategy: {best_strategy} (score: {best_score:.3f})")

        return best_strategy, best_score

    def analyze_performance_trends(self, days_back: int = 7) -> Dict[str, Any]:
        """Analyze performance trends over time."""

        cutoff_time = datetime.now() - timedelta(days=days_back)
        recent_snapshots = [
            snap for snap in self.performance_history if snap.timestamp >= cutoff_time
        ]

        if not recent_snapshots:
            return {"error": "No recent performance data"}

        # Overall trends
        success_rate = sum(1 for snap in recent_snapshots if snap.success) / len(
            recent_snapshots
        )

        # Metric trends
        metric_trends = {}
        for metric in PerformanceMetric:
            values = [
                snap.metrics.get(metric, 0.0)
                for snap in recent_snapshots
                if metric in snap.metrics
            ]
            if values:
                metric_trends[metric] = {
                    "average": statistics.mean(values),
                    "trend": self._calculate_trend(values),
                    "stability": 1.0
                    - (
                        statistics.stdev(values) / statistics.mean(values)
                        if statistics.mean(values) > 0
                        else 0
                    ),
                }

        # Strategy performance comparison
        strategy_comparison = {}
        for strategy in OrchestrationStrategy:
            strategy_snapshots = [
                snap for snap in recent_snapshots if snap.strategy == strategy
            ]
            if strategy_snapshots:
                strategy_success_rate = sum(
                    1 for snap in strategy_snapshots if snap.success
                ) / len(strategy_snapshots)
                strategy_comparison[strategy] = {
                    "runs": len(strategy_snapshots),
                    "success_rate": strategy_success_rate,
                    "avg_latency": (
                        statistics.mean(
                            [
                                snap.metrics.get(PerformanceMetric.LATENCY, 0.0)
                                for snap in strategy_snapshots
                                if PerformanceMetric.LATENCY in snap.metrics
                            ]
                        )
                        if any(
                            PerformanceMetric.LATENCY in snap.metrics
                            for snap in strategy_snapshots
                        )
                        else None
                    ),
                }

        return {
            "analysis_period_days": days_back,
            "total_sessions": len(recent_snapshots),
            "overall_success_rate": success_rate,
            "metric_trends": metric_trends,
            "strategy_comparison": strategy_comparison,
            "recommendations": self._generate_performance_recommendations(
                recent_snapshots
            ),
        }

    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics."""

        recent_snapshots = [
            snap
            for snap in self.performance_history
            if snap.timestamp >= datetime.now() - timedelta(minutes=30)
        ]

        if not recent_snapshots:
            return {"status": "no_recent_data"}

        # Current performance
        current_metrics = {}
        if recent_snapshots:
            latest = recent_snapshots[-1]
            current_metrics = dict(latest.metrics)

        # Short-term trends
        short_term_success_rate = sum(
            1 for snap in recent_snapshots if snap.success
        ) / len(recent_snapshots)

        return {
            "timestamp": datetime.now().isoformat(),
            "recent_sessions": len(recent_snapshots),
            "current_metrics": current_metrics,
            "short_term_success_rate": short_term_success_rate,
            "active_strategies": list(set(snap.strategy for snap in recent_snapshots)),
            "alerts": self._generate_performance_alerts(recent_snapshots),
        }

    def optimize_orchestration(
        self,
        current_strategy: OrchestrationStrategy,
        current_performance: Dict[PerformanceMetric, float],
        context: Dict[str, Any],
        objective: OptimizationObjective = OptimizationObjective.BALANCE_PERFORMANCE,
    ) -> Dict[str, Any]:
        """Optimize orchestration based on current performance."""

        optimization_result = {
            "current_strategy": current_strategy,
            "recommended_changes": [],
            "expected_improvements": {},
            "confidence": 0.0,
        }

        # Get strategy recommendation
        recommended_strategy, confidence = self.get_strategy_recommendation(
            context, objective
        )

        optimization_result["confidence"] = confidence

        # Compare with current strategy
        if recommended_strategy != current_strategy and confidence > 0.6:
            optimization_result["recommended_changes"].append(
                {
                    "type": "strategy_change",
                    "from": current_strategy,
                    "to": recommended_strategy,
                    "reason": f"Better performance expected based on {objective}",
                }
            )

        # Analyze current performance against benchmarks
        current_profile = self.strategy_profiles.get(current_strategy)
        if current_profile:
            for metric, current_value in current_performance.items():
                if metric in current_profile.avg_metrics:
                    avg_value = current_profile.avg_metrics[metric]

                    # Check if significantly below average
                    if current_value < avg_value * 0.8:  # 20% below average
                        optimization_result["recommended_changes"].append(
                            {
                                "type": "performance_improvement",
                                "metric": metric,
                                "current": current_value,
                                "target": avg_value,
                                "suggestion": self._get_improvement_suggestion(
                                    metric, current_value, avg_value
                                ),
                            }
                        )

        # Parameter tuning suggestions
        tuning_suggestions = self._get_parameter_tuning_suggestions(
            current_strategy, current_performance, context
        )
        optimization_result["recommended_changes"].extend(tuning_suggestions)

        return optimization_result

    def export_performance_report(self, filepath: str, days_back: int = 30) -> None:
        """Export comprehensive performance report."""

        cutoff_time = datetime.now() - timedelta(days=days_back)
        relevant_snapshots = [
            snap for snap in self.performance_history if snap.timestamp >= cutoff_time
        ]

        report = {
            "report_generated": datetime.now().isoformat(),
            "analysis_period_days": days_back,
            "total_sessions": len(relevant_snapshots),
            "performance_analysis": self.analyze_performance_trends(days_back),
            "strategy_profiles": {
                strategy.value: {
                    "total_runs": profile.total_runs,
                    "success_rate": profile.success_rate,
                    "avg_metrics": {k.value: v for k, v in profile.avg_metrics.items()},
                    "last_updated": profile.last_updated.isoformat(),
                }
                for strategy, profile in self.strategy_profiles.items()
            },
            "raw_snapshots": [
                {
                    "id": snap.id,
                    "timestamp": snap.timestamp.isoformat(),
                    "session_id": snap.session_id,
                    "strategy": snap.strategy.value,
                    "metrics": {k.value: v for k, v in snap.metrics.items()},
                    "agent_count": snap.agent_count,
                    "task_complexity": snap.task_complexity,
                    "success": snap.success,
                    "metadata": snap.metadata,
                }
                for snap in relevant_snapshots
            ],
        }

        import json

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported performance report to {filepath}")

    def _update_strategy_profile(self, snapshot: PerformanceSnapshot) -> None:
        """Update the performance profile for a strategy."""

        strategy = snapshot.strategy

        if strategy not in self.strategy_profiles:
            self.strategy_profiles[strategy] = StrategyPerformanceProfile(
                strategy=strategy
            )

        profile = self.strategy_profiles[strategy]

        # Update counts
        profile.total_runs += 1
        profile.success_rate = (
            profile.success_rate * (profile.total_runs - 1)
            + (1 if snapshot.success else 0)
        ) / profile.total_runs

        # Update metrics
        for metric, value in snapshot.metrics.items():
            if metric not in profile.avg_metrics:
                profile.avg_metrics[metric] = value
                profile.std_metrics[metric] = 0.0
            else:
                # Update running average and std
                old_avg = profile.avg_metrics[metric]
                new_avg = old_avg + (value - old_avg) / profile.total_runs
                profile.avg_metrics[metric] = new_avg

                # Simplified std update (not perfectly accurate but efficient)
                if profile.total_runs > 1:
                    profile.std_metrics[metric] = (
                        profile.std_metrics[metric] * 0.9 + abs(value - new_avg) * 0.1
                    )

        profile.last_updated = datetime.now()

    def _calculate_strategy_score(
        self,
        profile: StrategyPerformanceProfile,
        context: Dict[str, Any],
        objective: OptimizationObjective,
        task_complexity: float,
        agent_count: int,
        latency_constraint: float,
        accuracy_requirement: float,
    ) -> float:
        """Calculate a score for a strategy given context and objectives."""

        base_score = profile.success_rate

        # Objective-specific scoring
        if objective == OptimizationObjective.MINIMIZE_LATENCY:
            latency = profile.avg_metrics.get(PerformanceMetric.LATENCY, float("inf"))
            if latency <= latency_constraint:
                base_score += (latency_constraint - latency) / latency_constraint
            else:
                base_score *= 0.5  # Penalty for exceeding constraint

        elif objective == OptimizationObjective.MAXIMIZE_ACCURACY:
            accuracy = profile.avg_metrics.get(PerformanceMetric.ACCURACY, 0.0)
            if accuracy >= accuracy_requirement:
                base_score += accuracy
            else:
                base_score *= 0.7  # Penalty for not meeting requirement

        elif objective == OptimizationObjective.OPTIMIZE_COST:
            cost_efficiency = profile.avg_metrics.get(
                PerformanceMetric.COST_EFFICIENCY, 0.5
            )
            base_score += cost_efficiency

        elif objective == OptimizationObjective.BALANCE_PERFORMANCE:
            # Balanced scoring across multiple metrics
            accuracy = profile.avg_metrics.get(PerformanceMetric.ACCURACY, 0.5)
            latency_score = 1.0 - min(
                1.0, profile.avg_metrics.get(PerformanceMetric.LATENCY, 1.0) / 10.0
            )
            error_rate = profile.avg_metrics.get(PerformanceMetric.ERROR_RATE, 0.5)

            base_score = (
                base_score + accuracy + latency_score + (1.0 - error_rate)
            ) / 4.0

        # Context adjustments
        if task_complexity > 0.7 and profile.strategy in [
            OrchestrationStrategy.DEBATE,
            OrchestrationStrategy.CONSENSUS,
        ]:
            base_score += 0.1  # Bonus for complex tasks

        if agent_count > 5 and profile.strategy == OrchestrationStrategy.HIERARCHICAL:
            base_score += 0.1  # Bonus for many agents

        return max(0.0, min(1.0, base_score))

    def _fallback_strategy_selection(
        self, context: Dict[str, Any]
    ) -> OrchestrationStrategy:
        """Select strategy when no performance data is available."""

        task_complexity = context.get("task_complexity", 0.5)
        agent_count = context.get("agent_count", 2)

        if task_complexity > 0.7:
            return OrchestrationStrategy.DEBATE
        elif agent_count > 4:
            return OrchestrationStrategy.HIERARCHICAL
        else:
            return OrchestrationStrategy.CONSENSUS

    def _calculate_trend(self, values: List[float], window: int = 5) -> float:
        """Calculate trend direction from a series of values."""

        if len(values) < window:
            return 0.0

        recent = values[-window:]
        older = (
            values[-(window * 2) : -window]
            if len(values) >= window * 2
            else values[:-window]
        )

        if not older:
            return 0.0

        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)

        return (recent_avg - older_avg) / older_avg if older_avg != 0 else 0.0

    def _generate_performance_recommendations(
        self, snapshots: List[PerformanceSnapshot]
    ) -> List[str]:
        """Generate recommendations based on performance analysis."""

        recommendations = []

        # Success rate analysis
        success_rate = sum(1 for snap in snapshots if snap.success) / len(snapshots)
        if success_rate < 0.8:
            recommendations.append(
                "Low success rate detected - consider strategy adjustment"
            )

        # Latency analysis
        latency_values = [
            snap.metrics.get(PerformanceMetric.LATENCY, 0.0)
            for snap in snapshots
            if PerformanceMetric.LATENCY in snap.metrics
        ]
        if latency_values and statistics.mean(latency_values) > 5.0:
            recommendations.append(
                "High latency detected - consider parallel processing"
            )

        # Error rate analysis
        error_rates = [
            snap.metrics.get(PerformanceMetric.ERROR_RATE, 0.0)
            for snap in snapshots
            if PerformanceMetric.ERROR_RATE in snap.metrics
        ]
        if error_rates and statistics.mean(error_rates) > 0.1:
            recommendations.append("High error rate - review agent coordination")

        return recommendations

    def _generate_performance_alerts(
        self, recent_snapshots: List[PerformanceSnapshot]
    ) -> List[Dict[str, Any]]:
        """Generate performance alerts based on recent data."""

        alerts = []

        if not recent_snapshots:
            return alerts

        # Failure rate alert
        recent_failures = sum(1 for snap in recent_snapshots if not snap.success)
        if recent_failures > len(recent_snapshots) * 0.3:
            alerts.append(
                {
                    "type": "high_failure_rate",
                    "severity": "warning",
                    "message": f"High failure rate: {recent_failures}/{len(recent_snapshots)} recent sessions failed",
                }
            )

        # Latency spike alert
        recent_latencies = [
            snap.metrics.get(PerformanceMetric.LATENCY, 0.0)
            for snap in recent_snapshots
            if PerformanceMetric.LATENCY in snap.metrics
        ]
        if recent_latencies and max(recent_latencies) > 10.0:
            alerts.append(
                {
                    "type": "latency_spike",
                    "severity": "warning",
                    "message": f"Latency spike detected: {max(recent_latencies):.2f}s",
                }
            )

        return alerts

    def _get_improvement_suggestion(
        self, metric: PerformanceMetric, current: float, target: float
    ) -> str:
        """Get improvement suggestion for a specific metric."""

        suggestions = {
            PerformanceMetric.LATENCY: "Consider parallel processing or caching",
            PerformanceMetric.ACCURACY: "Review agent prompts and validation logic",
            PerformanceMetric.ERROR_RATE: "Improve error handling and retry mechanisms",
            PerformanceMetric.THROUGHPUT: "Optimize agent scheduling and resource allocation",
            PerformanceMetric.CONSENSUS_RATE: "Adjust consensus thresholds or add mediation",
            PerformanceMetric.COST_EFFICIENCY: "Optimize model selection and prompt engineering",
        }

        return suggestions.get(metric, "Monitor and analyze further")

    def _get_parameter_tuning_suggestions(
        self,
        strategy: OrchestrationStrategy,
        performance: Dict[PerformanceMetric, float],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Get parameter tuning suggestions for current strategy."""

        suggestions = []

        # Strategy-specific tuning
        if strategy == OrchestrationStrategy.DEBATE:
            if performance.get(PerformanceMetric.LATENCY, 0.0) > 5.0:
                suggestions.append(
                    {
                        "type": "parameter_tuning",
                        "parameter": "max_rounds",
                        "current_issue": "high_latency",
                        "suggestion": "Reduce maximum debate rounds to improve response time",
                    }
                )

        elif strategy == OrchestrationStrategy.CONSENSUS:
            consensus_rate = performance.get(PerformanceMetric.CONSENSUS_RATE, 0.0)
            if consensus_rate < 0.7:
                suggestions.append(
                    {
                        "type": "parameter_tuning",
                        "parameter": "consensus_threshold",
                        "current_issue": "low_consensus",
                        "suggestion": "Lower consensus threshold or add mediator agent",
                    }
                )

        return suggestions


class AdaptiveOrchestrator:
    """
    Adaptive orchestrator that dynamically adjusts strategy based on performance feedback.

    Features:
    - Integrates with PerformanceFeedbackCollector
    - Automatically switches strategies based on performance
    - Provides real-time optimization
    - Supports multiple optimization objectives
    """

    def __init__(
        self,
        feedback_collector: PerformanceFeedbackCollector,
        adaptation_interval: timedelta = timedelta(minutes=5),
        min_performance_threshold: float = 0.7,
    ):
        self.feedback_collector = feedback_collector
        self.adaptation_interval = adaptation_interval
        self.min_performance_threshold = min_performance_threshold

        self.current_strategy = OrchestrationStrategy.DEBATE
        self.current_objective = OptimizationObjective.BALANCE_PERFORMANCE
        self.last_adaptation = datetime.now()

        # Set up feedback collector callbacks
        self.feedback_collector.on_strategy_adapted = self._on_strategy_adapted

        logger.info("AdaptiveOrchestrator initialized")

    def set_optimization_objective(self, objective: OptimizationObjective) -> None:
        """Set the optimization objective for the orchestrator."""

        self.current_objective = objective
        logger.info(f"Optimization objective set to: {objective}")

    def should_adapt_strategy(
        self,
        current_performance: Dict[PerformanceMetric, float],
        context: Dict[str, Any],
    ) -> bool:
        """Determine if strategy should be adapted based on current performance."""

        # Check adaptation interval
        if datetime.now() - self.last_adaptation < self.adaptation_interval:
            return False

        # Check performance threshold
        success_indicator = current_performance.get(PerformanceMetric.ACCURACY, 0.5)
        if success_indicator < self.min_performance_threshold:
            return True

        # Get recommendation
        recommended_strategy, confidence = (
            self.feedback_collector.get_strategy_recommendation(
                context, self.current_objective
            )
        )

        return recommended_strategy != self.current_strategy and confidence > 0.7

    def adapt_strategy(
        self,
        current_performance: Dict[PerformanceMetric, float],
        context: Dict[str, Any],
    ) -> Optional[OrchestrationStrategy]:
        """Adapt orchestration strategy based on performance."""

        if not self.should_adapt_strategy(current_performance, context):
            return None

        # Get new strategy recommendation
        recommended_strategy, confidence = (
            self.feedback_collector.get_strategy_recommendation(
                context, self.current_objective
            )
        )

        old_strategy = self.current_strategy
        self.current_strategy = recommended_strategy
        self.last_adaptation = datetime.now()

        logger.info(
            f"Strategy adapted from {old_strategy} to {recommended_strategy} (confidence: {confidence:.3f})"
        )

        return recommended_strategy

    def get_orchestration_config(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get current orchestration configuration."""

        return {
            "strategy": self.current_strategy,
            "objective": self.current_objective,
            "last_adaptation": self.last_adaptation.isoformat(),
            "recommended_parameters": self._get_strategy_parameters(
                self.current_strategy, context
            ),
        }

    def _on_strategy_adapted(
        self, old_strategy: OrchestrationStrategy, new_strategy: OrchestrationStrategy
    ) -> None:
        """Callback when strategy is adapted."""

        logger.info(f"Strategy adaptation callback: {old_strategy} -> {new_strategy}")

    def _get_strategy_parameters(
        self, strategy: OrchestrationStrategy, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get recommended parameters for a strategy."""

        base_params = {
            "max_rounds": 3,
            "timeout_seconds": 30,
            "convergence_threshold": 0.8,
        }

        # Strategy-specific adjustments
        if strategy == OrchestrationStrategy.DEBATE:
            base_params.update(
                {"max_rounds": 5, "require_consensus": False, "enable_rebuttals": True}
            )
        elif strategy == OrchestrationStrategy.CONSENSUS:
            base_params.update(
                {
                    "consensus_threshold": 0.75,
                    "max_iterations": 3,
                    "allow_abstentions": True,
                }
            )
        elif strategy == OrchestrationStrategy.PARALLEL:
            base_params.update(
                {
                    "parallel_execution": True,
                    "merge_strategy": "weighted_average",
                    "timeout_seconds": 15,
                }
            )

        # Context-based adjustments
        task_complexity = context.get("task_complexity", 0.5)
        if task_complexity > 0.7:
            base_params["max_rounds"] += 2
            base_params["timeout_seconds"] += 20

        return base_params
