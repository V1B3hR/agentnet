"""
Performance Regression Testing for AgentNet

Tracks performance trends and detects regressions across versions.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression testing."""
    
    version: str
    timestamp: float
    
    # Performance metrics
    avg_latency_ms: float
    p95_latency_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    
    # Token metrics
    avg_tokens_per_turn: float
    token_efficiency_score: float
    
    # Success metrics
    success_rate: float
    error_rate: float
    
    # Test configuration
    test_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceRegression:
    """Detected performance regression."""
    
    metric_name: str
    current_value: float
    baseline_value: float
    regression_percentage: float
    
    severity: str  # low, medium, high, critical
    threshold_exceeded: bool
    
    # Context
    current_version: str
    baseline_version: str
    test_configuration: str
    
    # Details
    description: str
    recommendation: str


class RegressionTestSuite:
    """
    Performance regression testing suite.
    
    Tracks performance metrics over time and detects regressions
    across different versions and configurations.
    """
    
    def __init__(self, baseline_dir: str = "performance_baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)
        self._baselines: Dict[str, PerformanceBaseline] = {}
        self._regression_thresholds = {
            'avg_latency_ms': 0.20,  # 20% increase is concerning
            'p95_latency_ms': 0.25,  # 25% increase in p95
            'throughput_ops_per_sec': -0.15,  # 15% decrease in throughput
            'success_rate': -0.05,  # 5% decrease in success rate
            'token_efficiency_score': -0.10,  # 10% decrease in efficiency
        }
        
        # Load existing baselines
        self._load_baselines()
    
    def set_regression_threshold(self, metric: str, threshold: float) -> None:
        """Set regression threshold for a metric."""
        self._regression_thresholds[metric] = threshold
        logger.info(f"Set regression threshold for {metric}: {threshold:.1%}")
    
    def create_baseline(
        self,
        version: str,
        performance_data: Dict[str, Any],
        test_configuration: Dict[str, Any],
        force_update: bool = False
    ) -> PerformanceBaseline:
        """Create or update performance baseline."""
        
        baseline_key = f"{version}_{hash(str(test_configuration)) % 10000}"
        
        if baseline_key in self._baselines and not force_update:
            logger.warning(f"Baseline already exists for {version}. Use force_update=True to overwrite.")
            return self._baselines[baseline_key]
        
        baseline = PerformanceBaseline(
            version=version,
            timestamp=time.time(),
            avg_latency_ms=performance_data.get('avg_latency_ms', 0.0),
            p95_latency_ms=performance_data.get('p95_latency_ms', 0.0),
            throughput_ops_per_sec=performance_data.get('throughput_ops_per_sec', 0.0),
            memory_usage_mb=performance_data.get('memory_usage_mb', 0.0),
            avg_tokens_per_turn=performance_data.get('avg_tokens_per_turn', 0.0),
            token_efficiency_score=performance_data.get('token_efficiency_score', 0.0),
            success_rate=performance_data.get('success_rate', 1.0),
            error_rate=performance_data.get('error_rate', 0.0),
            test_configuration=test_configuration,
            metadata=performance_data.get('metadata', {})
        )
        
        self._baselines[baseline_key] = baseline
        self._save_baseline(baseline_key, baseline)
        
        logger.info(f"Created performance baseline for {version}")
        return baseline
    
    def detect_regressions(
        self,
        current_version: str,
        current_performance: Dict[str, Any],
        test_configuration: Dict[str, Any],
        baseline_version: Optional[str] = None
    ) -> List[PerformanceRegression]:
        """Detect performance regressions against baseline."""
        
        # Find appropriate baseline
        baseline = self._find_baseline(baseline_version, test_configuration)
        if not baseline:
            logger.warning("No baseline found for regression comparison")
            return []
        
        regressions = []
        
        # Check each metric against thresholds
        for metric_name, threshold in self._regression_thresholds.items():
            current_value = current_performance.get(metric_name, 0.0)
            baseline_value = getattr(baseline, metric_name, 0.0)
            
            if baseline_value == 0:
                continue  # Skip if baseline has no data
            
            # Calculate regression percentage
            if metric_name in ['throughput_ops_per_sec', 'success_rate', 'token_efficiency_score']:
                # For these metrics, lower is worse
                regression_pct = (current_value - baseline_value) / baseline_value
            else:
                # For latency metrics, higher is worse
                regression_pct = (current_value - baseline_value) / baseline_value
            
            # Check if regression exceeds threshold
            if self._is_regression(metric_name, regression_pct, threshold):
                severity = self._calculate_severity(regression_pct, threshold)
                
                regression = PerformanceRegression(
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    regression_percentage=regression_pct,
                    severity=severity,
                    threshold_exceeded=True,
                    current_version=current_version,
                    baseline_version=baseline.version,
                    test_configuration=str(test_configuration),
                    description=self._generate_regression_description(
                        metric_name, current_value, baseline_value, regression_pct
                    ),
                    recommendation=self._generate_regression_recommendation(metric_name, regression_pct)
                )
                
                regressions.append(regression)
        
        logger.info(f"Detected {len(regressions)} performance regressions")
        return regressions
    
    def _find_baseline(
        self, 
        baseline_version: Optional[str],
        test_configuration: Dict[str, Any]
    ) -> Optional[PerformanceBaseline]:
        """Find appropriate baseline for comparison."""
        
        if baseline_version:
            # Look for specific version
            baseline_key = f"{baseline_version}_{hash(str(test_configuration)) % 10000}"
            return self._baselines.get(baseline_key)
        
        # Find most recent baseline with similar configuration
        matching_baselines = []
        for baseline in self._baselines.values():
            if self._configurations_similar(baseline.test_configuration, test_configuration):
                matching_baselines.append(baseline)
        
        if matching_baselines:
            # Return most recent
            return max(matching_baselines, key=lambda b: b.timestamp)
        
        return None
    
    def _configurations_similar(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> bool:
        """Check if two test configurations are similar enough for comparison."""
        
        # Simple similarity check - in production this would be more sophisticated
        key_fields = ['agent_count', 'feature_set', 'agent_type']
        
        for field in key_fields:
            if config1.get(field) != config2.get(field):
                return False
        
        return True
    
    def _is_regression(self, metric_name: str, regression_pct: float, threshold: float) -> bool:
        """Check if the regression percentage exceeds the threshold."""
        
        if metric_name in ['throughput_ops_per_sec', 'success_rate', 'token_efficiency_score']:
            # For these metrics, negative regression is bad
            return regression_pct < threshold
        else:
            # For latency metrics, positive regression is bad
            return regression_pct > threshold
    
    def _calculate_severity(self, regression_pct: float, threshold: float) -> str:
        """Calculate regression severity."""
        
        abs_regression = abs(regression_pct)
        abs_threshold = abs(threshold)
        
        if abs_regression > abs_threshold * 3:
            return "critical"
        elif abs_regression > abs_threshold * 2:
            return "high"
        elif abs_regression > abs_threshold * 1.5:
            return "medium"
        else:
            return "low"
    
    def _generate_regression_description(
        self, 
        metric_name: str,
        current_value: float,
        baseline_value: float,
        regression_pct: float
    ) -> str:
        """Generate human-readable regression description."""
        
        metric_display = metric_name.replace('_', ' ').title()
        
        if metric_name in ['throughput_ops_per_sec', 'success_rate', 'token_efficiency_score']:
            change_word = "decreased" if regression_pct < 0 else "increased"
        else:
            change_word = "increased" if regression_pct > 0 else "decreased"
        
        return (
            f"{metric_display} {change_word} by {abs(regression_pct):.1%} "
            f"from {baseline_value:.2f} to {current_value:.2f}"
        )
    
    def _generate_regression_recommendation(self, metric_name: str, regression_pct: float) -> str:
        """Generate recommendation for addressing regression."""
        
        recommendations = {
            'avg_latency_ms': "Investigate potential bottlenecks in agent processing pipeline",
            'p95_latency_ms': "Check for outlier scenarios causing high tail latency",
            'throughput_ops_per_sec': "Profile system for performance bottlenecks or resource constraints",
            'success_rate': "Review recent changes that might affect reliability",
            'token_efficiency_score': "Optimize prompts and response generation for better token efficiency",
            'memory_usage_mb': "Investigate memory leaks or inefficient memory usage patterns"
        }
        
        base_recommendation = recommendations.get(
            metric_name, 
            "Investigate recent changes that might have affected performance"
        )
        
        if abs(regression_pct) > 0.5:  # > 50% regression
            return f"URGENT: {base_recommendation}. This is a significant regression."
        elif abs(regression_pct) > 0.25:  # > 25% regression
            return f"HIGH PRIORITY: {base_recommendation}"
        else:
            return base_recommendation
    
    def generate_regression_report(
        self,
        regressions: List[PerformanceRegression],
        current_version: str,
        output_format: str = "markdown"
    ) -> str:
        """Generate regression test report."""
        
        if output_format == "markdown":
            return self._generate_markdown_regression_report(regressions, current_version)
        elif output_format == "json":
            return json.dumps([
                {
                    'metric_name': r.metric_name,
                    'current_value': r.current_value,
                    'baseline_value': r.baseline_value,
                    'regression_percentage': r.regression_percentage,
                    'severity': r.severity,
                    'current_version': r.current_version,
                    'baseline_version': r.baseline_version,
                    'description': r.description,
                    'recommendation': r.recommendation
                }
                for r in regressions
            ], indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_markdown_regression_report(
        self, 
        regressions: List[PerformanceRegression],
        current_version: str
    ) -> str:
        """Generate markdown regression report."""
        
        lines = [
            f"# Performance Regression Report",
            f"",
            f"**Version:** {current_version}",
            f"**Generated:** {datetime.now().isoformat()}",
            f"**Total Regressions:** {len(regressions)}",
            f""
        ]
        
        if not regressions:
            lines.extend([
                "## ✅ No Regressions Detected",
                "",
                "All performance metrics are within acceptable thresholds compared to baseline."
            ])
        else:
            # Group by severity
            by_severity = {}
            for regression in regressions:
                by_severity.setdefault(regression.severity, []).append(regression)
            
            # Report by severity
            severity_order = ['critical', 'high', 'medium', 'low']
            severity_emojis = {'critical': '🔥', 'high': '⚠️', 'medium': '⚡', 'low': 'ℹ️'}
            
            for severity in severity_order:
                if severity in by_severity:
                    emoji = severity_emojis.get(severity, '•')
                    lines.extend([
                        f"## {emoji} {severity.title()} Severity Regressions",
                        ""
                    ])
                    
                    for regression in by_severity[severity]:
                        lines.extend([
                            f"### {regression.metric_name.replace('_', ' ').title()}",
                            f"",
                            f"**Description:** {regression.description}",
                            f"**Baseline Version:** {regression.baseline_version}",
                            f"**Regression:** {regression.regression_percentage:.1%}",
                            f"**Recommendation:** {regression.recommendation}",
                            f""
                        ])
            
            # Summary recommendations
            lines.extend([
                "## Summary Recommendations",
                ""
            ])
            
            critical_count = len(by_severity.get('critical', []))
            high_count = len(by_severity.get('high', []))
            
            if critical_count > 0:
                lines.append(f"🔥 **CRITICAL**: {critical_count} critical regressions require immediate attention")
            if high_count > 0:
                lines.append(f"⚠️ **HIGH**: {high_count} high-priority regressions should be addressed soon")
            
            lines.extend([
                "",
                "Consider:",
                "- Rolling back recent changes if regressions are severe",
                "- Profiling affected systems to identify bottlenecks", 
                "- Updating performance baselines if changes are intentional",
                "- Adding monitoring alerts for affected metrics"
            ])
        
        return "\n".join(lines)
    
    def get_performance_trends(
        self, 
        metric_name: str,
        days_back: int = 30
    ) -> List[Tuple[str, float, float]]:
        """Get performance trends for a metric over time."""
        
        cutoff_time = time.time() - (days_back * 24 * 3600)
        
        trends = []
        for baseline in self._baselines.values():
            if baseline.timestamp >= cutoff_time:
                value = getattr(baseline, metric_name, 0.0)
                trends.append((baseline.version, baseline.timestamp, value))
        
        # Sort by timestamp
        trends.sort(key=lambda x: x[1])
        return trends
    
    def _save_baseline(self, key: str, baseline: PerformanceBaseline) -> None:
        """Save baseline to disk."""
        
        baseline_file = self.baseline_dir / f"{key}.json"
        
        data = {
            'version': baseline.version,
            'timestamp': baseline.timestamp,
            'avg_latency_ms': baseline.avg_latency_ms,
            'p95_latency_ms': baseline.p95_latency_ms,
            'throughput_ops_per_sec': baseline.throughput_ops_per_sec,
            'memory_usage_mb': baseline.memory_usage_mb,
            'avg_tokens_per_turn': baseline.avg_tokens_per_turn,
            'token_efficiency_score': baseline.token_efficiency_score,
            'success_rate': baseline.success_rate,
            'error_rate': baseline.error_rate,
            'test_configuration': baseline.test_configuration,
            'metadata': baseline.metadata
        }
        
        with open(baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_baselines(self) -> None:
        """Load baselines from disk."""
        
        for baseline_file in self.baseline_dir.glob("*.json"):
            try:
                with open(baseline_file, 'r') as f:
                    data = json.load(f)
                
                baseline = PerformanceBaseline(**data)
                key = baseline_file.stem
                self._baselines[key] = baseline
                
            except Exception as e:
                logger.warning(f"Failed to load baseline {baseline_file}: {e}")
        
        logger.info(f"Loaded {len(self._baselines)} performance baselines")
    
    def list_baselines(self) -> List[PerformanceBaseline]:
        """List all available baselines."""
        return list(self._baselines.values())
    
    def delete_baseline(self, version: str, test_configuration: Dict[str, Any]) -> bool:
        """Delete a performance baseline."""
        
        baseline_key = f"{version}_{hash(str(test_configuration)) % 10000}"
        
        if baseline_key in self._baselines:
            del self._baselines[baseline_key]
            
            baseline_file = self.baseline_dir / f"{baseline_key}.json"
            if baseline_file.exists():
                baseline_file.unlink()
            
            logger.info(f"Deleted baseline for {version}")
            return True
        
        return False