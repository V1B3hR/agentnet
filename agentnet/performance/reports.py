"""
Performance Report Generation for AgentNet

Generates comprehensive performance reports with trends, visualizations,
and actionable recommendations as specified in Phase 5 requirements.
"""

import json
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from .harness import BenchmarkResult
from .latency import LatencyTracker, TurnLatencyMeasurement
from .tokens import TokenUtilizationTracker, TokenMetrics

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Supported report formats."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"


class PerformanceReporter:
    """
    Generates comprehensive performance reports from benchmark data,
    latency measurements, and token utilization metrics.
    """
    
    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_comprehensive_report(
        self,
        benchmark_results: List[BenchmarkResult],
        latency_tracker: Optional[LatencyTracker] = None,
        token_tracker: Optional[TokenUtilizationTracker] = None,
        format: ReportFormat = ReportFormat.MARKDOWN,
        include_recommendations: bool = True
    ) -> str:
        """Generate a comprehensive performance report."""
        
        timestamp = datetime.now()
        report_data = {
            'metadata': {
                'generated_at': timestamp.isoformat(),
                'agentnet_version': '0.5.0',
                'report_type': 'comprehensive_performance',
                'format': format.value
            },
            'executive_summary': self._generate_executive_summary(
                benchmark_results, latency_tracker, token_tracker
            ),
            'benchmark_analysis': self._analyze_benchmarks(benchmark_results),
            'latency_analysis': {},
            'token_analysis': {},
            'trends': {},
            'recommendations': []
        }
        
        # Add latency analysis if available
        if latency_tracker:
            report_data['latency_analysis'] = self._analyze_latency(latency_tracker)
        
        # Add token analysis if available
        if token_tracker:
            report_data['token_analysis'] = self._analyze_tokens(token_tracker)
        
        # Generate trends analysis
        report_data['trends'] = self._analyze_trends(
            benchmark_results, latency_tracker, token_tracker
        )
        
        # Generate recommendations
        if include_recommendations:
            report_data['recommendations'] = self._generate_recommendations(
                report_data, latency_tracker, token_tracker
            )
        
        # Format and save report
        if format == ReportFormat.MARKDOWN:
            content = self._format_markdown_report(report_data)
            filename = f"performance_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        elif format == ReportFormat.JSON:
            content = json.dumps(report_data, indent=2, default=str)
            filename = f"performance_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        elif format == ReportFormat.HTML:
            content = self._format_html_report(report_data)
            filename = f"performance_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Write report to file
        report_path = self.output_dir / filename
        with open(report_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Generated performance report: {report_path}")
        return str(report_path)
    
    def _generate_executive_summary(
        self,
        benchmark_results: List[BenchmarkResult],
        latency_tracker: Optional[LatencyTracker],
        token_tracker: Optional[TokenUtilizationTracker]
    ) -> Dict[str, Any]:
        """Generate executive summary of performance metrics."""
        
        summary = {
            'total_benchmarks': len(benchmark_results),
            'benchmark_success_rate': 0.0,
            'avg_turn_latency_ms': 0.0,
            'total_tokens_analyzed': 0,
            'overall_efficiency_score': 0.0,
            'key_findings': []
        }
        
        if benchmark_results:
            # Benchmark metrics
            successful_benchmarks = [r for r in benchmark_results if r.passed_thresholds]
            summary['benchmark_success_rate'] = len(successful_benchmarks) / len(benchmark_results)
            summary['avg_turn_latency_ms'] = sum(r.avg_turn_latency_ms for r in benchmark_results) / len(benchmark_results)
            
            # Key findings from benchmarks
            if summary['benchmark_success_rate'] < 0.8:
                summary['key_findings'].append("⚠️ Multiple benchmarks failed to meet performance thresholds")
            
            high_latency_benchmarks = [r for r in benchmark_results if r.avg_turn_latency_ms > 2000]
            if high_latency_benchmarks:
                summary['key_findings'].append(f"🐌 {len(high_latency_benchmarks)} benchmarks showed high latency (>2s)")
        
        # Token analysis
        if token_tracker:
            system_overview = token_tracker.get_system_token_overview()
            if system_overview:
                summary['total_tokens_analyzed'] = system_overview['overview']['total_tokens']
                summary['overall_efficiency_score'] = system_overview['overview']['avg_efficiency']
                
                if summary['overall_efficiency_score'] < 0.6:
                    summary['key_findings'].append("📉 Token efficiency is below optimal levels")
        
        # Latency analysis
        if latency_tracker:
            measurements = latency_tracker.get_measurements()
            if measurements:
                avg_latency = sum(m.total_latency_ms for m in measurements) / len(measurements)
                if avg_latency > 1500:
                    summary['key_findings'].append("⏱️ Average turn latency exceeds recommended thresholds")
        
        if not summary['key_findings']:
            summary['key_findings'].append("✅ No significant performance issues detected")
        
        return summary
    
    def _analyze_benchmarks(self, benchmark_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze benchmark results for patterns and insights."""
        
        if not benchmark_results:
            return {}
        
        analysis = {
            'summary': {
                'total_benchmarks': len(benchmark_results),
                'passed_benchmarks': len([r for r in benchmark_results if r.passed_thresholds]),
                'failed_benchmarks': len([r for r in benchmark_results if not r.passed_thresholds])
            },
            'latency_metrics': {},
            'throughput_metrics': {},
            'success_metrics': {},
            'benchmark_breakdown': []
        }
        
        # Latency analysis
        latencies = [r.avg_turn_latency_ms for r in benchmark_results]
        analysis['latency_metrics'] = {
            'avg_latency_ms': sum(latencies) / len(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        }
        
        # Throughput analysis
        throughputs = [r.operations_per_second for r in benchmark_results]
        analysis['throughput_metrics'] = {
            'avg_ops_per_sec': sum(throughputs) / len(throughputs),
            'min_ops_per_sec': min(throughputs),
            'max_ops_per_sec': max(throughputs)
        }
        
        # Success rate analysis
        success_rates = [r.success_rate for r in benchmark_results]
        analysis['success_metrics'] = {
            'avg_success_rate': sum(success_rates) / len(success_rates),
            'min_success_rate': min(success_rates),
            'max_success_rate': max(success_rates)
        }
        
        # Individual benchmark breakdown
        for result in benchmark_results:
            analysis['benchmark_breakdown'].append({
                'name': result.config.name,
                'type': result.config.benchmark_type.value,
                'latency_ms': result.avg_turn_latency_ms,
                'success_rate': result.success_rate,
                'ops_per_sec': result.operations_per_second,
                'passed_thresholds': result.passed_thresholds
            })
        
        return analysis
    
    def _analyze_latency(self, latency_tracker: LatencyTracker) -> Dict[str, Any]:
        """Analyze latency measurements for insights."""
        
        measurements = latency_tracker.get_measurements()
        if not measurements:
            return {}
        
        analysis = {
            'summary': {
                'total_measurements': len(measurements),
                'unique_agents': len(set(m.agent_id for m in measurements))
            },
            'overall_statistics': latency_tracker.get_latency_statistics(),
            'component_breakdown': latency_tracker.get_component_breakdown(),
            'performance_issues': latency_tracker.identify_performance_issues(),
            'agent_analysis': {}
        }
        
        # Per-agent analysis
        agent_ids = set(m.agent_id for m in measurements)
        for agent_id in agent_ids:
            agent_stats = latency_tracker.get_latency_statistics(agent_id=agent_id)
            agent_issues = latency_tracker.identify_performance_issues(agent_id=agent_id)
            
            analysis['agent_analysis'][agent_id] = {
                'statistics': agent_stats,
                'issues': agent_issues
            }
        
        return analysis
    
    def _analyze_tokens(self, token_tracker: TokenUtilizationTracker) -> Dict[str, Any]:
        """Analyze token utilization for insights."""
        
        system_overview = token_tracker.get_system_token_overview()
        if not system_overview:
            return {}
        
        analysis = {
            'system_overview': system_overview,
            'optimization_opportunities': token_tracker.identify_optimization_opportunities(),
            'recommendations': token_tracker.generate_optimization_recommendations(),
            'agent_breakdown': {}
        }
        
        # Per-agent token analysis
        for agent_id in system_overview['agents']:
            agent_summary = token_tracker.get_agent_token_summary(agent_id)
            agent_opportunities = token_tracker.identify_optimization_opportunities(agent_id)
            
            analysis['agent_breakdown'][agent_id] = {
                'summary': agent_summary,
                'opportunities': agent_opportunities
            }
        
        return analysis
    
    def _analyze_trends(
        self,
        benchmark_results: List[BenchmarkResult],
        latency_tracker: Optional[LatencyTracker],
        token_tracker: Optional[TokenUtilizationTracker]
    ) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        
        trends = {
            'benchmark_trends': {},
            'latency_trends': {},
            'token_trends': {}
        }
        
        # Benchmark trends (if multiple results over time)
        if len(benchmark_results) > 1:
            # Sort by start time
            sorted_results = sorted(benchmark_results, key=lambda r: r.start_time)
            
            latency_trend = [r.avg_turn_latency_ms for r in sorted_results]
            success_trend = [r.success_rate for r in sorted_results]
            
            trends['benchmark_trends'] = {
                'latency_trend': 'improving' if latency_trend[-1] < latency_trend[0] else 'degrading',
                'success_trend': 'improving' if success_trend[-1] > success_trend[0] else 'degrading',
                'latest_vs_first': {
                    'latency_change_pct': ((latency_trend[-1] - latency_trend[0]) / latency_trend[0]) * 100,
                    'success_change_pct': ((success_trend[-1] - success_trend[0]) / success_trend[0]) * 100
                }
            }
        
        # Additional trend analysis would require historical data storage
        # For now, provide basic structure
        
        return trends
    
    def _generate_recommendations(
        self,
        report_data: Dict[str, Any],
        latency_tracker: Optional[LatencyTracker],
        token_tracker: Optional[TokenUtilizationTracker]
    ) -> List[Dict[str, str]]:
        """Generate actionable performance recommendations."""
        
        recommendations = []
        
        # Benchmark-based recommendations
        benchmark_analysis = report_data.get('benchmark_analysis', {})
        if benchmark_analysis:
            avg_latency = benchmark_analysis.get('latency_metrics', {}).get('avg_latency_ms', 0)
            avg_success_rate = benchmark_analysis.get('success_metrics', {}).get('avg_success_rate', 1.0)
            
            if avg_latency > 2000:
                recommendations.append({
                    'category': 'latency',
                    'priority': 'high',
                    'title': 'High Average Latency Detected',
                    'description': f'Average turn latency of {avg_latency:.1f}ms exceeds recommended thresholds',
                    'action': 'Consider optimizing prompts, using faster models, or implementing caching'
                })
            
            if avg_success_rate < 0.9:
                recommendations.append({
                    'category': 'reliability',
                    'priority': 'high',
                    'title': 'Low Success Rate',
                    'description': f'Success rate of {avg_success_rate:.1%} indicates reliability issues',
                    'action': 'Review error patterns and implement better error handling'
                })
        
        # Token-based recommendations
        if token_tracker:
            token_recommendations = token_tracker.generate_optimization_recommendations()
            for i, rec in enumerate(token_recommendations):
                recommendations.append({
                    'category': 'token_efficiency',
                    'priority': 'medium',
                    'title': f'Token Optimization #{i+1}',
                    'description': rec,
                    'action': 'Implement suggested token optimization'
                })
        
        # Latency-based recommendations
        if latency_tracker:
            performance_issues = latency_tracker.identify_performance_issues()
            
            if performance_issues.get('high_latency_turns'):
                recommendations.append({
                    'category': 'latency',
                    'priority': 'medium',
                    'title': 'High Latency Turns Detected',
                    'description': f"{len(performance_issues['high_latency_turns'])} turns exceeded latency thresholds",
                    'action': 'Investigate specific high-latency operations and optimize bottlenecks'
                })
        
        # General recommendations
        if not recommendations:
            recommendations.append({
                'category': 'general',
                'priority': 'low',
                'title': 'Performance Baseline Established',
                'description': 'Current performance metrics look healthy',
                'action': 'Continue monitoring and establish automated alerting for regressions'
            })
        
        return recommendations
    
    def _format_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Format report data as Markdown."""
        
        md = []
        metadata = report_data['metadata']
        
        # Header
        md.append(f"# AgentNet Performance Report")
        md.append(f"")
        md.append(f"**Generated:** {metadata['generated_at']}")
        md.append(f"**AgentNet Version:** {metadata['agentnet_version']}")
        md.append(f"")
        
        # Executive Summary
        summary = report_data['executive_summary']
        md.append("## Executive Summary")
        md.append("")
        md.append(f"- **Total Benchmarks:** {summary['total_benchmarks']}")
        md.append(f"- **Benchmark Success Rate:** {summary['benchmark_success_rate']:.1%}")
        md.append(f"- **Average Turn Latency:** {summary['avg_turn_latency_ms']:.1f}ms")
        md.append(f"- **Total Tokens Analyzed:** {summary['total_tokens_analyzed']:,}")
        md.append(f"- **Overall Efficiency Score:** {summary['overall_efficiency_score']:.3f}")
        md.append("")
        
        # Key Findings
        md.append("### Key Findings")
        md.append("")
        for finding in summary['key_findings']:
            md.append(f"- {finding}")
        md.append("")
        
        # Benchmark Analysis
        benchmark_analysis = report_data.get('benchmark_analysis', {})
        if benchmark_analysis:
            md.append("## Benchmark Analysis")
            md.append("")
            
            # Summary table
            md.append("### Performance Metrics")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            
            latency_metrics = benchmark_analysis.get('latency_metrics', {})
            for metric, value in latency_metrics.items():
                md.append(f"| {metric.replace('_', ' ').title()} | {value:.1f}ms |")
            
            throughput_metrics = benchmark_analysis.get('throughput_metrics', {})
            for metric, value in throughput_metrics.items():
                md.append(f"| {metric.replace('_', ' ').title()} | {value:.2f} |")
            
            md.append("")
        
        # Recommendations
        recommendations = report_data.get('recommendations', [])
        if recommendations:
            md.append("## Recommendations")
            md.append("")
            
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {"high": "🔥", "medium": "⚠️", "low": "ℹ️"}.get(rec['priority'], "•")
                md.append(f"### {i}. {priority_emoji} {rec['title']}")
                md.append(f"**Category:** {rec['category'].title()}")
                md.append(f"**Priority:** {rec['priority'].title()}")
                md.append(f"**Description:** {rec['description']}")
                md.append(f"**Action:** {rec['action']}")
                md.append("")
        
        # Footer
        md.append("---")
        md.append("*Generated by AgentNet Performance Harness*")
        
        return "\n".join(md)
    
    def _format_html_report(self, report_data: Dict[str, Any]) -> str:
        """Format report data as HTML."""
        
        metadata = report_data['metadata']
        summary = report_data['executive_summary']
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>AgentNet Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #e9f4ff; border-radius: 5px; }}
        .recommendation {{ margin: 15px 0; padding: 15px; border-left: 4px solid #007acc; background: #f9f9f9; }}
        .high-priority {{ border-color: #ff4444; }}
        .medium-priority {{ border-color: #ffaa00; }}
        .low-priority {{ border-color: #00aa44; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AgentNet Performance Report</h1>
        <p><strong>Generated:</strong> {metadata['generated_at']}</p>
        <p><strong>AgentNet Version:</strong> {metadata['agentnet_version']}</p>
    </div>
    
    <h2>Executive Summary</h2>
    <div>
        <div class="metric">
            <strong>Total Benchmarks</strong><br>
            {summary['total_benchmarks']}
        </div>
        <div class="metric">
            <strong>Success Rate</strong><br>
            {summary['benchmark_success_rate']:.1%}
        </div>
        <div class="metric">
            <strong>Avg Latency</strong><br>
            {summary['avg_turn_latency_ms']:.1f}ms
        </div>
        <div class="metric">
            <strong>Efficiency Score</strong><br>
            {summary['overall_efficiency_score']:.3f}
        </div>
    </div>
    
    <h3>Key Findings</h3>
    <ul>
"""
        
        for finding in summary['key_findings']:
            html += f"        <li>{finding}</li>\n"
        
        html += "    </ul>\n"
        
        # Recommendations
        recommendations = report_data.get('recommendations', [])
        if recommendations:
            html += "    <h2>Recommendations</h2>\n"
            
            for i, rec in enumerate(recommendations, 1):
                priority_class = f"{rec['priority']}-priority"
                html += f"""
    <div class="recommendation {priority_class}">
        <h3>{i}. {rec['title']}</h3>
        <p><strong>Category:</strong> {rec['category'].title()}</p>
        <p><strong>Priority:</strong> {rec['priority'].title()}</p>
        <p><strong>Description:</strong> {rec['description']}</p>
        <p><strong>Action:</strong> {rec['action']}</p>
    </div>
"""
        
        html += """
    <hr>
    <p><em>Generated by AgentNet Performance Harness</em></p>
</body>
</html>"""
        
        return html