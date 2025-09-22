"""
Ethics monitor integration for AgentNet monitoring system.

This module provides integration between the centralized EthicsJudge and
the existing monitor system, allowing ethics oversight to be applied
consistently across all agent operations.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ..core.policy.ethics import get_ethics_judge, EthicsJudge, EthicsViolation
from .base import MonitorTemplate

logger = logging.getLogger("agentnet.monitors.ethics")


class EthicsMonitor(MonitorTemplate):
    """
    Monitor that integrates the centralized EthicsJudge with the monitoring system.
    
    This monitor delegates ethics evaluation to the singleton EthicsJudge,
    ensuring consistent ethics oversight across all agent operations.
    """
    
    def __init__(self, name: str = "ethics_monitor", 
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ethics monitor.
        
        Args:
            name: Monitor name
            config: Optional configuration for the monitor
        """
        super().__init__(name, config or {})
        self.ethics_judge = get_ethics_judge()
        self.evaluation_count = 0
        self.violation_count = 0
        
        # Configure ethics judge if config provided
        if config and 'ethics_config' in config:
            self.ethics_judge.configure(config['ethics_config'])
        
        logger.info(f"EthicsMonitor '{name}' initialized with centralized EthicsJudge")
    
    def check(self, agent: Any, task: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check the result using the centralized EthicsJudge.
        
        Args:
            agent: The agent instance
            task: The task being monitored
            result: The result to check
            
        Returns:
            Dictionary with monitoring results
        """
        start_time = time.perf_counter()
        
        try:
            self.evaluation_count += 1
            
            # Use EthicsJudge for evaluation
            outcome = result if isinstance(result, dict) else {"content": str(result)}
            passed, violations = self.ethics_judge.evaluate(outcome)
            
            evaluation_time = time.perf_counter() - start_time
            
            if violations:
                self.violation_count += len(violations)
                
                # Create violation details
                violation_details = []
                for violation in violations:
                    violation_details.append({
                        "type": violation.violation_type.value,
                        "severity": violation.severity.value,
                        "description": violation.description,
                        "rule_name": violation.rule_name,
                        "rationale": violation.rationale
                    })
                
                logger.warning(f"EthicsMonitor '{self.name}' found {len(violations)} violations")
                
                return {
                    "passed": passed,
                    "violations": violation_details,
                    "evaluation_time": evaluation_time,
                    "message": f"Ethics violations detected: {len(violations)} issues found"
                }
            else:
                logger.debug(f"EthicsMonitor '{self.name}' evaluation passed")
                return {
                    "passed": True,
                    "violations": [],
                    "evaluation_time": evaluation_time,
                    "message": None
                }
                
        except Exception as e:
            evaluation_time = time.perf_counter() - start_time
            logger.error(f"EthicsMonitor '{self.name}' evaluation failed: {e}")
            return {
                "passed": False,
                "violations": [],
                "evaluation_time": evaluation_time,
                "message": f"Ethics evaluation error: {str(e)}"
            }
    
    def evaluate(self, outcome: Dict[str, Any]) -> Tuple[bool, Optional[str], float]:
        """
        Evaluate outcome using the centralized EthicsJudge.
        
        This method provides backward compatibility with the expected monitor interface.
        
        Args:
            outcome: The outcome to evaluate
            
        Returns:
            Tuple of (passed, message, evaluation_time)
        """
        result = self.check(None, "evaluation", outcome)
        return result["passed"], result["message"], result["evaluation_time"]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        ethics_stats = self.ethics_judge.get_statistics()
        
        return {
            "name": self.name,
            "enabled": self.enabled,
            "ethics_judge_stats": ethics_stats,
            "monitor_evaluations": self.evaluation_count,
            "monitor_violations": self.violation_count,
        }
    
    def get_recent_violations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent ethics violations."""
        violations = self.ethics_judge.get_violation_history(limit)
        return [v.to_dict() for v in violations]


def create_ethics_monitor_spec(name: str = "central_ethics",
                             severity: str = "severe",
                             description: str = "Central ethics oversight using EthicsJudge",
                             ethics_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a monitor specification for the EthicsMonitor.
    
    Args:
        name: Monitor name
        severity: Monitor severity level
        description: Monitor description
        ethics_config: Optional ethics configuration
        
    Returns:
        Monitor specification dictionary
    """
    spec = {
        "name": name,
        "type": "ethics",
        "severity": severity,
        "description": description,
        "params": {}
    }
    
    if ethics_config:
        spec["params"]["ethics_config"] = ethics_config
    
    return spec


# Legacy compatibility function
def applied_ethics_check(outcome: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Legacy compatibility wrapper for applied_ethics_check.
    
    This function maintains compatibility with existing code while
    delegating to the centralized EthicsJudge.
    
    Args:
        outcome: The outcome to evaluate
        
    Returns:
        Tuple of (passed, message)
    """
    try:
        ethics_judge = get_ethics_judge()
        passed, violations = ethics_judge.evaluate(outcome)
        
        if violations:
            # Find applied ethics violations specifically
            ethics_violations = [v for v in violations if v.rule_name == "applied_ethics"]
            if ethics_violations:
                return False, ethics_violations[0].rationale
            
            # If other violations, return general message
            return False, f"Ethics violations detected: {len(violations)} issues found"
        
        return True, None
        
    except Exception as e:
        logger.error(f"Legacy applied_ethics_check failed: {e}")
        return True, None  # Be permissive on error to maintain compatibility