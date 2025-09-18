"""Cost recording and aggregation system."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .pricing import CostRecord, PricingEngine

logger = logging.getLogger("agentnet.cost")


class CostRecorder:
    """Records and persists cost events."""

    def __init__(self, storage_dir: str = "cost_logs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.pricing_engine = PricingEngine()
        logger.info(f"CostRecorder initialized with storage_dir: {storage_dir}")

    def record_inference_cost(
        self,
        provider: str,
        model: str,
        result: Dict[str, Any],
        agent_name: str,
        task_id: str,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostRecord:
        """Record cost for an inference operation."""

        # Extract token counts from result
        tokens_input = result.get("tokens_input", 0)
        tokens_output = result.get("tokens_output", 0)

        # If tokens not provided, estimate from content
        if tokens_input == 0 and tokens_output == 0:
            content = result.get("content", "")
            if content:
                # Rough estimation: ~4 characters per token
                estimated_tokens = len(str(content)) // 4
                tokens_output = estimated_tokens
                # Assume prompt was similar length
                tokens_input = estimated_tokens

        # Calculate cost
        cost_record = self.pricing_engine.calculate_cost(
            provider=provider,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            agent_name=agent_name,
            task_id=task_id,
            session_id=session_id,
            tenant_id=tenant_id,
            metadata=metadata,
        )

        # Persist the record
        self._persist_record(cost_record)

        logger.debug(
            f"Recorded cost: {cost_record.total_cost:.6f} for {agent_name}/{task_id}"
        )
        return cost_record

    def _persist_record(self, record: CostRecord):
        """Persist cost record to storage."""
        # Create filename based on date
        date_str = record.timestamp.strftime("%Y-%m-%d")
        filename = self.storage_dir / f"costs_{date_str}.jsonl"

        # Convert record to dict
        record_dict = asdict(record)
        record_dict["timestamp"] = record.timestamp.isoformat()

        # Append to file
        with open(filename, "a") as f:
            f.write(json.dumps(record_dict) + "\n")

    def get_records(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[CostRecord]:
        """Retrieve cost records with optional filtering."""
        records = []

        # Determine date range
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()

        # Iterate through relevant files
        current_date = start_date.date()
        while current_date <= end_date.date():
            filename = (
                self.storage_dir / f"costs_{current_date.strftime('%Y-%m-%d')}.jsonl"
            )
            if filename.exists():
                records.extend(self._load_records_from_file(filename))
            current_date += timedelta(days=1)

        # Apply filters
        filtered_records = []
        for record in records:
            if tenant_id is not None and record.tenant_id != tenant_id:
                continue
            if agent_name is not None and record.agent_name != agent_name:
                continue
            if session_id is not None and record.session_id != session_id:
                continue
            if record.timestamp < start_date or record.timestamp > end_date:
                continue
            filtered_records.append(record)

        return filtered_records

    def _load_records_from_file(self, filename: Path) -> List[CostRecord]:
        """Load cost records from a JSONL file."""
        records = []
        try:
            with open(filename, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                    records.append(CostRecord(**data))
        except Exception as e:
            logger.error(f"Error loading records from {filename}: {e}")
        return records


class CostAggregator:
    """Aggregates and analyzes cost data."""

    def __init__(self, recorder: CostRecorder):
        self.recorder = recorder

    def get_cost_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get cost summary for a time period."""
        records = self.recorder.get_records(
            start_date=start_date, end_date=end_date, tenant_id=tenant_id
        )

        if not records:
            return {
                "total_cost": 0.0,
                "total_tokens_input": 0,
                "total_tokens_output": 0,
                "provider_breakdown": {},
                "agent_breakdown": {},
                "session_breakdown": {},
                "record_count": 0,
            }

        total_cost = sum(r.total_cost for r in records)
        total_tokens_input = sum(r.tokens_input for r in records)
        total_tokens_output = sum(r.tokens_output for r in records)

        # Provider breakdown
        provider_costs = defaultdict(float)
        provider_tokens = defaultdict(lambda: {"input": 0, "output": 0})
        for record in records:
            provider_costs[record.provider] += record.total_cost
            provider_tokens[record.provider]["input"] += record.tokens_input
            provider_tokens[record.provider]["output"] += record.tokens_output

        # Agent breakdown
        agent_costs = defaultdict(float)
        agent_tokens = defaultdict(lambda: {"input": 0, "output": 0})
        for record in records:
            agent_costs[record.agent_name] += record.total_cost
            agent_tokens[record.agent_name]["input"] += record.tokens_input
            agent_tokens[record.agent_name]["output"] += record.tokens_output

        # Session breakdown
        session_costs = defaultdict(float)
        for record in records:
            if record.session_id:
                session_costs[record.session_id] += record.total_cost

        return {
            "total_cost": total_cost,
            "total_tokens_input": total_tokens_input,
            "total_tokens_output": total_tokens_output,
            "provider_breakdown": {
                provider: {
                    "cost": cost,
                    "tokens_input": provider_tokens[provider]["input"],
                    "tokens_output": provider_tokens[provider]["output"],
                }
                for provider, cost in provider_costs.items()
            },
            "agent_breakdown": {
                agent: {
                    "cost": cost,
                    "tokens_input": agent_tokens[agent]["input"],
                    "tokens_output": agent_tokens[agent]["output"],
                }
                for agent, cost in agent_costs.items()
            },
            "session_breakdown": dict(session_costs),
            "record_count": len(records),
        }

    def get_cost_trends(
        self, days: int = 7, tenant_id: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get daily cost trends."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        records = self.recorder.get_records(
            start_date=start_date, end_date=end_date, tenant_id=tenant_id
        )

        # Group by day
        daily_costs = defaultdict(
            lambda: {
                "date": None,
                "total_cost": 0.0,
                "total_tokens": 0,
                "record_count": 0,
                "providers": defaultdict(float),
            }
        )

        for record in records:
            date_key = record.timestamp.date().isoformat()
            daily_costs[date_key]["date"] = date_key
            daily_costs[date_key]["total_cost"] += record.total_cost
            daily_costs[date_key]["total_tokens"] += (
                record.tokens_input + record.tokens_output
            )
            daily_costs[date_key]["record_count"] += 1
            daily_costs[date_key]["providers"][record.provider] += record.total_cost

        # Convert to list and sort by date
        trends = list(daily_costs.values())
        trends.sort(key=lambda x: x["date"])

        return {"daily_trends": trends}

    def get_top_cost_agents(
        self,
        limit: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get top cost-generating agents."""
        records = self.recorder.get_records(
            start_date=start_date, end_date=end_date, tenant_id=tenant_id
        )

        agent_stats = defaultdict(
            lambda: {
                "agent_name": "",
                "total_cost": 0.0,
                "total_tokens": 0,
                "inference_count": 0,
                "avg_cost_per_inference": 0.0,
            }
        )

        for record in records:
            stats = agent_stats[record.agent_name]
            stats["agent_name"] = record.agent_name
            stats["total_cost"] += record.total_cost
            stats["total_tokens"] += record.tokens_input + record.tokens_output
            stats["inference_count"] += 1

        # Calculate averages
        for stats in agent_stats.values():
            if stats["inference_count"] > 0:
                stats["avg_cost_per_inference"] = (
                    stats["total_cost"] / stats["inference_count"]
                )

        # Sort by total cost and limit
        top_agents = sorted(
            agent_stats.values(), key=lambda x: x["total_cost"], reverse=True
        )
        return top_agents[:limit]


class TenantCostTracker:
    """Tracks costs per tenant with budget enforcement."""

    def __init__(self, recorder: CostRecorder):
        self.recorder = recorder
        self.tenant_budgets: Dict[str, float] = {}  # tenant_id -> monthly budget
        self.tenant_alerts: Dict[str, Dict[str, float]] = (
            {}
        )  # tenant_id -> {threshold -> alert_level}

    def set_tenant_budget(self, tenant_id: str, monthly_budget: float):
        """Set monthly budget for a tenant."""
        self.tenant_budgets[tenant_id] = monthly_budget
        logger.info(f"Set monthly budget for tenant {tenant_id}: ${monthly_budget:.2f}")

    def set_tenant_alerts(self, tenant_id: str, alerts: Dict[str, float]):
        """Set cost alert thresholds for a tenant.

        Args:
            tenant_id: Tenant identifier
            alerts: Dictionary mapping threshold percentages to alert levels
                   e.g., {"warning": 0.75, "critical": 0.90}
        """
        self.tenant_alerts[tenant_id] = alerts
        logger.info(f"Set cost alerts for tenant {tenant_id}: {alerts}")

    def check_tenant_budget(self, tenant_id: str) -> Dict[str, Any]:
        """Check current budget status for a tenant."""
        if tenant_id not in self.tenant_budgets:
            return {"status": "no_budget", "message": "No budget set for tenant"}

        budget = self.tenant_budgets[tenant_id]

        # Get current month costs
        now = datetime.now()
        start_of_month = datetime(now.year, now.month, 1)

        records = self.recorder.get_records(
            start_date=start_of_month, tenant_id=tenant_id
        )

        current_spend = sum(r.total_cost for r in records)
        usage_percentage = (current_spend / budget) if budget > 0 else 0
        remaining_budget = budget - current_spend

        # Check alerts
        alert_level = None
        if tenant_id in self.tenant_alerts:
            for level, threshold in self.tenant_alerts[tenant_id].items():
                if usage_percentage >= threshold:
                    alert_level = level

        status = "ok"
        if usage_percentage >= 1.0:
            status = "over_budget"
        elif alert_level:
            status = f"alert_{alert_level}"

        return {
            "status": status,
            "budget": budget,
            "current_spend": current_spend,
            "remaining_budget": remaining_budget,
            "usage_percentage": usage_percentage,
            "alert_level": alert_level,
            "record_count": len(records),
        }

    def get_all_tenant_status(self) -> Dict[str, Dict[str, Any]]:
        """Get budget status for all tenants."""
        statuses = {}
        for tenant_id in self.tenant_budgets:
            statuses[tenant_id] = self.check_tenant_budget(tenant_id)
        return statuses
