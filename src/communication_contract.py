"""Versioned local communication contract for the KAYAN demo agents.

Events are validated and persisted locally. This module does not claim delivery
to remote CEO, Marketing, or Inventory services.
"""
from __future__ import annotations

from datetime import date
from typing import Any


SCHEMA_VERSION = 1
AGENTS = {"ceo", "sales", "marketing", "inventory"}
EVENT_REQUIRED_FIELDS: dict[str, set[str]] = {
    "ceo.task_assigned": {"task_id", "title", "assigned_date", "assigned_to"},
    "ceo.task_updated": {"task_id", "changes"},
    "sales.task_completed": {"task_id", "completed_at"},
    "sales.task_status_updated": {"task_id", "is_done"},
    "sales.customer_interest_recorded": {"customer_id", "customer_label", "product_id"},
    "sales.customer_feedback_recorded": {"customer_id", "customer_label", "feedback_type"},
    "sales.unmet_demand_recorded": {"customer_id", "customer_label", "criteria"},
    "sales.purchase_confirmed": {"order_id", "customer_id", "items", "total_piasters", "status"},
    "sales.business_day_closed": {"business_day_id", "report_versions"},
}


def validate_event(event_type: str, source_agent: str, target_agent: str | None,
                   business_day_id: str | None, payload: dict[str, Any],
                   schema_version: int = SCHEMA_VERSION) -> None:
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported communication event schema version: {schema_version}")
    if event_type not in EVENT_REQUIRED_FIELDS:
        raise ValueError(f"Unsupported communication event type: {event_type}")
    if source_agent not in AGENTS:
        raise ValueError(f"Unknown source agent: {source_agent}")
    if target_agent is not None and target_agent not in AGENTS:
        raise ValueError(f"Unknown target agent: {target_agent}")
    if not business_day_id:
        raise ValueError("Communication events require a business-day ID")
    if not isinstance(payload, dict):
        raise ValueError("Communication event payload must be an object")
    missing = EVENT_REQUIRED_FIELDS[event_type] - payload.keys()
    if missing:
        raise ValueError(f"{event_type} payload is missing: {', '.join(sorted(missing))}")
    if event_type == "ceo.task_assigned":
        date.fromisoformat(str(payload["assigned_date"]))
    if event_type == "sales.purchase_confirmed":
        if not isinstance(payload["items"], list) or int(payload["total_piasters"]) < 0:
            raise ValueError("Purchase event requires item rows and a non-negative integer total")
        if payload["status"] != "confirmed_mock_purchase":
            raise ValueError("Only persisted confirmed MOCK requests can be published as purchases")
