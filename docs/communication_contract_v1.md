# KAYAN local Communication Contract, version 1

## Scope and delivery

The contract is a local, SQLite-backed demo adapter. It does not connect to a remote CEO, Marketing, or Inventory service and does not claim that those agents received a network message. Events are append-only, validated against schema version 1, and delivered to local consumer ledgers idempotently. Every event is associated with a stable business-day ID. Event creation and consumer delivery are separate operations.

Common event envelope:

```json
{
  "schema_version": 1,
  "event_id": "event-<stable-or-generated-id>",
  "event_type": "sales.customer_interest_recorded",
  "source_agent": "sales",
  "target_agent": "marketing",
  "business_day_id": "day-<id>",
  "customer_id": "cust-<id-or-null>",
  "interaction_key": "<idempotency-key>",
  "timestamp": "<UTC timestamp>",
  "payload": {}
}
```

## Event types and minimum payloads

| Event | Source → target | Required payload fields | Meaning |
| --- | --- | --- | --- |
| `ceo.task_assigned` | CEO → Sales | `task_id`, `title`, `assigned_date`, `assigned_to` | A dated task assigned to Sales. |
| `ceo.task_updated` | CEO → Sales | `task_id`, `changes` | CEO-owned task description/date changes. |
| `sales.task_completed` | Sales → CEO | `task_id`, `completed_at` | Sales verified completion. |
| `sales.task_status_updated` | Sales → CEO | `task_id`, `is_done` | Sales changed the two-state Done/Not Done status. |
| `sales.customer_interest_recorded` | Sales → Marketing | `customer_id`, `customer_label`, `product_id` | A deduplicated product interest from a displayed catalog result. |
| `sales.customer_feedback_recorded` | Sales → Marketing | `customer_id`, `customer_label`, `feedback_type` | Explicit positive, accepted, or rejected feedback; rejection may include a reason. |
| `sales.unmet_demand_recorded` | Sales → Marketing | `customer_id`, `customer_label`, `criteria` | A constrained request returned no catalog match. |
| `sales.purchase_confirmed` | Sales → Inventory (local ledger) | `order_id`, `customer_id`, `items`, `total_piasters`, `status` | A successfully persisted, explicitly confirmed MOCK purchase; status must be `confirmed_mock_purchase`. |
| `sales.business_day_closed` | Sales → local reports | `business_day_id`, `report_versions` | The day was closed and immutable report snapshots were created. |

Purchase line items retain exact Product ID, Variant ID, color, size, quantity, currency, unit price and line total in integer minor units. A cart addition or product view does not emit a purchase event. The event records `transaction_type: MOCK` and `stock_deducted: false`. Workbook prices and stock remain simulated demo values; cost, profit, authoritative stock, and real payment are unavailable.

## Ownership and reports

- CEO owns task creation, edits, and assigned date. Sales can change task completion only after verifying the work. Task states are only Done and Not Done; End Day does not auto-complete tasks.
- Marketing receives customer-attributed product interest, explicit feedback, unmet demand, and separately identified confirmed MOCK purchases. Events do not contain full chat transcripts.
- Inventory receives only persisted confirmed MOCK purchase lines and reconciled day reports. Demo stock is informational and is never deducted.
- End Day writes one versioned snapshot per agent (`ceo`, `marketing`, `inventory`, `sales`) and is idempotent. Start New Day preserves historical records and future-dated CEO tasks.

## Storage and idempotency

The schema is added using non-destructive `CREATE TABLE IF NOT EXISTS` migrations. Core tables are `communication_events`, `communication_deliveries`, `ceo_tasks`, and `communication_reports`. Event interaction keys and report `(business_day_id, agent, schema_version)` keys prevent duplicate creation. Delivery is unique per event/consumer. Tests use temporary SQLite databases, never the operator's active demo database.
