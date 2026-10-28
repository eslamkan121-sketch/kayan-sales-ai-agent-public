"""Deterministic communication-contract, ownership, reporting and persistence tests."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from src.communication_contract import validate_event
from src.session_store import CustomerSessionStore


def main():
    with TemporaryDirectory(prefix="kayan-contract-") as directory:
        database=Path(directory)/"contract.sqlite3"
        store=CustomerSessionStore(database)
        day1=store.active_business_day();today=day1["business_date"]
        tomorrow=(date.fromisoformat(today)+timedelta(days=1)).isoformat()

        today_task=store.create_ceo_task("Prepare today's recommendations",today)
        tomorrow_task=store.create_ceo_task("Review tomorrow's follow-ups",tomorrow)
        assert today_task.startswith("task-") and tomorrow_task.startswith("task-") and today_task!=tomorrow_task
        assert store.update_ceo_task(today_task,"Prepare the verified recommendations",today)
        try:store.set_task_done_by_sales(today_task,True,source_agent="ceo")
        except PermissionError:pass
        else:raise AssertionError("CEO must not update Sales execution state")
        assert store.set_task_done_by_sales(today_task,True)
        assert store.set_task_done_by_sales(today_task,True) # idempotent no duplicate completion
        assert store.list_ceo_tasks(today)[0]["is_done"]==1
        assert store.list_ceo_tasks(tomorrow)[0]["is_done"]==0

        c1=store.create_customer("Customer 1");c2=store.create_customer("Customer 2");c3=store.create_customer("Customer 3")
        # One interest per customer/product/day; repeated delivery/retry returns
        # the original event ID and does not inflate counts.
        interest={"customer_id":c1,"customer_label":"Customer 1","product_id":"PVP-001","variant_id":"V-PVP-001-YLW-M"}
        interest_id=store.publish_communication_event("sales.customer_interest_recorded","sales","marketing",day1["day_id"],interest,c1,"interest:c1:p1")
        assert store.publish_communication_event("sales.customer_interest_recorded","sales","marketing",day1["day_id"],interest,c1,"interest:c1:p1")==interest_id
        positive={"customer_id":c1,"customer_label":"Customer 1","product_id":"PVP-001","variant_id":"V-PVP-001-YLW-M","feedback_type":"positive","reason":None}
        reject={"customer_id":c1,"customer_label":"Customer 1","product_id":"PVP-002","variant_id":"V-PVP-002-BLK-M","feedback_type":"rejected","reason":"too_expensive"}
        for payload in (positive,reject):
            key=f"feedback:{payload['product_id']}:{payload['feedback_type']}"
            store.publish_communication_event("sales.customer_feedback_recorded","sales","marketing",day1["day_id"],payload,c1,key)
        store.publish_communication_event("sales.unmet_demand_recorded","sales","marketing",day1["day_id"],
            {"customer_id":c2,"customer_label":"Customer 2","criteria":{"category":"t-shirt","color":"green"}},c2,"unmet:c2:green")

        # Adding to Selected Products is not a purchase event.
        outfit=store.create_outfit(c3,"Selected Products")
        store.add_or_replace_outfit_item(outfit,"V-PVP-003-NAV-32","jeans","PVP-003",quantity=2)
        assert store.communication_event_count(day1["day_id"],["sales.purchase_confirmed"])==0
        purchase_items=[{"product_id":"PVP-003","variant_id":"V-PVP-003-NAV-32","category":"jeans",
                         "color":"Navy","size":"32","currency":"EGP","quantity":2,
                         "unit_price_piasters":119000,"line_total_piasters":238000}]
        order=store.record_mock_purchase(c3,outfit,238000,purchase_items,"confirm-once",day1["day_id"])
        assert store.record_mock_purchase(c3,outfit,238000,purchase_items,"confirm-once",day1["day_id"])==order
        assert store.communication_event_count(day1["day_id"],["sales.purchase_confirmed"])==1
        purchase_event=store.consume_communication_events("inventory",day1["day_id"])
        assert len(purchase_event)==1 and purchase_event[0]["payload"]["order_id"]==order
        assert store.consume_communication_events("inventory",day1["day_id"])==[]
        # Purchase broadcast is independently consumable by Marketing.
        assert any(e["event_type"]=="sales.purchase_confirmed" for e in store.consume_communication_events("marketing",day1["day_id"]))

        # Contract validation rejects unknown event types and incomplete schema.
        try:validate_event("sales.customer_feedback_recorded","sales","marketing",day1["day_id"],{"customer_id":c1})
        except ValueError:pass
        else:raise AssertionError("invalid communication payload was accepted")

        snapshot,created=store.end_active_day()
        assert created and snapshot["reports"]["ceo"]["total_tasks"]==1
        assert snapshot["reports"]["ceo"]["completed_tasks"]==1 and snapshot["reports"]["ceo"]["not_done_tasks"]==0
        assert snapshot["reports"]["marketing"]["product_interest_counts"]=={"PVP-001":1}
        assert snapshot["reports"]["marketing"]["explicit_positive_feedback_counts"]=={"PVP-001":1}
        assert snapshot["reports"]["marketing"]["explicit_rejection_counts"]=={"PVP-002":1}
        assert snapshot["reports"]["marketing"]["rejection_reason_counts"]["PVP-002"]=={"too_expensive":1}
        assert snapshot["reports"]["marketing"]["unmet_demand_count"]==1
        assert snapshot["reports"]["marketing"]["confirmed_mock_purchases"][0]["customer_label"]=="Customer 3"
        assert snapshot["reports"]["inventory"]["variant_sales"][0]["quantity_sold"]==2
        assert snapshot["reports"]["inventory"]["mock_revenue_piasters"]==238000
        assert snapshot["reports"]["inventory"]["stock_deducted"] is False
        assert snapshot["reports"]["inventory"]["gross_profit_piasters"] is None
        assert store.list_ceo_tasks(tomorrow)[0]["is_done"]==0
        assert store.communication_report(day1["day_id"],"marketing")["schema_version"]==1
        same_snapshot,not_created=store.end_active_day();assert not not_created and same_snapshot==snapshot

        day2,started=store.start_new_day(tomorrow);assert started and day2["day_number"]==2
        assert store.list_customers()==[]
        new_customer=store.create_customer("Fresh customer")
        reopened=CustomerSessionStore(database)
        assert reopened.active_business_day()["day_id"]==day2["day_id"]
        assert reopened.list_customers()[0]["customer_id"]==new_customer
        assert reopened.communication_event_count(day1["day_id"],["sales.purchase_confirmed"])==1
        assert reopened.communication_report(day1["day_id"],"inventory")["report"]["mock_revenue_piasters"]==238000
        assert {r["task_id"] for r in reopened.list_ceo_tasks()}=={today_task,tomorrow_task}

    print("PASS: CEO task ownership/dates/status, marketing customer attribution and event dedupe, unmet demand, confirmed MOCK order idempotency, inventory totals/no stock deduction, day reports, rollover, and restart persistence")


if __name__=="__main__":main()
