"""Regression test for isolated customer sessions and idempotent end-of-day data."""
from __future__ import annotations

from pathlib import Path
import tempfile

from src.session_store import CustomerSessionStore


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = CustomerSessionStore(Path(directory) / "customers.sqlite3")
        customer_1 = store.create_customer()
        customer_2 = store.create_customer("Mona")
        assert store.customer(customer_1)["display_name"] == "Customer 1"
        assert store.customer(customer_2)["display_name"] == "Mona"

        history_1 = [{"role": "user", "content": "red shirt"}]
        state_1 = {"favorite_colors": ["Red"], "last_product": "KYN-01-01"}
        store.save_conversation(customer_1, history_1, state_1)
        store.save_conversation(customer_2, [{"role": "user", "content": "blue dress"}], {"favorite_colors": ["Blue"]})
        assert store.conversation(customer_1) == (history_1, state_1)
        assert store.conversation(customer_2)[1]["favorite_colors"] == ["Blue"]

        current_day = store.active_business_day()["day_id"]
        store.set_active_customer(customer_1)
        outfit = store.create_outfit(customer_1, "Shopping cart")
        store.add_or_replace_outfit_item(outfit, "VARIANT-DEMO-1", "t-shirt", "PRODUCT-DEMO-1", 2)
        purchase = store.record_mock_purchase(customer_1, outfit, 12500, ["VARIANT-DEMO-1"],
                                              confirmation_key="archive-regression", business_day=current_day)
        assert store.mock_purchase(purchase)
        assert store.is_current_day_customer(customer_1)
        assert store.remove_customer(customer_1, current_day)
        assert not store.is_current_day_customer(customer_1)
        assert customer_1 not in {row["customer_id"] for row in store.list_customers()}
        assert store.customer(customer_1)["removed"] == 1
        assert store.conversation(customer_1) == (history_1, state_1)
        assert store.outfit_record(outfit)["items"][0]["variant_id"] == "VARIANT-DEMO-1"
        assert store.mock_purchase(purchase)["total_piasters"] == 12500
        assert not store.remove_customer(customer_1, current_day), "repeated archive must be an idempotent no-op"
        with store._connect() as conn:
            archived_events = conn.execute(
                "SELECT COUNT(*) FROM customer_activity WHERE customer_id=? AND event_type='customer_removed'",
                (customer_1,),
            ).fetchone()[0]
        assert archived_events == 1
        assert store.is_current_day_customer(customer_2)

        day = "2030-01-02"
        store.record_activity(customer_2, day, "customer_message", {})
        store.record_activity(customer_2, day, "ranked_catalog_inquiry", {"product_ids": ["KYN-02-01", "KYN-02-01"]})
        store.end_customer(customer_1, day)
        assert not store.is_active(customer_1), "completed customer must reject new messages in the UI"
        assert store.conversation(customer_1) == (history_1, state_1), "ending retains history and state"
        assert store.is_active(customer_2), "another active customer remains isolated"

        snapshot, created = store.end_day(day)
        assert created and store.day_is_closed(day)
        assert snapshot["sales"]["confirmed_orders"] == 0
        assert snapshot["sales"]["revenue_egp"] == 0
        assert snapshot["inventory"]["verified_purchased_variants"] == []
        assert snapshot["marketing"]["ranked_product_interest"] == {"KYN-02-01": 2}
        repeat, created_again = store.end_day(day)
        assert not created_again and repeat == snapshot, "repeated End Day must not duplicate or mutate the snapshot"

        next_day, next_created = store.end_day("2030-01-03")
        assert next_created and next_day["business_day"] == "2030-01-03", "the next business day remains available"
    print("PASS: customer isolation, completed-session retention, truthful daily reports, idempotent End Day, and next-day continuity")


if __name__ == "__main__":
    main()
