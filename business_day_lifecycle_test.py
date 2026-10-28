"""Sequential business-day lifecycle, rollover, persistence, and report regression."""
from pathlib import Path
import tempfile

from src.session_store import CustomerSessionStore


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "days.sqlite3"
        store = CustomerSessionStore(db)
        day1 = store.active_business_day()
        assert day1 and day1["day_number"] == 1 and day1["status"] == "active"
        assert store.start_new_day("2030-01-02")[1] is False, "cannot roll over an unclosed active day"
        c1 = store.create_customer("Day 1 customer 1")
        c2 = store.create_customer("Day 1 customer 2")
        store.set_active_customer(c1)
        history = [{"role": "user", "content": "white shirt"}, {"role": "assistant", "content": "catalog response"}]
        store.save_conversation(c1, history, {"demo_conversation": {"favorite_colors": ["White"]}})
        store.record_activity(c1, day1["day_id"], "customer_message", {})
        store.record_activity(c1, day1["day_id"], "ranked_catalog_inquiry", {"product_ids": ["P1"], "categories": ["shirt"], "colors": ["White"]})
        cart = store.create_outfit(c1, "Selected Products")
        item = [{"variant_id": "V1-WHT-M", "product_id": "P1", "quantity": 2, "unit_price_piasters": 45000}]
        request_id = store.record_mock_purchase(c1, cart, 90000, item, "day1-purchase", day1["day_id"])

        closed, created = store.end_active_day()
        assert created and closed["day_number"] == 1 and closed["status"] == "closed"
        assert set(closed["customer_session_ids"]) == {c1, c2}
        assert closed["events"]["customer_messages"] == 1
        assert closed["reports"]["marketing"]["category_interest"] == {"shirt": 1}
        assert closed["reports"]["marketing"]["color_interest"] == {"White": 1}
        assert closed["demo_mock_purchases"]["confirmed_requests"] == 1
        assert closed["demo_mock_purchases"]["simulated_value_piasters"] == 90000
        assert closed["sales"]["revenue_piasters"] == 0
        assert closed["reports"]["inventory"]["stock_deducted"] is False
        assert store.list_customers() == [] and store.active_customer_id() is None
        repeated, made_again = store.end_active_day()
        assert not made_again and repeated == closed, "End Day must be immutable/idempotent"

        day2, started = store.start_new_day("2030-01-03")
        assert started and day2["day_number"] == 2 and day2["status"] == "active"
        same_day, duplicate = store.start_new_day("2030-01-03")
        assert not duplicate and same_day["day_id"] == day2["day_id"]
        assert len(store.list_business_days()) == 2
        assert store.list_customers() == [] and store.active_customer_id() is None
        assert store.business_day_snapshot(day2["day_id"]) == {"reports": {}}
        c3 = store.create_customer("Day 2 customer")
        assert {r["customer_id"] for r in store.list_customers()} == {c3}
        assert not store.is_current_day_customer(c1) and store.is_current_day_customer(c3)
        assert store.conversation(c1)[0] == history
        assert store.mock_purchase(request_id)["items"] == item
        store.save_conversation(c3, [{"role": "user", "content": "new chat"}], {})

        restarted = CustomerSessionStore(db)
        assert restarted.active_business_day()["day_id"] == day2["day_id"]
        assert restarted.list_customers()[0]["customer_id"] == c3
        assert restarted.conversation(c1)[0] == history
        assert restarted.mock_purchase(request_id)["business_day"] == day1["day_id"]
        assert restarted.business_day_snapshot(day1["day_id"])["reports"]["ceo"]["real_sales_piasters"] == 0
        assert [d["day_number"] for d in restarted.list_business_days()] == [2, 1]
    print("PASS: Day 1 close, immutable reports, Day 2 rollover, no duplicate day, isolated sessions, and restart persistence")


if __name__ == "__main__":
    main()
