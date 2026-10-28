"""Isolated session-removal, creation retry, mock-purchase, and End Day regression."""
from pathlib import Path
import tempfile

from src.session_store import CustomerSessionStore


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = CustomerSessionStore(Path(directory) / "isolated.sqlite3")
        customer = store.create_customer_once("button-request-1", "Review customer")
        assert store.create_customer_once("button-request-1", "Review customer") == customer
        assert len(store.list_customers()) == 1
        other = store.create_customer("Other customer")
        store.set_active_customer(customer)
        history = [{"role": "user", "content": "white t-shirt"}]
        store.save_conversation(customer, history, {"demo_conversation": {"favorite_colors": ["white"]}})
        cart = store.create_outfit(customer, "Selected Products")
        store.add_or_replace_outfit_item(cart, "VAR-WHITE-M", "t-shirt", "PROD-WHITE", 2)

        # Use the real ID for the isolated active-day membership check; report
        # persistence and membership now share the same durable day identity.
        day = store.active_business_day()["day_id"]
        purchase_items = [{"variant_id": "VAR-WHITE-M", "product_id": "PROD-WHITE", "quantity": 2, "unit_price_piasters": 45000}]
        request = store.record_mock_purchase(customer, cart, 90000, purchase_items, "confirm-token-1", day)
        assert store.record_mock_purchase(customer, cart, 90000, purchase_items, "confirm-token-1", day) == request
        assert store.mock_purchase_for_confirmation("confirm-token-1")["request_id"] == request

        assert store.remove_customer(customer, day)
        assert not store.remove_customer(customer, day), "repeat removal is idempotent"
        assert customer not in {row["customer_id"] for row in store.list_customers()}
        assert customer in {row["customer_id"] for row in store.removed_customers()}
        assert store.conversation(customer)[0] == history
        assert store.outfit_record(cart)["items"][0]["quantity"] == 2
        assert store.mock_purchase(request)["items"] == purchase_items
        assert other in {row["customer_id"] for row in store.list_customers()}

        snapshot, created = store.end_day(day)
        assert created
        assert snapshot["sales"]["confirmed_orders"] == 0 and snapshot["sales"]["revenue_egp"] == 0
        assert snapshot["demo_mock_purchases"]["confirmed_requests"] == 1
        assert snapshot["demo_mock_purchases"]["simulated_value_egp"] == 900
        assert snapshot["inventory"]["mock_requested_variant_quantities"] == {"VAR-WHITE-M": 2}
        assert snapshot["inventory"]["stock_deducted"] is False
        assert snapshot["customer_sessions"]["removed_from_active_list"] == 1
        repeat, made_again = store.end_day(day)
        assert not made_again and repeat == snapshot
        next_day, next_created = store.end_day("2030-02-04")
        assert next_created and next_day["demo_mock_purchases"]["confirmed_requests"] == 0
        assert store.restore_customer(customer)
        assert customer in {row["customer_id"] for row in store.list_customers()}
    print("PASS: isolated/idempotent customer creation, reversible removal, historical retention, exact mock purchase, quantity, and idempotent per-day reporting")


if __name__ == "__main__":
    main()
