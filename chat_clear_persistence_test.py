"""SQLite regression for explicit chat clear/archive semantics."""
from pathlib import Path
import json
import sqlite3
import tempfile

from src.session_store import CustomerSessionStore


def main():
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "sessions.sqlite3"
        store = CustomerSessionStore(database)
        c1, c2, c3 = store.create_customer(), store.create_customer(), store.create_customer()
        assert store.set_active_customer(c1)
        assert store.active_customer_id() == c1
        old_chat = [
            {"role": "user", "content": "blue shirt"},
            {"role": "assistant", "content": "Blue demo shirt"},
        ]
        original_state = {"demo_conversation": {
            "favorite_colors": ["blue", "yellow"], "style_preferences": ["classic"],
            "active": {"color": "blue", "category": "t-shirt"},
            "displayed_product_ids": ["PVP-007"], "displayed_variant_ids": ["V-PVP-007-WHI-M"],
            "focused_product_id": "PVP-007", "focused_variant_id": "V-PVP-007-WHI-M",
            "selected_color": "White", "selected_size": "M", "pending_action": {"kind": "add"},
        }}
        store.save_conversation(c1, old_chat, original_state)
        store.save_conversation(c2, [{"role": "user", "content": "second customer"}], {"demo_conversation": {"favorite_colors": ["red"]}})
        store.save_conversation(c3, [{"role": "user", "content": "third customer"}], {})

        # These operational records must survive a chat reset.
        cart = store.create_outfit(c1, "Shopping cart")
        store.add_or_replace_outfit_item(cart, "V-PVP-007-WHI-M", "t-shirt", "PVP-007")
        purchase = store.record_mock_purchase(c1, cart, 44900, ["V-PVP-007-WHI-M"])
        day = "2030-05-06"
        store.record_activity(c1, day, "customer_message", {})
        store.record_activity(c1, day, "ranked_catalog_inquiry", {"product_ids": ["V-PVP-007-WHI-M"]})

        archived, count = store.clear_conversation(c1)
        assert archived and count == 2
        assert store.conversation(c1)[0] == []
        cleared = store.conversation(c1)[1]["demo_conversation"]
        assert cleared["favorite_colors"] == ["blue", "yellow"]
        assert cleared["style_preferences"] == ["classic"]
        assert cleared["active"] == {} and cleared["focused_product_id"] is None and cleared["pending_action"] == {}
        assert len(store.archived_conversations(c1)) == 1
        assert store.archived_conversations(c1)[0]["conversation"] == old_chat
        assert store.active_customer_id() == c1, "clearing chat keeps the active customer identity"

        # A newly sent turn persists by appending to SQLite's empty active chat.
        new_chat = [{"role": "user", "content": "هل عندكم تيشيرتات"}, {"role": "assistant", "content": "New answer"}]
        store.save_conversation(c1, new_chat, cleared)
        assert store.conversation(c1)[0] == new_chat
        assert all("blue shirt" not in json.dumps(row["conversation"], ensure_ascii=False) for row in store.archived_conversations(c1)[1:])

        # Reopening the store simulates a UI refresh/server reconstruction.
        refreshed = CustomerSessionStore(database)
        assert refreshed.conversation(c1)[0] == new_chat
        assert refreshed.conversation(c2)[0][0]["content"] == "second customer"
        assert refreshed.conversation(c3)[0][0]["content"] == "third customer"
        assert len(refreshed.archived_conversations(c1)) == 1

        # Repeated clear of an already empty chat is a no-op: no empty archive,
        # no new customer, and no duplicate activity event.
        assert refreshed.clear_conversation(c1) == (True, 2)
        assert refreshed.clear_conversation(c1) == (False, 0)
        assert len(refreshed.list_customers(include_ended=True)) == 3
        assert len(refreshed.archived_conversations(c1)) == 2  # new_chat was archived once
        assert refreshed.conversation(c1)[0] == []
        assert refreshed.outfit_record(cart)["items"][0]["variant_id"] == "V-PVP-007-WHI-M"
        connection = sqlite3.connect(database)
        try:
            assert connection.execute("SELECT COUNT(*) FROM demo_mock_purchase_requests WHERE request_id=?", (purchase,)).fetchone()[0] == 1
            assert connection.execute("SELECT COUNT(*) FROM customer_activity WHERE business_day=?", (day,)).fetchone()[0] == 2
        finally:
            connection.close()

        # Ending the customer keeps the active/archived records but blocks chat
        # restoration as an active customer. End Day still counts business events.
        refreshed.end_customer(c1, day)
        assert not refreshed.is_active(c1)
        assert refreshed.conversation(c1)[0] == []
        snapshot, created = refreshed.end_day(day)
        assert created
        assert snapshot["customer_sessions"]["customer_messages_today"] == 1
        assert snapshot["marketing"]["ranked_catalog_inquiries"] == 1
        print("PASS: clear/archive, stale-history isolation, refresh persistence, customer switching/isolation, cart and mock-purchase retention, idempotency, End Customer, and End Day counts")


if __name__ == "__main__":
    main()
