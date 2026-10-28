"""Live Gradio workflow test for clear, send, refresh and customer switching.

Start ``python app.py`` first. The test creates isolated demo customers and
never invokes End Day on the live business database.
"""
from pathlib import Path
import uuid
import sys

from gradio_test_support import isolated_gradio_server
from src.session_store import CustomerSessionStore


ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def reply(history):
    last = history[-1]
    content = last.get("content", "")
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content)
    return str(content)


def db_history(history):
    normalized = []
    for item in history:
        content = item.get("content", "")
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content)
        normalized.append({"role": item["role"], "content": content})
    return normalized


def main():
    with isolated_gradio_server() as (client, database):
        run(client, database)


def run(client, DB):
    suffix = uuid.uuid4().hex[:6]
    first = client.predict(f"Clear regression {suffix} A", api_name="/new_customer_cart")
    customer_1 = first[0]["value"] if isinstance(first[0], dict) else first[0]
    history = first[2]
    assert history == []

    before = client.predict("أنا بحب لون الأزرق هل عندكم حاجة زرقاء", api_name="/chat")
    history = before[0]
    assert len(history) == 2 and any(color in repr(history) for color in ("الأزرق", "أزرق", "Blue"))

    cleared = client.predict(api_name="/clear_chat")
    assert cleared[0] == []
    # Invoke the built-in Chatbot clear callback too; repeated clear is safe.
    built_in_clear = client.predict(api_name="/clear_chat_1")
    assert built_in_clear[0] == []
    store = CustomerSessionStore(DB)
    empty_active, reset_state = store.conversation(customer_1)
    assert empty_active == []
    assert reset_state["demo_conversation"]["active"] == {}
    assert reset_state["demo_conversation"]["favorite_colors"] == ["blue"]
    assert reset_state["demo_conversation"]["focused_product_id"] is None

    # Deliberately pass the stale browser value that existed before clear.
    after = client.predict("هل عندكم تيشيرتات", api_name="/chat")
    history_after = after[0]
    assert len(history_after) == 2, f"old messages were restored: {history_after!r}"
    assert all("بحب لون الأزرق" not in block.get("text", "") for block in history_after[0].get("content", []))

    active, state = store.conversation(customer_1)
    expected_active = db_history(history_after)
    assert [(row["role"], row["content"]) for row in active] == [(row["role"], row["content"]) for row in expected_active]
    assert state["demo_conversation"]["active"].get("categories") == ["t-shirt"]
    assert "color" not in state["demo_conversation"]["active"]
    assert state["demo_conversation"]["favorite_colors"] == ["blue"]
    assert state["demo_conversation"]["active"].get("categories") == ["t-shirt"]
    assert state["demo_conversation"]["focused_product_id"] in state["demo_conversation"]["displayed_product_ids"], "category overview focus must reference a displayed product"
    assert len(store.archived_conversations(customer_1)) == 1

    # The page's actual load callback must restore the selected customer and
    # only the active new thread after a browser refresh.
    restored = client.predict(api_name="/restore_active_customer")
    restored_customer = restored[0].get("value") if isinstance(restored[0], dict) else restored[0]
    assert restored_customer == customer_1
    assert restored[1] == history_after
    second = client.predict(f"Clear regression {suffix} B", api_name="/new_customer_cart")
    customer_2 = second[0]["value"] if isinstance(second[0], dict) else second[0]
    assert second[2] == []
    assert client.predict(customer_1, api_name="/switch_customer_cart")[0] == history_after
    assert store.conversation(customer_2)[0] == []

    # Preference remains available after clear without restoring old turns.
    memory = client.predict("طب عارف أنا بحب لون ايه", api_name="/chat")
    assert "blue" in reply(memory[0]).lower() or "أزرق" in reply(memory[0])
    assert all("بحب لون الأزرق" not in block.get("text", "") for block in memory[0][0].get("content", []))

    print("PASS: live Gradio explicit and built-in clear, stale client history rejection, new send, database refresh, customer switch isolation, and preference retention")


if __name__ == "__main__":
    main()
