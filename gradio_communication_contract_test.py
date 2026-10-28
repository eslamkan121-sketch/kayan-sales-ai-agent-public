"""Live Gradio/API coverage for the local Communication Contract dashboards."""
from datetime import date, timedelta
import json

from gradio_test_support import isolated_gradio_server
from src.demo_catalog import load_demo_catalog
from src.session_store import CustomerSessionStore


def main():
    today = date.today().isoformat()
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    with isolated_gradio_server() as (client, database):
        config_text = json.dumps(client.config, ensure_ascii=False)
        for label in ("Communication Contract", "CEO", "Marketing", "Inventory"):
            assert label in config_text, f"Missing live Gradio tab: {label}"

        client.predict("Verify customer greeting", today, api_name="/create_ceo_task_ui")
        store = CustomerSessionStore(database)
        today_tasks = store.list_ceo_tasks(today)
        assert len(today_tasks) == 1
        task = today_tasks[0]
        client.predict(task["task_id"], "Verify customer greeting and product details", today,
                       api_name="/update_ceo_task_ui")
        client.predict(task["task_id"], True, today, api_name="/sales_task_status_ui")
        assert store.list_ceo_tasks(today)[0]["title"] == "Verify customer greeting and product details"
        assert bool(store.list_ceo_tasks(today)[0]["is_done"])
        client.predict("Verify catalog request", tomorrow, api_name="/create_ceo_task_ui")
        future_tasks = store.list_ceo_tasks(tomorrow)
        assert len(future_tasks) == 1
        assert not bool(store.list_ceo_tasks(tomorrow)[0]["is_done"])

        created = client.predict("Contract E2E Customer", api_name="/new_customer_cart")
        customer_id = created[0]["value"]
        assert customer_id
        response = client.predict("هل عندكم تيشيرت أبيض", api_name="/chat")
        assert response and response[0]
        marketing = store.marketing_contract_data(today, today)
        assert marketing["conversation_customer_count"] == 1
        assert marketing["interest_count"] >= 1
        assert any(row.get("customer_id") == customer_id and row.get("customer_label") == "Contract E2E Customer"
                   for row in marketing["customer_details"])

        # Seed a verified confirmed-MOCK record in the isolated DB so the
        # dashboard callback can be tested with non-empty exact variant data.
        catalog = load_demo_catalog(
            "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
            "catalog_preview_review/preview_assets")
        state = store.conversation(customer_id)[1]["demo_conversation"]
        variant = catalog.get(state["displayed_variant_ids"][0])
        assert variant and variant.in_stock
        outfit = store.list_outfits(customer_id)[0]
        store.add_or_replace_outfit_item(outfit["outfit_id"], variant.variant_id, variant.category, variant.product_id)
        day_id = store.active_business_day()["day_id"]
        item = {"variant_id": variant.variant_id, "product_id": variant.product_id, "category": variant.category,
                "color": variant.color, "size": variant.size, "currency": variant.currency, "quantity": 1,
                "unit_price_piasters": variant.price_piasters, "line_total_piasters": variant.price_piasters}
        store.record_mock_purchase(customer_id, outfit["outfit_id"], variant.price_piasters, [item],
                                   "contract-test-confirmation", day_id)
        inventory = store.inventory_contract_data(today, today)
        assert inventory["confirmed_mock_orders"] == 1
        assert inventory["mock_revenue_piasters"] > 0
        assert inventory["variant_sales"] and inventory["variant_sales"][0]["quantity_sold"] == 1
        marketing = store.marketing_contract_data(today, today)
        assert marketing["confirmed_mock_purchase_count"] == 1

        client.predict(today, today, api_name="/marketing_refresh")
        inventory_ui = client.predict(today, today, api_name="/inventory_refresh")
        assert "Confirmed MOCK orders:** 1" in str(inventory_ui[0])
        assert "Communication Contract" in config_text
    print("PASS: live Gradio nested contract tabs, dated CEO task create/edit/Done ownership, customer-attributed marketing interest and confirmed purchase, exact inventory totals, and Inventory UI rendering")


if __name__ == "__main__":
    main()
