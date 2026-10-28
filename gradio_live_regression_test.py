"""Live Gradio regression. Start ``python app.py`` first, then run this file."""
from gradio_test_support import isolated_gradio_server
from pathlib import Path
from src.demo_catalog import load_demo_catalog
from src.session_store import CustomerSessionStore
from src.presentation import format_egp


ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
                            ROOT / "catalog_preview_review/preview_assets")


def text(history):
    return "".join(block.get("text", "") for block in history[-1]["content"])


def main():
    with isolated_gradio_server() as (client, database):
        run(client, database)


def run(client, database):
    created = client.predict("Live Regression Customer", api_name="/new_customer_cart")
    customer_id = created[0]["value"]

    first = client.predict("أنا بحب لون الأبيض اديني ليست خمس حجات عندكم لونها ابيض", api_name="/chat")
    first_reply = repr(first[0])
    saved_state = CustomerSessionStore(database).conversation(customer_id)[1]["demo_conversation"]
    assert "الاختيارات" in first_reply and len(saved_state["displayed_product_ids"]) == 5
    first_variant_id = saved_state["displayed_variant_ids"][0]
    first_variant = CATALOG.get(first_variant_id)
    assert first_variant and "white" in first_variant.color.lower()

    price = client.predict("عاجبني اول حاجه بكام ده", api_name="/chat")
    assert "جنيه" in repr(price[0]) and format_egp(first_variant.price_egp, True) in repr(price[0])

    available = [v for v in CATALOG.variants if v.product_id == first_variant.product_id and v.color == first_variant.color and v.in_stock]
    chosen = next((v for v in available if v.size.upper() == "M"), available[0])
    added = client.predict(f"ضيفه للسلة مقاس {chosen.size}", api_name="/chat")
    assert "تمت إضافة" in repr(added[0])
    stored = CustomerSessionStore(database)
    outfit = stored.list_outfits(customer_id)[0]
    assert any(row["variant_id"] == chosen.variant_id for row in stored.outfit_record(outfit["outfit_id"])["items"])

    # A pending shoe-size choice accepts a numeric-only answer while retaining
    # Arabic and resolving the exact workbook variant.
    shoe = client.predict("عايز أشتري جزمة", api_name="/chat")
    pending = client.predict("ضيفه للسلة", api_name="/chat")
    assert "39" in repr(pending[0])
    selected = client.predict("39", api_name="/chat")
    assert "تمت إضافة" in repr(selected[0]) and "I’ll add" not in repr(selected[0])
    cart_items = stored.outfit_record(outfit["outfit_id"])["items"]
    assert any(CATALOG.get(row["variant_id"]).size == "39" for row in cart_items)
    print("PASS: live Gradio reference, variant colour/size continuity, and Selected Products total")


if __name__ == "__main__":
    main()
