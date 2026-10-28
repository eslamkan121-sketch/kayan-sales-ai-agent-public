"""Application-render regression for localized cards, cart, and mock receipt."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
_db_dir = tempfile.TemporaryDirectory(prefix="kayan-presentation-")
os.environ["KAYAN_SESSION_DB"] = str(Path(_db_dir.name) / "sessions.sqlite3")

import app  # noqa: E402  (set the isolated database before app initialization)


def main():
    ui_config = app.build_ui().get_config_file()
    galleries = [component for component in ui_config["components"] if component.get("type") == "gallery"]
    product_gallery = next(component for component in galleries if component.get("props", {}).get("label", "").startswith("Matching product images"))
    catalog_gallery = next(component for component in galleries if component.get("props", {}).get("label", "").startswith("Catalog product images"))
    assert "value" in product_gallery["props"]
    assert len(catalog_gallery["props"].get("value") or []) >= 20
    assert all(v.image_path and v.image_path.is_file() for v in app.DEMO_CATALOG.variants)
    assert all("image_provenance" in app.DEMO_CATALOG.get(v.variant_id).__dict__ for v in app.DEMO_CATALOG.variants)

    # Dynamic chat cards show only requested fields and bind their image bytes
    # to the precise selected color variant.
    white_polo = app.DEMO_CATALOG.get("V-PVP-015-WHI-S")
    assert white_polo and white_polo.image_path and white_polo.image_path.is_file()
    image_only = app.render_history([{"role": "assistant", "content": "دي صورتها.",
        "_cards": {"variant_ids": [white_polo.variant_id], "fields": ["name", "image"], "language": "ar"}}])
    card_content = image_only[0]["content"]
    assert len(card_content) == 2 and card_content[0] == "دي صورتها."
    image_html = card_content[1].value
    assert app.image_data(white_polo.image_path) in image_html
    assert "تيشيرت بولو كلاسيك" in image_html
    assert "Illustrative DEMO image" not in image_html and "صورة توضيحية" in image_html
    assert "جنيه" not in image_html and "المقاسات:" not in image_html
    assert white_polo.product_id not in image_html and white_polo.variant_id not in image_html

    # Exact image identity remains persisted internally, but user-facing
    # captions contain readable product facts only (never IDs or local paths).
    image_history = app.render_history([
        {"role": "user", "content": "وريني صورته"},
        {"role": "assistant", "content": "دي صورة توضيحية.", "_images": [{
            "path": str(white_polo.image_path), "product_id": white_polo.product_id,
            "variant_id": white_polo.variant_id, "display_name": "تيشيرت بولو كلاسيك",
            "display_color": "أبيض", "illustrative": True,
        }]},
    ])
    image_component = image_history[1]["content"][1]
    assert "تيشيرت بولو كلاسيك" in image_component.label and "أبيض" in image_component.label
    assert white_polo.product_id not in image_component.label
    assert white_polo.variant_id not in image_component.label
    assert str(white_polo.image_path) not in image_component.label
    internal_ids = {value for variant in app.DEMO_CATALOG.variants
                    for value in (variant.product_id, variant.variant_id)}
    assert all(not any(identifier in caption for identifier in internal_ids)
               for _path, caption in app.catalog_image_gallery())

    price_sizes = app.render_history([{"role": "assistant", "content": "دي المقاسات والسعر.",
        "_cards": {"variant_ids": [white_polo.variant_id], "fields": ["name", "price", "sizes"], "language": "ar"}}])
    price_html = price_sizes[0]["content"][1].value
    assert "٦٤٩ جنيه" in price_html and "المقاسات:" in price_html
    assert "<img" not in price_html

    variant = app.DEMO_CATALOG.get("V-PVP-021-NAV-M")
    assert variant and variant.price_egp == 654
    arabic_card = app.cards([variant], arabic=True)
    english_card = app.cards([variant], arabic=False)
    assert "تيشيرت بتصميم ألوان متباينة" in arabic_card
    assert "تيشيرت" in arabic_card and "كحلي / أبيض" in arabic_card and "٦٥٤ جنيه" in arabic_card
    assert "Color-Block Crew T-Shirt" in english_card and "Navy / White" in english_card and "654 EGP" in english_card

    customer_id = app.STORE.create_customer("Arabic localization test")
    outfit_id = app.STORE.create_outfit(customer_id, "Shopping cart")
    app.STORE.save_conversation(customer_id, [{"role": "user", "content": "عايز التيشيرت ده"}],
                                {"demo_conversation": {"language": "ar"}})
    app.STORE.add_or_replace_outfit_item(outfit_id, variant.variant_id, variant.category, variant.product_id, 2)
    html, table, _remove, total, selected = app.outfit_payload(outfit_id)
    assert "<img" not in html  # images are shown on request, not as automatic catalog tiles in the cart.
    assert "تيشيرت بتصميم ألوان متباينة" in html and "كحلي / أبيض" in html
    assert "٦٥٤ جنيه" in html and "١٬٣٠٨ جنيه" in html and total == "١٬٣٠٨ جنيه"
    assert selected[0][1].variant_id == "V-PVP-021-NAV-M"
    assert table.iloc[0]["Unit price"] == "٦٥٤ جنيه"
    assert table.iloc[0]["Line total"] == "١٬٣٠٨ جنيه"
    assert selected[0][1].price_piasters * selected[0][2] == 130800

    review, token = app.prepare_buy(outfit_id, {"customer_id": customer_id})
    assert token and "١٬٣٠٨ جنيه" in review
    status, confirmed, _key, _history = app.confirm_buy(
        {"customer_id": customer_id}, outfit_id, True, token
    )
    assert "saved and verified" in status and not confirmed
    history, _state = app.STORE.conversation(customer_id)
    receipt = history[-1]["content"]
    assert "تيشيرت بتصميم ألوان متباينة" in receipt
    assert "كحلي / أبيض/M" in receipt and "١٬٣٠٨ جنيه" in receipt
    purchase = app.STORE.mock_purchase_for_confirmation(token)
    assert purchase["total_piasters"] == 130800
    assert purchase["items"][0]["variant_id"] == "V-PVP-021-NAV-M"

    english_customer = app.STORE.create_customer("English localization test")
    english_outfit = app.STORE.create_outfit(english_customer, "Shopping cart")
    app.STORE.save_conversation(english_customer, [{"role": "user", "content": "I want this shirt"}],
                                {"demo_conversation": {"language": "en"}})
    app.STORE.add_or_replace_outfit_item(english_outfit, variant.variant_id, variant.category, variant.product_id, 1)
    english_html, _table, _remove, english_total, _selected = app.outfit_payload(english_outfit)
    assert "Color-Block Crew T-Shirt" in english_html and "Navy / White" in english_html
    assert "654 EGP" in english_html and english_total == "654 EGP"
    english_review, english_token = app.prepare_buy(english_outfit, {"customer_id": english_customer})
    assert "654 EGP" in english_review
    english_status, _confirmed, _key, _history = app.confirm_buy(
        {"customer_id": english_customer}, english_outfit, True, english_token
    )
    english_history, _english_state = app.STORE.conversation(english_customer)
    assert "saved and verified" in english_status
    assert "Color-Block Crew T-Shirt" in english_history[-1]["content"]
    assert "Navy / White/M" in english_history[-1]["content"]
    assert "1,308 EGP" not in english_history[-1]["content"]
    assert "654 EGP" in english_history[-1]["content"]

    print("PASS: cards and image captions hide internal IDs; cart/mock-purchase render localizes Arabic while preserving English and exact numeric variant totals")


if __name__ == "__main__":
    main()
    _db_dir.cleanup()
