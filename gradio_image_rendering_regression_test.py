"""Live Gradio regression for unified compact image cards and transcript text."""
from __future__ import annotations

from pathlib import Path

from gradio_test_support import isolated_gradio_server
from src.demo_catalog import load_demo_catalog
from src.session_store import CustomerSessionStore
from src.presentation import localize_product_name
from src.presentation import format_egp
from src.conversation_intelligence import route
from src.demo_conversation import DemoConversationEngine

ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
                            ROOT / "catalog_preview_review/preview_assets")
PLANNER = DemoConversationEngine(CATALOG, lambda _query, variants, limit: variants[:limit])


def collect(value, component=None):
    found = []
    if isinstance(value, dict):
        if value.get("component") == component:
            found.append(value)
        for child in value.values():
            found.extend(collect(child, component))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(collect(child, component))
    return found


def strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [part for child in value.values() for part in strings(child)]
    if isinstance(value, (list, tuple)):
        return [part for child in value for part in strings(child)]
    return []


def main():
    with isolated_gradio_server() as (client, database):
        store = CustomerSessionStore(database)
        first = client.predict("Image Customer A", "image-a", api_name="/new_customer_cart")
        customer_a = first[0]["value"]

        overview = client.predict("هل عندكم جزم", api_name="/chat")
        a_history, a_state = store.conversation(customer_a)
        assert "كوتشي" in a_history[-1]["content"] or "سنيكرز" in a_history[-1]["content"]
        expected_variant = CATALOG.get(a_state["demo_conversation"]["focused_variant_id"])
        assert expected_variant and expected_variant.category == "shoes"
        expected_name = localize_product_name(expected_variant.name, True)

        image_payloads = []
        for prompt in ("صور", "صور ليه", "وريني صورة الجزمة"):
            response = client.predict(prompt, api_name="/chat")
            payload = response[0]
            html_blocks = collect(payload, "html")
            image_blocks = collect(payload, "image")
            assert html_blocks and not image_blocks, (prompt, payload)
            rendered = repr(payload)
            assert expected_name in rendered, (prompt, expected_name, payload)
            assert expected_variant.image_path.name not in rendered
            assert expected_variant.variant_id not in rendered and expected_variant.product_id not in rendered
            assert "undefined" not in rendered.lower(), (prompt, payload)
            history, state = store.conversation(customer_a)
            turn = history[-1]
            assert turn["content"].strip(), turn
            assert expected_name in turn["content"], turn
            assert turn["_cards"]["variant_ids"] == [expected_variant.variant_id], turn
            assert turn["_cards"]["fields"] == ["name", "image"], turn
            assert turn["_images"][0]["variant_id"] == expected_variant.variant_id
            assert turn["_images"][0]["image_provenance"] == expected_variant.image_provenance
            image_payloads.append(payload)
        assert all("Illustrative DEMO image" in repr(payload) or "صورة توضيحية" in repr(payload)
                   for payload in image_payloads)

        # A named product and a plural request both resolve to the same card
        # component and preserve result order in transcript and images.
        sneaker = next(v for v in CATALOG.variants if v.category == "shoes" and v.in_stock)
        named = client.predict(f"وريني صورة {localize_product_name(sneaker.name, True)}", api_name="/chat")
        assert collect(named[0], "html") and not collect(named[0], "image")
        named_history, _ = store.conversation(customer_a)
        assert named_history[-1]["_cards"]["variant_ids"] == [sneaker.variant_id]
        assert sneaker.variant_id in [item["variant_id"] for item in named_history[-1]["_images"]]

        # Five-product list identity survives image/price follow-ups and ordinal
        # references. Exact fields track what the customer requested.
        client.predict("اديني ليست لخمس تيشيرتات بأسعارهم وألوانهم ومقاساتهم", api_name="/chat")
        listed_history, listed_state = store.conversation(customer_a)
        listed_ids = list(listed_state["demo_conversation"]["displayed_variant_ids"])
        listed_products = list(listed_state["demo_conversation"]["displayed_product_ids"])
        assert len(listed_ids) == 5 and len(set(listed_products)) == 5
        assert listed_history[-1]["_cards"]["fields"] == ["name", "price", "sizes", "colors"]
        plural = client.predict("صور ليهم", api_name="/chat")
        assert collect(plural[0], "html") and not collect(plural[0], "image")
        plural_history, _ = store.conversation(customer_a)
        assert plural_history[-1]["_cards"]["variant_ids"] == listed_ids
        html_blocks = collect(plural[0], "html")
        html_value = next((value for value in strings(html_blocks[-1]) if "class='kayan-product-grid'" in value), "")
        html_card_count = html_value.count("<article class='kayan-product-card'>")
        assert html_card_count == 5, (html_card_count, plural_history[-1]["_cards"], plural_history[-1]["content"], html_value[:1200])
        assert "undefined" not in repr(plural[0]).lower()
        assert all(localize_product_name(CATALOG.get(vid).name, True) in plural_history[-1]["content"]
                   for vid in listed_ids)
        prices = client.predict("أسعارهم", api_name="/chat")
        prices_history, _ = store.conversation(customer_a)
        assert prices_history[-1]["_cards"]["variant_ids"] == listed_ids
        assert prices_history[-1]["_cards"]["fields"] == ["name", "price"]
        assert all(format_egp(CATALOG.get(vid).price_egp, True) in prices_history[-1]["content"] for vid in listed_ids)
        ordinal = client.predict("الأول بكام", api_name="/chat")
        ordinal_history, _ = store.conversation(customer_a)
        assert format_egp(CATALOG.get(listed_ids[0]).price_egp, True) in ordinal_history[-1]["content"]
        assert "undefined" not in repr(ordinal[0]).lower()

        # Product cards show only requested optional facts; product names are
        # mandatory for identity and the image layout remains responsive.
        field_cases = (
            ("اديني ليست لخمس تيشيرتات", ["name"]),
            ("اديني ليست لخمس تيشيرتات بصورها", ["name", "image"]),
            ("اديني ليست لخمس تيشيرتات بأسعارهم", ["name", "price"]),
            ("اديني ليست لخمس تيشيرتات بألوانهم ومقاساتهم وأسعارهم", ["name", "price", "sizes", "colors"]),
            ("اديني ليست لخمس تيشيرتات بصورها وأسعارهم وألوانهم ومقاساتهم", ["name", "image", "price", "sizes", "colors"]),
        )
        for prompt, expected_fields in field_cases:
            client.predict(prompt, api_name="/chat")
            field_history, field_state = store.conversation(customer_a)
            assert len(field_state["demo_conversation"]["displayed_product_ids"]) == 5
            assert field_history[-1]["_cards"]["fields"] == expected_fields, (prompt, field_history[-1]["_cards"])
            assert all(localize_product_name(CATALOG.get(vid).name, True) in field_history[-1]["content"]
                       for vid in field_state["demo_conversation"]["displayed_variant_ids"])

        # Add an exact in-stock shoe SKU, then prove new lists/occasions reset
        # stale shoe constraints without deleting the separately selected item.
        shoe_state = client.predict("عايز جزمة", api_name="/chat")
        shoe_id = store.conversation(customer_a)[1]["demo_conversation"]["focused_variant_id"]
        shoe = CATALOG.get(shoe_id)
        chosen_shoe = next(v for v in CATALOG.variants if v.product_id == shoe.product_id and
                           v.color == shoe.color and v.in_stock)
        add = client.predict(f"ضيفه للسلة مقاس {chosen_shoe.size}", api_name="/chat")
        assert "تمت إضافة" in repr(add[0]) and "Added " not in repr(add[0])
        outfit = store.list_outfits(customer_a)[0]
        assert any(item["variant_id"] == chosen_shoe.variant_id
                   for item in store.outfit_record(outfit["outfit_id"])["items"])

        occasion_cases = (
            ("عاوز حاجة للبحر", "beach"),
            ("عليا فرح بكرة وعاوز لبس مناسب", "wedding"),
            ("عاوز حاجة للفرح", "wedding"),
            ("عاوز لبس رياضي", "sports"),
            ("عاوز حاجة للشغل", "work"),
            ("عاوز لبس خروجة", "casual"),
        )
        for prompt, occasion in occasion_cases:
            decision = route(prompt)
            assert decision.occasion == occasion, (prompt, decision)
            result = client.predict(prompt, api_name="/chat")
            occasion_history, occasion_state = store.conversation(customer_a)
            demo = occasion_state["demo_conversation"]
            assert demo["active"].get("occasion") == occasion, (prompt, demo)
            variants = [CATALOG.get(vid) for vid in demo["displayed_variant_ids"]]
            assert variants and all(v and PLANNER._occasion_score(v, occasion) > 0 for v in variants), (prompt, variants)
            assert all(v.category != "shoes" for v in variants), (prompt, variants)
            assert "ماتش" not in occasion_history[-1]["content"] and "ملقتش" not in occasion_history[-1]["content"]
            assert "undefined" not in repr(result[0]).lower()
            assert any(item["variant_id"] == chosen_shoe.variant_id
                       for item in store.outfit_record(outfit["outfit_id"])["items"])

        # Simulate a saved pre-migration turn that only has _images. On a real
        # customer restore it must be converted to the shared card renderer.
        legacy_history, legacy_state = store.conversation(customer_a)
        legacy_turn = next(turn for turn in reversed(legacy_history) if turn.get("_images"))
        legacy_image_ids = [image["variant_id"] for image in legacy_turn["_images"]]
        legacy_turn.pop("_cards", None)
        store.save_conversation(customer_a, legacy_history, legacy_state)

        # Customer switching and restored history re-render legacy image turns
        # through compact cards without losing exact SKU or provenance.
        second = client.predict("Image Customer B", "image-b", api_name="/new_customer_cart")
        customer_b = second[0]["value"]
        assert customer_b != customer_a
        assert store.conversation(customer_b)[0] == []
        restored = client.predict(customer_a, api_name="/switch_customer_cart")
        assert restored[0] and "component': 'html'" in repr(restored[0])
        restored_text = repr(restored[0])
        assert "undefined" not in restored_text.lower()
        assert expected_name in restored_text
        assert "component': 'image'" not in restored_text
        restored_history = store.conversation(customer_a)[0]
        assert any(set(legacy_image_ids).intersection(image["variant_id"] for image in turn.get("_images", []))
                   for turn in restored_history)
        print("PASS: live Gradio single/plural images, five-SKU context and ordinals, requested card fields, occasion recommendations/context resets, localized cart, customer isolation, and restored transcripts")


if __name__ == "__main__":
    main()
