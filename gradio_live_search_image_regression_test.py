"""Focused live Gradio regressions for fresh searches, price ranges, and transcript text."""
from __future__ import annotations

from pathlib import Path
import json

from gradio_test_support import isolated_gradio_server
from src.demo_catalog import load_demo_catalog
from src.session_store import CustomerSessionStore
from src.presentation import format_egp, localize_product_name
from src.demo_conversation import DemoConversationEngine, _price_range

ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
                            ROOT / "catalog_preview_review/preview_assets")


def find(value, component):
    if isinstance(value, dict):
        return ([value] if value.get("component") == component else []) + [
            item for child in value.values() for item in find(child, component)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in find(child, component)]
    return []


def main():
    assert _price_range("منتجات من ٤٠٠ جنيه لـ ٥٠٠ جنيه") == (400, 500)
    assert _price_range("products between EGP 400 and 500") == (400, 500)
    planner = DemoConversationEngine(CATALOG, lambda _query, variants, limit: variants[:limit])
    exact_boundary = planner._variants({"price_range_egp": [449, 449]})
    assert exact_boundary and all(v.price_egp == 449 for v in exact_boundary)

    with isolated_gradio_server() as (client, database):
        created = client.predict("Search regressions", api_name="/new_customer_cart")
        customer_id = created[0]["value"]
        store = CustomerSessionStore(database)

        # Scenario A: five distinct tees, then all of the same displayed items.
        client.predict("هاتلي خمس تيشيرتات بأسعارهم وألوانهم ومقاساتهم", api_name="/chat")
        listed_history, listed_storage = store.conversation(customer_id)
        listed_state = listed_storage["demo_conversation"]
        listed_ids = list(listed_state["displayed_variant_ids"])
        product_ids = list(listed_state["displayed_product_ids"])
        assert len(listed_ids) == len(product_ids) == 5 and len(set(product_ids)) == 5
        assert all(CATALOG.get(variant_id).category == "t-shirt" for variant_id in listed_ids)
        assert all(term in listed_history[-1]["content"] for term in ("جنيه", "المقاسات"))
        image_result = client.predict("صورهم", api_name="/chat")
        image_history, image_storage = store.conversation(customer_id)
        image_turn = image_history[-1]
        image_state = image_storage["demo_conversation"]
        assert image_turn["_cards"]["variant_ids"] == listed_ids
        html_blocks = find(image_result[0], "html")
        assert html_blocks and not find(image_result[0], "image")
        assert len(listed_ids) == 5
        assert all(CATALOG.get(v).variant_id in repr(image_turn["_images"]) for v in listed_ids)
        assert image_state["displayed_variant_ids"] == listed_ids

        # Scenario B: an image follow-up on shoes, then a fresh global cheap sort.
        client.predict("هل في جزمة؟", api_name="/chat")
        client.predict("عاوز صورة ليها", api_name="/chat")
        shoe_state = store.conversation(customer_id)[1]["demo_conversation"]
        shoe_ids = list(shoe_state["displayed_variant_ids"])
        assert shoe_ids and all(CATALOG.get(v).category == "shoes" for v in shoe_ids)
        cheapest_response = client.predict(
            "عاوز جدول لأقل خمس منتجات سعر في الكتالوج، فيه أسعارهم وصورهم ومقاساتهم المتاحة",
            api_name="/chat")
        cheap_history, cheap_storage = store.conversation(customer_id)
        cheap_state = cheap_storage["demo_conversation"]
        cheap_ids = list(cheap_state["displayed_variant_ids"])
        cheap_variants = [CATALOG.get(variant_id) for variant_id in cheap_ids]
        assert len(cheap_state["displayed_product_ids"]) == 5
        assert len(set(cheap_state["displayed_product_ids"])) == 5
        assert all(v and v.in_stock for v in cheap_variants)
        assert [v.price_piasters for v in cheap_variants] == sorted(v.price_piasters for v in cheap_variants)
        assert cheap_state["active"].get("sort_by") == "price_asc"
        assert not set(cheap_state["active"]) & {"categories", "color", "size", "price_range_egp"}
        assert all(localize_product_name(v.name, True) in cheap_history[-1]["content"] for v in cheap_variants)
        assert "جنيه" in cheap_history[-1]["content"] and "المقاسات" in cheap_history[-1]["content"]
        cheapest_cards = repr(find(cheapest_response[0], "html"))
        assert cheapest_cards and all(format_egp(v.price_egp, True) in cheapest_cards for v in cheap_variants)
        assert cheap_state["displayed_price_evidence"] == [
            {"product_id": v.product_id, "variant_id": v.variant_id,
             "price_piasters": v.price_piasters, "eligible": True,
             "reason": "lowest_valid_eligible_variant"} for v in cheap_variants]
        print("LIVE CHEAPEST FIVE: " + ", ".join(f"{v.product_id}/{v.variant_id}={v.price_egp} EGP" for v in cheap_variants))
        assert "undefined" not in repr(cheapest_response[0]).lower()

        # Scenario C: the range query replaces the previous list and applies
        # inclusive bounds to the actual variants shown.
        range_response = client.predict("عاوز جدول يعرض منتجات من ٤٠٠ جنيه لـ ٥٠٠ جنيه", api_name="/chat")
        range_history, range_storage = store.conversation(customer_id)
        range_state = range_storage["demo_conversation"]
        range_ids = list(range_state["displayed_variant_ids"])
        range_variants = [CATALOG.get(variant_id) for variant_id in range_ids]
        assert range_state["active"].get("price_range_egp") == [400, 500]
        assert "sort_by" not in range_state["active"]
        assert range_ids != cheap_ids or range_state["active"].get("price_range_egp") == [400, 500]
        assert range_variants and all(v and 400 <= v.price_egp <= 500 for v in range_variants)
        assert all(format_egp(v.price_egp, True) in range_history[-1]["content"] for v in range_variants)
        assert find(range_response[0], "html"), "range results must render through the visible product cards"
        rendered_range = repr(find(range_response[0], "html"))
        assert all(format_egp(v.price_egp, True) in rendered_range for v in range_variants)
        assert range_state["displayed_price_evidence"] == [
            {"product_id": v.product_id, "variant_id": v.variant_id,
             "price_piasters": v.price_piasters, "eligible": True,
             "reason": "within_inclusive_price_range"} for v in range_variants]
        assert "undefined" not in repr(range_response[0]).lower()

        # A second fresh range (450–500) must discard prior results and keep
        # the UI cards, saved transcript, eligibility evidence and workbook
        # values aligned to the same exact variant IDs.
        range_450_response = client.predict("عاوز منتجات من ٤٥٠ لـ ٥٠٠ جنيه", api_name="/chat")
        range_450_history, range_450_storage = store.conversation(customer_id)
        range_450_state = range_450_storage["demo_conversation"]
        range_450_ids = list(range_450_state["displayed_variant_ids"])
        range_450_variants = [CATALOG.get(variant_id) for variant_id in range_450_ids]
        assert range_450_state["active"].get("price_range_egp") == [450, 500]
        assert range_450_ids and all(v and 45000 <= v.price_piasters <= 50000 for v in range_450_variants)
        assert len({v.product_id for v in range_450_variants}) == len(range_450_variants)
        assert range_450_state["displayed_price_evidence"] == [
            {"product_id": v.product_id, "variant_id": v.variant_id,
             "price_piasters": v.price_piasters, "eligible": True,
             "reason": "within_inclusive_price_range"} for v in range_450_variants]
        visible_450 = repr(find(range_450_response[0], "html"))
        assert visible_450 and all(format_egp(v.price_egp, True) in visible_450 for v in range_450_variants)
        assert all(format_egp(v.price_egp, True) in range_450_history[-1]["content"] for v in range_450_variants)
        print("LIVE 400-500: " + ", ".join(f"{v.product_id}/{v.variant_id}={v.price_egp} EGP" for v in range_variants))
        print("LIVE 450-500: " + ", ".join(f"{v.product_id}/{v.variant_id}={v.price_egp} EGP" for v in range_450_variants))

        # Scenario D: all equivalent single-image turns use compact cards and
        # preserve the exact image SKU with no user-visible IDs or paths.
        client.predict("هل في جزمة؟", api_name="/chat")
        focused = store.conversation(customer_id)[1]["demo_conversation"]["focused_variant_id"]
        focused_variant = CATALOG.get(focused)
        for prompt in ("عاوز صورة ليه", "صور", "عاوز صور ليه"):
            rendered = client.predict(prompt, api_name="/chat")[0]
            turn = store.conversation(customer_id)[0][-1]
            assert find(rendered, "html") and not find(rendered, "image"), (prompt, rendered)
            assert turn["_cards"]["variant_ids"] == [focused_variant.variant_id]
            customer_text = turn["content"]
            assert "undefined" not in repr(rendered).lower() and "undefined" not in customer_text.lower()
            assert focused_variant.variant_id not in customer_text and focused_variant.image_path.name not in customer_text
            card_html = repr(find(rendered, "html"))
            assert "height:150px" in card_html or "height: 150px" in card_html
            assert focused_variant.variant_id not in card_html and focused_variant.image_path.name not in card_html

        # Scenario E: independent occasion searches replace previous filters.
        client.predict("عاوز حاجة للبحر", api_name="/chat")
        client.predict("عندي فرح بكرة", api_name="/chat")
        _history, advice_storage = store.conversation(customer_id)
        advice_state = advice_storage["demo_conversation"]
        assert advice_state["active"].get("occasion") == "wedding", advice_state["active"]
        assert not set(advice_state["active"]) & {"price_range_egp", "sort_by", "color", "size"}
        assert not advice_state["displayed_variant_ids"], advice_state["displayed_variant_ids"]
        for prompt, occasion in (("عاوز حاجة للبحر", "beach"), ("اقترح عليا حاجة للفرح بكرة", "wedding"),
                                 ("عاوز لبس رياضي", "sports"), ("عاوز حاجة للشغل", "work")):
            client.predict(prompt, api_name="/chat")
            _history, storage = store.conversation(customer_id)
            state = storage["demo_conversation"]
            assert state["active"].get("occasion") == occasion, (prompt, state["active"])
            assert not set(state["active"]) & {"price_range_egp", "sort_by", "color", "size"}
            variants = [CATALOG.get(v) for v in state["displayed_variant_ids"]]
            assert variants and all(v and v.in_stock for v in variants), (prompt, state)

        # Scenario F: use the application-level readable-text alternative for
        # cards, since a browser's native copy handler may stringify HTML as
        # "undefined". Preserve every displayed product name in the export.
        client.predict("هاتلي خمس تيشيرتات بأسعارهم وألوانهم ومقاساتهم", api_name="/chat")
        transcript = client.predict(api_name="/copy_safe_transcript")
        transcript_text = repr(transcript)
        final_history = store.conversation(customer_id)[0]
        final_ids = final_history[-1].get("_cards", {}).get("variant_ids", [])
        assert final_ids and all(localize_product_name(CATALOG.get(v).name, True) in transcript_text for v in final_ids)
        assert "undefined" not in transcript_text.lower()

        log_path = database.parent / "outputs" / "demo_conversation_dev.log"
        events = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            payload = line[line.find("{"):] if "{" in line else ""
            if payload:
                events.append(json.loads(payload))
        search_events = [event for event in events if event.get("event") == "search_trace"]
        assert any(event.get("search_decision") == "new_global_price_sort" and
                   event.get("displayed_product_ids") for event in search_events)
        range_event = next(event for event in search_events
                           if event.get("search_decision") == "new_price_range_search")
        assert range_event["active_filters"]["price_range_egp"] == [400, 500]
        assert "outside_price_range" in repr(range_event["constraint_rejection_reasons"])
        assert range_event["price_evidence"] and all(item["eligible"] and item["reason"] == "within_inclusive_price_range" for item in range_event["price_evidence"])
        assert all(item["price_piasters"] // 100 in range(400, 501) for item in range_event["price_evidence"])
        assert any(event.get("selected_image_variant_ids") and
                   event.get("rendered_response_type") == "compact_product_cards" for event in events)

    print("PASS: live isolated Gradio price-range UI values match exact eligible variants; global cheapest-five and other conversation regressions passed")


if __name__ == "__main__":
    main()
