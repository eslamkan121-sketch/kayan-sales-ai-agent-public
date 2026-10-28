"""Regression coverage for named products, ordered references, pending SKU choices, and image IDs."""
from pathlib import Path

from src.demo_catalog import load_demo_catalog
from src.demo_conversation import DemoConversationEngine
from src.presentation import ARABIC_PRODUCT_NAMES, format_egp, localize_color, localize_product_name

ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
                            ROOT / "catalog_preview_review/preview_assets")
ENGINE = DemoConversationEngine(CATALOG, lambda _query, items, limit: items[:limit])


def turn(text, state):
    reply, saved, action = ENGINE.respond(text, state)
    return reply, saved, action


def main():
    # Selecting an Arabic display name resolves to the workbook Product ID and
    # does not fall through to generic styling advice.
    _overview, state, _ = turn("هل عندكم تيشرتات؟", {})
    assert state["demo_conversation"]["displayed_product_ids"]
    reply, state, _ = turn("حلو تيشيرت بولو كلاسيك", state)
    selected = state["demo_conversation"]
    assert selected["focused_product_id"] == "PVP-015", (reply, selected)
    assert "تيشيرت بولو كلاسيك" in reply and "رايح فين" not in reply

    reply, state, _ = turn("أنا عايز تيشرت بولو كلاسيك ايه الألوان المتاحه", state)
    assert "الألوان المتاحة" in reply and selected["focused_product_id"] == "PVP-015"
    actual_colors = {v.color for v in CATALOG.variants if v.product_id == "PVP-015" and v.in_stock}
    assert all(localize_color(color, True) in reply for color in actual_colors), (reply, actual_colors)
    assert state["demo_conversation"]["focused_product_id"] == "PVP-015"

    # A fresh five-item list keeps order and distinct Product IDs. Ordinal
    # follow-ups resolve against that list and update focus to the selected ID.
    reply, state, _ = turn("هاتلي خمس تيشيرتات بأسعارها ومقاساتها", {})
    state_data = state["demo_conversation"]
    shown = state_data["displayed_product_ids"]
    assert len(shown) == 5 and len(set(shown)) == 5, (reply, shown)
    for phrase, index in (("اختار الاختيار الاول فيه ايه الوان متاحه", 0),
                          ("اختار الاختيار التاني فيه ايه الوان متاحه", 1),
                          ("اختار الاختيار التالت فيه ايه الوان متاحه", 2)):
        reply, state, _ = turn(phrase, state)
        assert "الألوان المتاحة" in reply
        assert state["demo_conversation"]["focused_product_id"] == shown[index], (phrase, state)

    # Price facts stay within the exact focused product's source prices.
    product_id = state["demo_conversation"]["focused_product_id"]
    source_prices = sorted({v.price_egp for v in CATALOG.variants if v.product_id == product_id and v.in_stock})
    reply, state, _ = turn("طب بكام؟", state)
    assert any(format_egp(price, True) in reply for price in source_prices), (reply, source_prices)

    # Completing a pending cart action with a bare size and explicit color
    # resolves the exact Product ID + color + size; it never substitutes.
    _reply, polo_state, _ = turn("تيشيرت بولو كلاسيك", {})
    _reply, polo_state, action = turn("ضيفه للسلة", polo_state)
    assert polo_state["demo_conversation"]["pending_action"]["kind"] == "choose_size_for_cart"
    reply, polo_state, action = turn("L و أبيض", polo_state)
    variant = CATALOG.get(action.get("add_variant_id"))
    assert variant and (variant.product_id, variant.color, variant.size) == ("PVP-015", "White", "L"), (reply, action)

    # Numeric shoe sizes remain valid pending answers.
    _reply, shoe_state, _ = turn("عايز جزمة", {})
    _reply, shoe_state, _ = turn("ضيفه للسلة", shoe_state)
    assert shoe_state["demo_conversation"]["pending_action"]["kind"] == "choose_size_for_cart"
    reply, shoe_state, action = turn("39", shoe_state)
    shoe = CATALOG.get(action.get("add_variant_id"))
    assert shoe and shoe.category == "shoes" and shoe.size == "39", (reply, action)

    # Images are associated with the exact focused/displayed variant IDs.
    reply, state, action = turn("هاتلي خمس تيشيرتات بصورهم وأسعارهم ومقاساتهم", {})
    demo = state["demo_conversation"]
    assert len(demo["displayed_product_ids"]) == 5
    assert len(action["image_variant_ids"]) == 5
    assert action["image_variant_ids"] == demo["displayed_variant_ids"]
    assert action["presentation"]["fields"] == ["name", "image", "price", "sizes"]
    assert all(CATALOG.get(variant_id).image_path.is_file() for variant_id in action["image_variant_ids"])
    reply, state, action = turn("عاوز صورهم", state)
    assert action["presentation"]["variant_ids"] == state["demo_conversation"]["displayed_variant_ids"]
    assert action["presentation"]["fields"] == ["name", "image"]
    assert "الاختيارات" in reply and "image" in action["presentation"]["fields"]
    reply, state, action = turn("وريني صورته", state)
    assert action["presentation"]["variant_ids"] == [state["demo_conversation"]["focused_variant_id"]]
    assert action["presentation"]["fields"] == ["name", "image"]

    # Bare, pronoun, and named/category image requests share one compact card
    # path and preserve the exact known focus instead of launching a new search.
    _reply, state, _ = turn("هل عندكم جزم", {})
    focused = state["demo_conversation"]["focused_variant_id"]
    for phrase in ("صور", "صور ليه", "وريني صورة الجزمة"):
        reply, state, action = turn(phrase, state)
        assert action.get("presentation"), (phrase, action)
        assert action["presentation"]["variant_ids"] == [focused], (phrase, focused, action)
        assert action["presentation"]["fields"] == ["name", "image"]
        assert "كوتشي" in reply or "سنيكرز" in reply, (phrase, reply)

    # A named product request resolves by its catalog identity and uses the
    # identical card schema; plural images keep the previous ranking order.
    _reply, state, _ = turn("هاتلي خمس تيشيرتات", {})
    shown = list(state["demo_conversation"]["displayed_variant_ids"])
    reply, state, action = turn("صورهم", state)
    assert action["presentation"]["variant_ids"] == shown
    assert all(localize_product_name(CATALOG.get(item).name, True) in reply for item in shown)
    assert action["presentation"]["fields"] == ["name", "image"]

    assert len(ARABIC_PRODUCT_NAMES) >= 24
    assert all(v.image_path and v.image_path.is_file() for v in CATALOG.variants)
    print("PASS: named Product ID selection, color facts, first/second/third item references, exact price facts, pending White/L and shoe 39, and single/plural image SKU continuity")


if __name__ == "__main__":
    main()
