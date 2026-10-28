"""Regression tests for same-list follow-ups and exact color/image matching."""
from pathlib import Path

from src.demo_catalog import load_demo_catalog
from src.demo_conversation import DemoConversationEngine
from src.presentation import format_egp

ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
                            ROOT / "catalog_preview_review/preview_assets")
ENGINE = DemoConversationEngine(CATALOG, lambda _query, variants, limit: variants[:limit])


def turn(text, state):
    return ENGINE.respond(text, state)


def main():
    # Product interest resolves an exact catalog ID and remains product focused.
    reply, state, _ = turn("هل عندكم تيشيرتات؟", {})
    reply, state, _ = turn("حلو تيشيرت بولو كلاسيك", state)
    assert state["demo_conversation"]["focused_product_id"] == "PVP-015", (reply, state)
    assert "اختيار حلو" in reply and "رايح فين" not in reply

    # A product list without optional details contains names only.
    reply, state, action = turn("هاتلي خمس تيشيرتات", {})
    view = state["demo_conversation"]
    ids = list(view["displayed_variant_ids"])
    assert len(ids) == 5 and len(set(view["displayed_product_ids"])) == 5
    assert action["presentation"]["fields"] == ["name"]
    assert all(ENGINE.product_name(CATALOG.get(v), True) in reply for v in ids)

    # Image, price, and size follow-ups reuse those same ordered variants.
    for question, expected_fields in (
        ("وريني صورهم بس", ["name", "image"]),
        ("صور ليهم", ["name", "image"]),
        ("طب أسعارهم؟", ["name", "price"]),
        ("وريني المقاسات والأسعار", ["name", "price", "sizes"]),
    ):
        reply, state, action = turn(question, state)
        presentation = action["presentation"]
        assert presentation["variant_ids"] == ids, (question, presentation, ids)
        assert presentation["fields"] == expected_fields, (question, presentation)
        assert all(ENGINE.product_name(CATALOG.get(v), True) in reply for v in ids), (question, reply)
    assert all(format_egp(CATALOG.get(v).price_egp, True) in reply for v in ids)

    # Pink is a hard color constraint; selected images belong to those exact
    # pink variants rather than another color in the same product family.
    pink_reply, pink_state, pink_action = turn("هل عندك حاجة لونها وردي بصورها؟", {})
    pink_ids = pink_action["presentation"]["variant_ids"]
    assert pink_ids
    for variant_id in pink_ids:
        variant = CATALOG.get(variant_id)
        assert variant and variant.color == "Pink"
        assert variant.image_path and variant.image_path.name == f"{variant.product_id}-Pink.png"

    # A fresh no-match request clears the previous displayed SKUs, so a later
    # plural image request cannot leak stale pink results.
    miss_reply, miss_state, miss_action = turn("عاوز تيشيرت بنفسجي", pink_state)
    assert miss_action.get("presentation") is None
    assert miss_state["demo_conversation"]["displayed_variant_ids"] == []
    stale_reply, _stale_state, stale_action = turn("وريني صورهم", miss_state)
    assert stale_action.get("presentation") is None
    assert "اختيارات" not in stale_reply or "صور" not in stale_reply

    # Multi-color and striped assets remain attached to the exact variant rows.
    mixed = next(v for v in CATALOG.variants if v.color == "Navy / White" and v.in_stock)
    stripe = next(v for v in CATALOG.variants if "stripe" in v.pattern.lower() and v.in_stock)
    assert mixed.image_path and mixed.image_path.name == f"{mixed.product_id}-Navy-White.png"
    assert stripe.image_path and stripe.color in {"Black / White Stripe", "Blue / White Stripe"}
    print("PASS: product interest, dynamic fields, same-list follow-ups, exact pink image variants, stale-result clearing, and mixed/striped image mappings")


if __name__ == "__main__":
    main()
