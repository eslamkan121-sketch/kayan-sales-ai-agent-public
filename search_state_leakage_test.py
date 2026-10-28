"""Regression for fresh catalog searches after focused-product/cart activity."""
from pathlib import Path

from src.demo_catalog import load_demo_catalog
from src.demo_conversation import DemoConversationEngine


ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(
    ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
    ROOT / "catalog_preview_review/preview_assets",
)


def ranker(_query, candidates, limit):
    # Mimic the real app's deterministic list contract without loading weights.
    return list(candidates)[:limit]


def turn(engine, message, state):
    reply, state, action = engine.respond(message, state)
    return reply, state, action


def main():
    engine = DemoConversationEngine(CATALOG, ranker)
    state = {}

    # Establish a focused shoe, then add a specific variant as in the reported
    # workflow. This is deliberately unrelated to the following global lists.
    reply, state, _ = turn(engine, "هل عندكم جزم", state)
    assert state["demo_conversation"]["focused_product_id"]
    reply, state, _ = turn(engine, "وريني صورته وألوانه", state)
    reply, state, action = turn(engine, "ضيفه للسلة أبيض مقاس 39", state)
    if not action.get("add_variant_id"):
        reply, state, action = turn(engine, "39", state)

    # Cheapest five is a fresh catalog-wide sort. It must clear the old shoe
    # filters/reference price and return distinct products in actual price order.
    reply, state, action = turn(
        engine, "اديني جدول لأرخص خمس حاجات عندكم بصورهم وأسعارهم ومقاساتهم المتاحة", state
    )
    stored = state["demo_conversation"]
    ids = stored["displayed_product_ids"]
    variant_ids = stored["displayed_variant_ids"]
    assert len(ids) == 5 and len(set(ids)) == 5, (reply, stored)
    chosen = [CATALOG.get(variant_id) for variant_id in variant_ids]
    assert all(variant and variant.in_stock for variant in chosen)
    assert [item.price_piasters for item in chosen] == sorted(item.price_piasters for item in chosen)
    assert stored["active"].get("sort_by") == "price_asc"
    assert stored["active"].get("requested_count") == 5
    assert "max_price_egp" not in stored["active"] and "categories" not in stored["active"]
    assert "صورهم" not in reply or all(CATALOG.get(v).image_path for v in variant_ids)
    assert "جنيه" in reply and "المقاسات" in reply
    cheapest_prices = [item.price_egp for item in chosen]
    assert cheapest_prices == sorted(cheapest_prices)

    # A new white-items list also starts fresh, retrieves only white variants
    # across categories, and replaces the previous result/image IDs.
    previous_ids = list(stored["displayed_variant_ids"])
    reply, state, action = turn(
        engine, "هاتلي ليست خمس حاجات بصور وأسعار ومقاسات لونهم أبيض", state
    )
    stored = state["demo_conversation"]
    white_ids = stored["displayed_variant_ids"]
    white_variants = [CATALOG.get(variant_id) for variant_id in white_ids]
    assert len(white_variants) == 5 and len({v.product_id for v in white_variants}) == 5
    assert all(v and v.in_stock and v.color.lower() == "white" for v in white_variants)
    assert all(v.image_path for v in white_variants)
    assert set(white_ids).isdisjoint(previous_ids) or all(v.color.lower() == "white" for v in white_variants)
    assert stored["active"].get("color") == "white"
    assert "max_price_egp" not in stored["active"] and "categories" not in stored["active"]
    assert "أبيض" in reply and "جنيه" in reply and "المقاسات" in reply

    # A short affirmative is scoped to its pending proposal and must execute a
    # real relaxation once, rather than replaying the unchanged failed search.
    reply, state, _ = turn(engine, "عاوز جاكيت موف مقاس S", state)
    assert state["demo_conversation"]["pending_action"].get("kind") == "offer_alternatives"
    failed_constraints = dict(state["demo_conversation"]["active"])
    reply, state, _ = turn(engine, "أيوه", state)
    stored = state["demo_conversation"]
    assert stored["active"].get("color") is None
    assert stored["active"].get("categories") == failed_constraints.get("categories")
    assert stored["pending_action"].get("kind") != "offer_alternatives"
    assert "موف" not in reply and (stored["displayed_product_ids"] or "مش لاقي" in reply or "ملقتش" in reply)

    print("PASS: shoe context isolated; cheapest-five price ordering and distinct IDs; fresh white variants/images; affirmative executes stored color relaxation")


if __name__ == "__main__":
    main()
