"""Catalog-grounded occasion routing and search-context isolation checks."""
from pathlib import Path

from src.conversation_intelligence import route
from src.demo_catalog import load_demo_catalog
from src.demo_conversation import DemoConversationEngine

ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
                            ROOT / "catalog_preview_review/preview_assets")
ENGINE = DemoConversationEngine(CATALOG, lambda _query, variants, limit: variants[:limit])


def ask(text, state=None):
    return ENGINE.respond(text, state or {})


def check_request(text, expected):
    decision = route(text)
    assert decision.occasion == expected, (text, decision)
    reply, state, action = ask(text)
    demo = state["demo_conversation"]
    ids = demo["displayed_variant_ids"]
    assert ids and action.get("presentation"), (text, reply, action)
    variants = [CATALOG.get(variant_id) for variant_id in ids]
    assert all(variant and variant.in_stock for variant in variants), variants
    assert all(ENGINE._occasion_score(variant, expected) > 0 for variant in variants), (text, variants)
    assert "ملقتش" not in reply and "no matching" not in reply.lower(), (text, reply)
    assert demo["active"].get("occasion") == expected
    return reply, state, variants


def main():
    beach_reply, state, beach = check_request("عاوز حاجة للبحر", "beach")
    assert all({part.strip().lower() for part in variant.occasion.split(";")} & {"beach", "resort"} for variant in beach)
    assert "البحر" in beach_reply or "المنتجع" in beach_reply

    wedding_a, wedding_state, wedding = check_request("عليا فرح بكرة وعاوز لبس مناسب", "wedding")
    wedding_b, _, _ = check_request("عاوز حاجة للفرح", "wedding")
    assert any("event" in v.occasion.lower() or "smart casual" in v.occasion.lower() for v in wedding)
    assert "مناسبات" in wedding_a or "event" in wedding_a.lower()

    _sports_reply, _sports_state, sports = check_request("عاوز لبس رياضي", "sports")
    assert all("sports" in v.occasion.lower() or v.category == "activewear" or v.fit == "Athletic" for v in sports)
    _work_reply, _work_state, work = check_request("عاوز حاجة للشغل", "work")
    assert all("work" in v.occasion.lower() or (v.category == "blazer" and v.fit in {"Tailored", "Structured"}) for v in work)
    _casual_reply, _casual_state, casual = check_request("عاوز لبس خروجة", "casual")
    assert all(any(tag in v.occasion.lower() for tag in ("casual", "everyday", "streetwear")) for v in casual)
    _summer_reply, _summer_state, summer = check_request("عاوز حاجة للصيف", "summer")
    assert all("summer" in v.season.lower() or "beach" in v.occasion.lower() or "resort" in v.occasion.lower() for v in summer)
    _winter_reply, _winter_state, winter = check_request("عاوز لبس للشتاء", "winter")
    assert all("winter" in v.season.lower() or v.category in {"knitwear", "outerwear", "jacket"} for v in winter)

    # A shoe discussion and its exact SKU remain in the cart, while independent
    # list and occasion searches clear shoe search filters and displayed IDs.
    _reply, shoe_state, _action = ask("عايز جزمة", {})
    shoe_id = shoe_state["demo_conversation"]["focused_variant_id"]
    shoe = CATALOG.get(shoe_id)
    exact_shoe = next(v for v in CATALOG.variants if v.product_id == shoe.product_id and
                      v.color == shoe.color and v.in_stock)
    _reply, shoe_state, add_action = ask(f"ضيفه للسلة مقاس {exact_shoe.size}", shoe_state)
    assert add_action.get("add_variant_id") == exact_shoe.variant_id, (shoe, exact_shoe, add_action)
    list_reply, list_state, list_action = ask("اديني ليست لخمس تيشيرتات بأسعارهم وألوانهم ومقاساتهم", shoe_state)
    list_demo = list_state["demo_conversation"]
    assert len(list_demo["displayed_product_ids"]) == 5
    assert all(CATALOG.get(v).category == "t-shirt" for v in list_demo["displayed_variant_ids"])
    assert "الألوان" in list_reply or "تيشيرت" in list_reply
    wedding_reply, wedding_state, wedding_action = ask("عليا فرح بكرة وعاوز لبس مناسب", list_state)
    assert wedding_state["demo_conversation"]["active"].get("occasion") == "wedding"
    assert all(CATALOG.get(v).category != "shoes" for v in wedding_state["demo_conversation"]["displayed_variant_ids"])
    sports_reply, sports_state, _ = ask("عاوز لبس رياضي", wedding_state)
    assert sports_state["demo_conversation"]["active"].get("occasion") == "sports"
    assert all(CATALOG.get(v).category == "activewear" or
               ("sports" in CATALOG.get(v).occasion.lower() and "performance" in CATALOG.get(v).material.lower())
               for v in sports_state["demo_conversation"]["displayed_variant_ids"])
    assert "رياضي" in sports_reply or "الرياضة" in sports_reply
    print("PASS: beach, wedding, sports, work, casual, summer and winter suitability; stale shoe constraints cleared on new list/occasion requests")


if __name__ == "__main__":
    main()
