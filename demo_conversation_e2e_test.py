"""End-to-end planner regression using the real workbook (no model weights needed)."""
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.demo_catalog import load_demo_catalog
from src.demo_conversation import DemoConversationEngine


ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx", ROOT / "catalog_preview_review/preview_assets")


def ranker(_query, candidates, limit):
    # The real app passes trained XLM-R here.  This deterministic ranker lets
    # the state/planner regression run without allocating the model process.
    return sorted(candidates, key=lambda v: (v.product_id, v.variant_id))[:limit]


def turn(engine, message, state):
    reply, state, _ = engine.respond(message, state)
    print(message, "=>", reply.replace("\n", " | "))
    return reply, state


def main():
    engine = DemoConversationEngine(CATALOG, ranker)
    state = {}
    reply, state = turn(engine, "أنا بحب لون الأزرق هل عندك حاجه لونها ازرق", state)
    assert "أزرق" in reply and state["demo_conversation"]["favorite_colors"] == ["blue"]
    reply, state = turn(engine, "طب عندكم تيشرت احمر", state)
    assert "أحمر" in reply and "تيشيرت" in reply
    primary = state["demo_conversation"]["focused_product_id"]
    reply, state = turn(engine, "طب عندكم تيشرت ابيض", state)
    assert "أبيض" in reply and state["demo_conversation"]["focused_product_id"] != primary
    primary = state["demo_conversation"]["focused_product_id"]
    reply, state = turn(engine, "حلو ده سعره كام ومقاساته المتاحة واللون", state)
    assert "سعر" in reply and "المقاسات" in reply and "الألوان" in reply
    assert state["demo_conversation"]["focused_product_id"] == primary
    reply, state = turn(engine, "طب عندك حاجه لي خروجه البحر", state)
    assert "اختيارات" in reply or "البحر" in reply
    assert state["demo_conversation"]["active"].get("occasion") == "beach"
    reply, state = turn(engine, "طب عندكم تيشرت مخطط", state)
    assert "مخطط" in reply and "غير قابل" not in reply

    # Workwear begins a fresh task: no stale tee, white or beach constraint.
    reply, state = turn(engine, "عاوز لبس في مقابله شغل", state)
    active = state["demo_conversation"]["active"]
    assert active.get("occasion") == "work" and "color" not in active and "t-shirt" not in active.get("categories", [])
    assert "بليزر" in reply or "قميص" in reply or "فستان" in reply

    # Out-of-catalog colour must remain a real explicit constraint.
    reply, state = turn(engine, "عاوز تيشيرت اخضر", state)
    assert "ملقتش" in reply and state["demo_conversation"]["pending_action"].get("missing_color") == "green"
    reply, state = turn(engine, "ايوه ياريت", state)
    assert "اختيار" in reply or "option" in reply

    # Requested-count output is distinct by Product ID; unavailable max price
    # is a hard constraint, not a reference to an older product.
    reply, state = turn(engine, "اديني ليست خمس حجات لونها ابيض", state)
    ids = state["demo_conversation"]["displayed_product_ids"]
    assert len(ids) == len(set(ids)) and len(ids) <= 5
    reply, state = turn(engine, "أنا عاوز حاجه سعرها اقل من ٣٠٠ جنيه", state)
    assert "300" in reply or "299" in reply or "ملقتش" in reply

    # Multi-favourite memory and unsupported mixed-colour must not invent SKU.
    reply, state = turn(engine, "أنا بحب لون الأصفر و الأزرق هل في حاجه مكس", state)
    assert {"blue", "yellow"}.issubset(set(state["demo_conversation"]["favorite_colors"]))
    reply, state = turn(engine, "طب عارف أنا بحب لون ايه", state)
    assert "أزرق" in reply and "أصفر" in reply

    # The reported current-demo journey: independent searches must not inherit
    # stale blue, groups must be distinct products, and product references must
    # survive image/cart questions.
    state = {}
    reply, state = turn(engine, "أنا بحب لون الأزرق هل عندك حاجه لونها ازرق", state)
    reply, state = turn(engine, "هل عندك تيشرت", state)
    assert len(state["demo_conversation"]["displayed_product_ids"]) == len(set(state["demo_conversation"]["displayed_product_ids"]))
    reply, state = turn(engine, "أنا بحب لون الأبيض اديني ليست خمس حجات عندكم لونها ابيض", state)
    assert len(state["demo_conversation"]["displayed_product_ids"]) == 5
    reply, state = turn(engine, "yellow shirt", state)
    assert "Yellow" in reply and "Shirt" in reply
    reply, state = turn(engine, "عاوزه صوره ليه", state)
    assert "صورة" in reply or "image" in reply.lower()
    reply, state, action = engine.respond("ضيفه للسلة مقاس M", state)
    assert action.get("add_variant_id"), reply
    assert CATALOG.get(action["add_variant_id"]).in_stock
    reply, state = turn(engine, "هل عندك جزم", state)
    assert "حذاء" in reply or "كوتشي" in reply
    reply, state = turn(engine, "لا عاوز حاجه ب ٤٠٠ جنيه", state)
    assert state["demo_conversation"]["active"].get("max_price_egp") == 400
    assert all(CATALOG.get(v).price_egp <= 400 for v in state["demo_conversation"]["displayed_variant_ids"])

    # Variant attributes supplied after a pending cart prompt are applied
    # together; a size-only reply retains the language of the Arabic thread.
    state = {}
    reply, state = turn(engine, "عايز تيشيرت أزرق", state)
    reply, state = turn(engine, "ضيفه للسلة", state)
    reply, state, action = engine.respond("أزرق مقاس L", state)
    assert action.get("add_variant_id") == "V-PVP-020-BLU-L", (reply, action)

    # A numeric shoe size is a continuation of the pending Arabic selection,
    # not a new casual greeting or an English-language reset.
    state = {}
    reply, state = turn(engine, "عايز أشتري جزمة", state)
    reply, state = turn(engine, "ضيفه للسلة", state)
    reply, state, action = engine.respond("39", state)
    assert action.get("add_variant_id") == "V-PVP-008-WHT-39", (reply, action)
    assert ("تمت إضافة" in reply or "هضيف" in reply) and "I’ll add" not in reply, reply
    print("demo conversation e2e test passed")


if __name__ == "__main__":
    main()
