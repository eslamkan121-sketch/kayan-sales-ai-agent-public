"""Regression coverage for Arabic display localization without source mutation."""
from pathlib import Path

from src.demo_catalog import load_demo_catalog
from src.demo_conversation import DemoConversationEngine
from src.presentation import (
    format_egp,
    format_piasters,
    localize_category,
    localize_color,
    localize_product_name,
)

ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(
    ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
    ROOT / "catalog_preview_review/preview_assets",
)


def ranker(_query, candidates, limit):
    return candidates[:limit]


def main():
    assert localize_product_name("Relaxed Oversized T-Shirt", True) == "تيشيرت أوفر سايز مريح"
    assert localize_product_name("Essential Crew T-Shirt", True) == "تيشيرت قطن أساسي برقبة دائرية"
    assert localize_product_name("Minimal Leather-Look Sneaker", True) == "كوتشي بتصميم جلد بسيط"
    assert localize_product_name("Relaxed Oversized T-Shirt", False) == "Relaxed Oversized T-Shirt"
    assert localize_product_name("Future Product", True) == "قطعة من الكتالوج (Future Product)"
    assert localize_category("t-shirt", True) == "تيشيرت"
    assert localize_category("shoes", True) == "أحذية"
    assert localize_color("Navy / White", True) == "كحلي / أبيض"
    assert localize_color("Black / White Stripe", True) == "أسود / أبيض مخطط"
    assert localize_color("Yellow / Red", True) == "أصفر / أحمر"
    assert localize_color("Black / Red Color-Block", True) == "أسود / أحمر ألوان متباينة"
    assert localize_color("Navy / White", False) == "Navy / White"
    assert format_egp(356, True) == "٣٥٦ جنيه"
    assert format_egp(1299, True) == "١٬٢٩٩ جنيه"
    assert format_egp(1299, False) == "1,299 EGP"
    assert format_piasters(129900, True) == "١٬٢٩٩ جنيه"

    # Names and prices shown in responses come from the reviewed mapping and
    # formatter; the underlying Product ID, Variant ID, price, and size stay put.
    mixed = next(v for v in CATALOG.variants if v.product_id == "PVP-021" and v.color == "Navy / White" and v.size == "M")
    source_identity = (mixed.product_id, mixed.variant_id, mixed.price_egp, mixed.size, mixed.name, mixed.color)
    assert localize_product_name(mixed.name, True) == "تيشيرت بتصميم ألوان متباينة"
    assert localize_color(mixed.color, True) == "كحلي / أبيض"
    assert (mixed.product_id, mixed.variant_id, mixed.price_egp, mixed.size, mixed.name, mixed.color) == source_identity

    engine = DemoConversationEngine(CATALOG, ranker)
    arabic_reply, arabic_state, _ = engine.respond("عايز تيشيرت أبيض", {})
    arabic_reply, arabic_state, _ = engine.respond("ده بكام؟", arabic_state)
    assert "تيشيرت كاجوال بجيب" in arabic_reply and "جنيه" in arabic_reply
    assert any(digit in arabic_reply for digit in "٠١٢٣٤٥٦٧٨٩")
    english_reply, english_state, _ = engine.respond("I need a white t-shirt", {})
    english_reply, english_state, _ = engine.respond("What is the price?", english_state)
    assert "Sunset Pocket T-Shirt" in english_reply and "449 EGP" in english_reply

    # Formatting never participates in internal arithmetic.
    total_piasters = mixed.price_piasters * 2
    assert total_piasters == mixed.price_piasters * 2
    assert format_piasters(total_piasters, True) == format_egp(mixed.price_egp * 2, True)
    assert mixed.price_egp == 654 and mixed.variant_id == "V-PVP-021-NAV-M"
    print("PASS: reviewed Arabic names/categories/colors, Arabic-Indic prices, English output, mixed colors, identity, and numeric totals")


if __name__ == "__main__":
    main()
