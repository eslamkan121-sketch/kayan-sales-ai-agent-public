"""Numeric price provenance, eligibility, and ranking regressions."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from src.demo_catalog import DemoCatalog, DemoCatalogValidationError, load_demo_catalog, parse_price_piasters
from src.demo_conversation import DemoConversationEngine, DemoConversationState
from src.presentation import format_egp

ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx"
ASSETS = ROOT / "catalog_preview_review/preview_assets"


def raises_price_error(value):
    try:
        parse_price_piasters(value)
    except ValueError:
        return
    raise AssertionError(f"invalid price accepted: {value!r}")


def main():
    catalog = load_demo_catalog(WORKBOOK, ASSETS)
    assert catalog.validation["sheet"] == "Preview variants"
    assert catalog.validation["rows"] == 133
    assert catalog.validation["unique_product_ids"] == 24
    assert catalog.validation["unique_variant_ids"] == 133
    assert all(v.price_status == "Simulated demo price" for v in catalog.variants)
    assert min(v.price_piasters for v in catalog.variants) == 39900
    assert all(v.price_piasters > 0 for v in catalog.variants)

    # Parse values without float rounding; null, zero and malformed values are
    # invalid because this schema has no explicit, reviewed free-product flag.
    assert parse_price_piasters(450) == 45000
    assert parse_price_piasters("1,299 EGP") == 129900
    assert parse_price_piasters(Decimal("450.25")) == 45025
    for invalid in (None, float("nan"), 0, "0 EGP", -1, "", "not a price", "12x", "1,29", "NaN"):
        raises_price_error(invalid)
    assert format_egp(Decimal("1299"), True) == "١٬٢٩٩ جنيه"
    assert format_egp(Decimal("450.25"), False) == "450.25 EGP"

    # The workbook loader itself rejects invalid source prices; it never
    # converts a missing/malformed/zero value into a free item.
    with TemporaryDirectory(prefix="kayan-price-input-") as directory:
        for label, bad_value in (("zero", 0), ("missing", None), ("malformed", "not a price")):
            copy = Path(directory) / f"{label}.xlsx"
            source_frame = pd.read_excel(WORKBOOK, sheet_name="Preview variants")
            source_frame["Price"] = source_frame["Price"].astype(object)
            source_frame.loc[0, "Price"] = bad_value
            source_frame.to_excel(copy, sheet_name="Preview variants", index=False)
            try:
                load_demo_catalog(copy, ASSETS)
            except DemoCatalogValidationError:
                pass
            else:
                raise AssertionError(f"loader accepted {label} catalog price")
        formatted = Path(directory) / "thousands.xlsx"
        source_frame = pd.read_excel(WORKBOOK, sheet_name="Preview variants")
        source_frame["Price"] = source_frame["Price"].astype(object)
        source_frame.loc[0, "Price"] = "1,299 EGP"
        source_frame.to_excel(formatted, sheet_name="Preview variants", index=False)
        assert load_demo_catalog(formatted, ASSETS).variants[0].price_piasters == 129900

    # An adversarial same-product fixture proves that a variant passing a
    # range is also the exact variant whose price is displayed. A free/invalid
    # row is not eligible and cannot win the global cheapest sort.
    base = catalog.variants[0]
    custom_variants = [
        replace(base, product_id="TEST-A", variant_id="TEST-A-450", name="Test Tee A", price_piasters=45000),
        replace(base, product_id="TEST-A", variant_id="TEST-A-700", name="Test Tee A", price_piasters=70000),
        replace(base, product_id="TEST-B", variant_id="TEST-B-475", name="Test Tee B", price_piasters=47500),
        replace(base, product_id="TEST-C", variant_id="TEST-C-425", name="Test Tee C", price_piasters=42500),
        replace(base, product_id="TEST-ZERO", variant_id="TEST-ZERO-000", name="Invalid Free Item", price_piasters=0),
        replace(base, product_id="TEST-MISSING", variant_id="TEST-MISSING-NONE", name="Invalid Missing Item", price_piasters=None),
    ]
    custom = DemoCatalog(custom_variants, {"sheet": "test_fixture", "unique_variant_ids": len(custom_variants), "unique_product_ids": 4})
    engine = DemoConversationEngine(custom, lambda _query, variants, limit: variants[:limit])

    state = DemoConversationState()
    reply, stored, _ = engine.respond("منتجات من ٤٠٠ لـ ٥٠٠ جنيه", state.storage())
    shown = stored["demo_conversation"]["displayed_variant_ids"]
    assert shown == ["TEST-A-450", "TEST-B-475", "TEST-C-425"], shown
    evidence = stored["demo_conversation"]["displayed_price_evidence"]
    assert [(item["variant_id"], item["price_piasters"], item["eligible"]) for item in evidence] == [
        ("TEST-A-450", 45000, True), ("TEST-B-475", 47500, True), ("TEST-C-425", 42500, True)]
    assert all(item["reason"] == "within_inclusive_price_range" for item in evidence)
    assert all(format_egp(price, True) in reply for price in (425, 450, 475))
    assert format_egp(700, True) not in reply and "450–700" not in reply

    # A new range replaces prior results. The 450–500 query must not retain
    # the earlier product's 700 EGP variant or stale result IDs.
    reply2, stored2, _ = engine.respond("منتجات من ٤٥٠ لـ ٥٠٠ جنيه", stored)
    shown2 = stored2["demo_conversation"]["displayed_variant_ids"]
    assert shown2 == ["TEST-A-450", "TEST-B-475"] and "TEST-C-425" not in shown2
    assert all(45000 <= item["price_piasters"] <= 50000 for item in stored2["demo_conversation"]["displayed_price_evidence"])
    assert format_egp(700, True) not in reply2

    cheap_reply, cheap_storage, _ = engine.respond("أرخص خمس منتجات في الكتالوج", stored2)
    cheap_state = cheap_storage["demo_conversation"]
    cheap_evidence = cheap_state["displayed_price_evidence"]
    assert cheap_state["displayed_product_ids"] == ["TEST-C", "TEST-A", "TEST-B"]
    assert [row["price_piasters"] for row in cheap_evidence] == [42500, 45000, 47500]
    assert [row["variant_id"] for row in cheap_evidence] == ["TEST-C-425", "TEST-A-450", "TEST-B-475"]
    assert all(row["reason"] == "lowest_valid_eligible_variant" for row in cheap_evidence)
    assert all(format_egp(price, True) in cheap_reply for price in (425, 450, 475))
    assert "Invalid Free Item" not in cheap_reply and format_egp(700, True) not in cheap_reply

    # Test the real active workbook through the same planner: inclusive ranges,
    # five distinct products, cheapest ordering, evidence and exact displayed
    # values all match the catalog's selected Variant IDs.
    live_engine = DemoConversationEngine(catalog, lambda _query, variants, limit: variants[:limit])
    for low, high in ((400, 500), (450, 500)):
        text, state, _ = live_engine.respond(f"عاوز منتجات من {low} لـ {high} جنيه", None)
        payload = state["demo_conversation"]
        variants = [catalog.get(variant_id) for variant_id in payload["displayed_variant_ids"]]
        assert variants and all(v is not None and low * 100 <= v.price_piasters <= high * 100 for v in variants)
        assert len({v.product_id for v in variants}) == len(variants)
        evidence = payload["displayed_price_evidence"]
        assert [(e["product_id"], e["variant_id"], e["price_piasters"]) for e in evidence] == [
            (v.product_id, v.variant_id, v.price_piasters) for v in variants]
        assert all(e["eligible"] and e["reason"] == "within_inclusive_price_range" for e in evidence)
        assert all(format_egp(v.price_egp, True) in text for v in variants)
        print(f"RANGE {low}-{high}: " + ", ".join(f"{v.product_id}/{v.variant_id}={v.price_egp} EGP" for v in variants))

    text, state, _ = live_engine.respond("أرخص خمس منتجات في الكتالوج", None)
    payload = state["demo_conversation"]
    variants = [catalog.get(variant_id) for variant_id in payload["displayed_variant_ids"]]
    assert len(variants) == min(5, len({v.product_id for v in catalog.variants if v.in_stock and v.price_piasters > 0}))
    assert len({v.product_id for v in variants}) == len(variants) == 5
    assert [v.price_piasters for v in variants] == sorted(v.price_piasters for v in variants)
    evidence = payload["displayed_price_evidence"]
    assert [(e["product_id"], e["variant_id"], e["price_piasters"]) for e in evidence] == [
        (v.product_id, v.variant_id, v.price_piasters) for v in variants]
    assert all(format_egp(v.price_egp, True) in text for v in variants)
    print("CHEAPEST FIVE: " + ", ".join(f"{v.product_id}/{v.variant_id}={v.price_egp} EGP" for v in variants))
    print("PASS: strict numeric price parsing, inclusive range eligibility, exact variant-price display, distinct cheapest-product ranking, and active demo catalog provenance")


if __name__ == "__main__":
    main()
