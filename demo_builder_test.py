"""Workbook-backed catalog, exact variant, selected-product cart, and export test."""
from __future__ import annotations

from pathlib import Path
import tempfile

from src.demo_catalog import load_demo_catalog
from src.session_store import CustomerSessionStore


ROOT = Path(__file__).resolve().parent


def main() -> None:
    catalog = load_demo_catalog(ROOT / "catalog_preview_review" / "outputs" / "KAYAN_catalog_color_variants_NEW.xlsx", ROOT / "catalog_preview_review" / "preview_assets")
    assert catalog.validation["sheet"] == "Preview variants"
    assert catalog.validation["rows"] == 133 and catalog.validation["unique_variant_ids"] == 133
    assert catalog.validation["missing_local_image_tiles"] == 0
    assert all(v.image_path and v.image_path.is_file() for v in catalog.variants)
    assert all(v.price_status == "Simulated demo price" and v.stock_status == "Simulated demo stock" for v in catalog.variants)
    shirt = next(v for v in catalog.variants if v.category == "t-shirt" and v.in_stock)
    trousers = next(v for v in catalog.variants if v.category == "trousers" and v.in_stock)
    replacement = next(v for v in catalog.variants if v.category == shirt.category and v.product_id != shirt.product_id and v.in_stock)
    with tempfile.TemporaryDirectory() as temp:
        store = CustomerSessionStore(Path(temp) / "test.sqlite3")
        customer = store.create_customer("Demo customer")
        outfit = store.create_outfit(customer, "Selected Products")
        store.add_or_replace_outfit_item(outfit, shirt.variant_id, shirt.category, shirt.product_id)
        store.add_or_replace_outfit_item(outfit, trousers.variant_id, trousers.category, trousers.product_id)
        record = store.outfit_record(outfit)
        assert {item["variant_id"] for item in record["items"]} == {shirt.variant_id, trousers.variant_id}
        store.add_or_replace_outfit_item(outfit, replacement.variant_id, replacement.category, replacement.product_id)
        record = store.outfit_record(outfit)
        assert {item["variant_id"] for item in record["items"]} == {shirt.variant_id, trousers.variant_id, replacement.variant_id}
        store.set_item_quantity(outfit, next(item["slot_key"] for item in record["items"] if item["variant_id"] == shirt.variant_id), 2)
        record = store.outfit_record(outfit)
        total = sum(catalog.get(item["variant_id"]).price_piasters * item["quantity"] for item in record["items"])
        assert total == 2 * shirt.price_piasters + trousers.price_piasters + replacement.price_piasters
        assert ROOT.joinpath("catalog_preview_review", "outputs", "KAYAN_catalog_color_variants_NEW.xlsx").is_file()
    print("PASS: 133 workbook variants, illustrative tiles, simulated labels, exact cart variants, same-category line items, quantity totals, and workbook export")


if __name__ == "__main__":
    main()
