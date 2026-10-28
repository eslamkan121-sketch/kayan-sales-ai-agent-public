"""Hard catalog-filter regressions before the trained ranker is applied."""
from pathlib import Path

from src.demo_catalog import load_demo_catalog

ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(
    ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
    ROOT / "catalog_preview_review/preview_assets",
)


def test_hard_constraints():
    results = CATALOG.search("black cotton t-shirt size M", only_in_stock=True, limit=40)
    assert results
    assert all(item.category == "t-shirt" for item in results)
    assert all(item.color == "Black" for item in results)
    assert all(item.size == "M" for item in results)
    assert all(item.stock_qty > 0 for item in results)

    budget = CATALOG.search("white t-shirt under EGP 500", only_in_stock=True, limit=40)
    assert budget
    assert all(item.color == "White" and item.category == "t-shirt" for item in budget)
    assert all(item.price_egp < 500 for item in budget)

    arabic = CATALOG.search("تيشيرت أبيض مقاس M", only_in_stock=True, limit=40)
    assert arabic
    assert all(item.category == "t-shirt" and item.color == "White" and item.size == "M" for item in arabic)

    missing_color = CATALOG.search("green t-shirt size M", only_in_stock=True, limit=40)
    assert not missing_color

    stripe = CATALOG.search("striped t-shirt", only_in_stock=True, limit=40)
    assert stripe
    assert all(item.category == "t-shirt" and item.pattern == "Stripe" for item in stripe)

    beach = CATALOG.search("beach outfit", only_in_stock=True, limit=40)
    assert beach
    assert all("Beach" in item.occasion for item in beach)

    print("demo catalog hard-constraint tests passed")


if __name__ == "__main__":
    test_hard_constraints()
