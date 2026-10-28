"""Regression coverage for Arabic queries against the workbook-backed demo catalog."""
from pathlib import Path

from src.agent import normalize_query
from src.demo_catalog import load_demo_catalog


ROOT = Path(__file__).resolve().parent
CATALOG = load_demo_catalog(
    ROOT / "catalog_preview_review" / "outputs" / "KAYAN_catalog_color_variants_NEW.xlsx",
    ROOT / "catalog_preview_review" / "preview_assets",
)


def results(query: str):
    return CATALOG.search(normalize_query(query), only_in_stock=True, limit=30)


def main() -> None:
    blue = results("أنا بحب لون الأزرق هل عندك حاجه لونها ازرق")
    assert blue and any("Blue" in item.color for item in blue)
    red_tee = results("طب عندكم تيشرت احمر")
    assert red_tee and any(item.category == "t-shirt" and "Red" in item.color for item in red_tee)
    white_tee = results("طب عندكم تيشرت ابيض")
    assert white_tee and any(item.category == "t-shirt" and "White" in item.color for item in white_tee)
    beach = results("طب عندك حاجه لي خروجة البحر")
    assert beach and any("Beach" in item.occasion for item in beach)
    striped = results("طب عندكم تيشرت مخطط")
    assert striped and any(item.category == "t-shirt" and item.pattern == "Stripe" for item in striped)
    assert all(item.stock_qty > 0 for item in blue + red_tee + white_tee + beach + striped)
    print("PASS: Arabic colors, t-shirt, striped pattern, and beach occasion retrieve in-stock workbook variants")


if __name__ == "__main__":
    main()
