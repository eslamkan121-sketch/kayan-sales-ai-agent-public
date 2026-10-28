"""Validated loader for the approved workbook-backed KAYAN demo catalog."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

import pandas as pd

from .conversation_intelligence import arabic_fold, route


REQUIRED_COLUMNS = {
    "Product ID", "Variant ID", "Product name", "Category", "Color", "Size",
    "Stock qty", "Price", "Currency", "Material", "Fit", "Pattern", "Occasion",
    "Image path / URL", "Image provenance", "Product field status", "Price status",
    "Stock status", "Record status",
}


class DemoCatalogValidationError(ValueError):
    pass


def parse_price_piasters(value: object) -> int:
    """Parse a positive catalog price exactly; zero is unsupported without an explicit free flag."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        raise ValueError("missing_price")
    text = str(value).strip()
    if not text:
        raise ValueError("missing_price")
    text = re.sub(r"\s*(?:EGP|جنيه)\s*", "", text, flags=re.I)
    # Accept standard thousands separators, but reject malformed grouping.
    if "," in text:
        if not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?", text):
            raise ValueError("malformed_price")
        text = text.replace(",", "")
    try:
        amount = Decimal(text)
    except (InvalidOperation, ValueError):
        raise ValueError("malformed_price") from None
    if not amount.is_finite():
        raise ValueError("malformed_price")
    if amount <= 0:
        raise ValueError("non_positive_price_without_free_flag")
    scaled = amount * 100
    if scaled != scaled.to_integral_value():
        raise ValueError("price_has_more_than_two_decimals")
    return int(scaled)


def valid_price_piasters(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class DemoVariant:
    variant_id: str
    product_id: str
    name: str
    description: str
    category: str
    color: str
    size: str
    stock_qty: int
    price_piasters: int
    currency: str
    material: str
    fit: str
    pattern: str
    occasion: str
    image_path: Path | None
    image_provenance: str
    product_status: str
    price_status: str
    stock_status: str
    record_status: str
    season: str = ""
    sleeve_length: str = ""

    @property
    def price_egp(self) -> Decimal:
        return Decimal(self.price_piasters) / Decimal(100)

    @property
    def in_stock(self) -> bool:
        return self.stock_qty > 0


class DemoCatalog:
    def __init__(self, variants: list[DemoVariant], validation: dict):
        self.variants = variants
        self.validation = validation
        self.by_id = {variant.variant_id: variant for variant in variants}

    def get(self, variant_id: str) -> DemoVariant | None:
        return self.by_id.get(variant_id)

    def options(self) -> list[tuple[str, str]]:
        return [
            (f"{v.name} — {v.color}/{v.size} — {v.price_egp:,} {v.currency} — demo stock {v.stock_qty}", v.variant_id)
            for v in self.variants
        ]

    def search(self, query: str, only_in_stock: bool = True, limit: int = 12) -> list[DemoVariant]:
        terms = [term for term in re.findall(r"[\w-]+", (query or "").lower()) if term]
        candidates = [v for v in self.variants if not only_in_stock or v.in_stock]
        # Explicit customer constraints are hard filters. XLM-R is only called
        # after this retrieval boundary, so its relevance score cannot promote
        # a wrong color, category, size, budget, pattern, or occasion.
        decision = route(query)
        if decision.categories:
            wanted = {arabic_fold(value) for value in decision.categories}
            candidates = [v for v in candidates if arabic_fold(v.category) in wanted]
        if decision.colors:
            wanted_colors = {arabic_fold(value) for value in decision.colors}
            if len(wanted_colors) == 1 and not decision.mixed_color_request:
                wanted_color = next(iter(wanted_colors))
                candidates = [v for v in candidates if arabic_fold(v.color) == wanted_color]
            else:
                candidates = [v for v in candidates if wanted_colors.intersection({arabic_fold(part) for part in v.color.split("/")})]
        if decision.size:
            wanted_size = arabic_fold(decision.size)
            candidates = [v for v in candidates if arabic_fold(v.size) == wanted_size]
        candidates = [v for v in candidates if valid_price_piasters(v.price_piasters)]
        if decision.max_price_egp is not None:
            candidates = [v for v in candidates if v.price_piasters < (decision.max_price_egp + 1) * 100]
        if decision.color_tone == "light":
            light = {"white", "beige", "cream", "tan", "pink", "yellow"}
            candidates = [v for v in candidates if any(arabic_fold(part) in light for part in v.color.split("/"))]
        if decision.color_tone == "dark":
            dark = {"black", "navy", "blue", "olive", "burgundy", "grey"}
            candidates = [v for v in candidates if any(arabic_fold(part) in dark for part in v.color.split("/"))]
        pattern_aliases = {
            "stripe": ("striped", "stripe", "مخطط", "مخططة", "خطوط"),
            "color-block": ("color-block", "color block", "مكس", "متعدد الالوان"),
            "floral": ("floral", "زهري", "ورد"),
        }
        requested_patterns = [key for key, aliases in pattern_aliases.items()
                              if any(arabic_fold(alias) in arabic_fold(query) for alias in aliases)]
        if requested_patterns:
            wanted_patterns = {arabic_fold(value) for value in requested_patterns}
            candidates = [v for v in candidates if arabic_fold(v.pattern) in wanted_patterns]
        if decision.occasion:
            occasion_aliases = {
                "beach": ("beach", "resort", "بحر", "شاطئ"),
                "work": ("work", "office", "business", "مقابلة شغل", "الشغل"),
                "wedding": ("wedding", "event", "فرح", "عرس"),
                "travel": ("travel", "trip", "سفر"),
                "sports": ("sports", "gym", "رياضة", "تمرين"),
            }
            aliases = occasion_aliases.get(decision.occasion, (decision.occasion,))
            candidates = [v for v in candidates if any(arabic_fold(alias) in arabic_fold(v.occasion) for alias in aliases)]
        if terms:
            def score(variant: DemoVariant) -> int:
                haystack = " ".join((variant.name, variant.description, variant.category, variant.color, variant.size, variant.material, variant.fit, variant.pattern, variant.occasion)).lower()
                return sum(term in haystack for term in terms)
            candidates = [variant for variant in candidates if score(variant)]
            candidates.sort(key=lambda variant: (-score(variant), variant.price_piasters, variant.variant_id))
        return candidates[:limit]


def _color_file(color: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", color.strip()).strip("-")


def load_demo_catalog(workbook_path: str | Path, asset_root: str | Path) -> DemoCatalog:
    workbook_path, asset_root = Path(workbook_path), Path(asset_root)
    if not workbook_path.is_file():
        raise DemoCatalogValidationError(f"Workbook not found: {workbook_path}")
    try:
        frame = pd.read_excel(workbook_path, sheet_name="Preview variants")
    except Exception as exc:
        raise DemoCatalogValidationError(f"Could not read Preview variants sheet: {exc}") from exc
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise DemoCatalogValidationError(f"Preview variants is missing required columns: {', '.join(missing_columns)}")
    if frame.empty:
        raise DemoCatalogValidationError("Preview variants has no variant rows")
    for field in ("Product ID", "Variant ID", "Product name", "Category", "Color", "Size", "Currency"):
        if frame[field].isna().any() or (frame[field].astype(str).str.strip() == "").any():
            raise DemoCatalogValidationError(f"Preview variants has missing required values in {field}")
    if frame["Variant ID"].duplicated().any():
        duplicates = frame.loc[frame["Variant ID"].duplicated(), "Variant ID"].astype(str).tolist()
        raise DemoCatalogValidationError(f"Duplicate Variant IDs: {duplicates[:5]}")
    numeric_stock = pd.to_numeric(frame["Stock qty"], errors="coerce")
    if numeric_stock.isna().any() or (numeric_stock < 0).any():
        raise DemoCatalogValidationError("Preview variants has invalid Stock qty values")
    parsed_prices = []
    for row_number, value in enumerate(frame["Price"], start=2):
        try:
            parsed_prices.append(parse_price_piasters(value))
        except ValueError as exc:
            raise DemoCatalogValidationError(f"Preview variants has invalid Price at row {row_number}: {exc}") from exc
    variants: list[DemoVariant] = []
    missing_tiles = 0
    for row_index, (_, row) in enumerate(frame.iterrows()):
        product_id, color = str(row["Product ID"]).strip(), str(row["Color"]).strip()
        tile = asset_root / "variant_tiles" / f"{product_id}-{_color_file(color)}.png"
        image_path = tile if tile.is_file() else None
        missing_tiles += image_path is None
        variants.append(DemoVariant(
            variant_id=str(row["Variant ID"]).strip(), product_id=product_id,
            name=str(row["Product name"]).strip(),
            description=f"{row['Material']} {row['Fit']} {row['Pattern']} {row['Occasion']}".strip(),
            category=str(row["Category"]).strip(), color=color, size=str(row["Size"]).strip(),
            stock_qty=int(row["Stock qty"]), price_piasters=parsed_prices[row_index],
            currency=str(row["Currency"]).strip(), material=str(row["Material"]).strip(),
            fit=str(row["Fit"]).strip(), pattern=str(row["Pattern"]).strip(), occasion=str(row["Occasion"]).strip(),
            image_path=image_path, image_provenance=str(row["Image provenance"]),
            product_status=str(row["Product field status"]), price_status=str(row["Price status"]),
            stock_status=str(row["Stock status"]), record_status=str(row["Record status"]),
            season=str(row.get("Season", "") if pd.notna(row.get("Season", "")) else "").strip(),
            sleeve_length=str(row.get("Sleeve / length", "") if pd.notna(row.get("Sleeve / length", "")) else "").strip(),
        ))
    return DemoCatalog(variants, {
        "sheet": "Preview variants", "rows": len(variants), "unique_product_ids": int(frame["Product ID"].nunique()),
        "unique_variant_ids": len(variants), "duplicate_product_ids_expected": int(frame["Product ID"].duplicated().sum()),
        "missing_local_image_tiles": missing_tiles, "workbook": str(workbook_path),
    })
