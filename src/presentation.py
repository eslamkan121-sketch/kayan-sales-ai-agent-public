"""Reviewed display localization for workbook-backed catalog values.

Only presentation text is translated. Product/variant IDs and source values
remain untouched for search, persistence, totals, and reporting.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


ARABIC_PRODUCT_NAMES = {
    "Belted Poplin Shirt Dress": "فستان قميص بوبلين بحزام",
    "Cable Knit Crew Sweater": "بلوفر تريكو بضفائر برقبة دائرية",
    "Classic Pique Polo": "تيشيرت بولو كلاسيك",
    "Coastal Linen Camp Shirt": "قميص كتان صيفي",
    "Color-Block Crew T-Shirt": "تيشيرت بتصميم ألوان متباينة",
    "Essential Crew T-Shirt": "تيشيرت قطن أساسي برقبة دائرية",
    "Layered Cotton Overshirt": "قميص قطني واسع للّبس فوق طبقة",
    "Lightweight Field Jacket": "جاكيت خفيف عملي",
    "Lightweight Sundress": "فستان صيفي خفيف",
    "Longline Performance T-Shirt": "تيشيرت رياضي طويل",
    "Merino Rib Scarf": "وشاح ميرينو مضلع",
    "Minimal Crossbody Bag": "شنطة كروس بسيطة",
    "Minimal Leather-Look Sneaker": "كوتشي بتصميم جلد بسيط",
    "Performance Zip Top": "بلوزة رياضية بسوستة",
    "Pleated Midi Skirt": "جيبة ميدي بكسرات",
    "Pocket Polo T-Shirt": "تيشيرت بولو بجيب",
    "Relaxed Oversized T-Shirt": "تيشيرت أوفر سايز مريح",
    "Ribbed Knit Cardigan": "كارديجان تريكو مضلع",
    "Straight Selvedge Jeans": "جينز بقصة مستقيمة",
    "Striped Knit T-Shirt": "تيشيرت تريكو مخطط",
    "Structured Canvas Tote": "شنطة توت كانفاس متينة",
    "Structured Evening Blazer": "بليزر سهرة بقصة مرتبة",
    "Sunset Pocket T-Shirt": "تيشيرت كاجوال بجيب",
    "Utility Cargo Trousers": "بنطلون كارغو عملي",
}

ARABIC_CATEGORY_NAMES = {
    "accessories": "إكسسوارات",
    "activewear": "ملابس رياضية",
    "blazer": "بليزر",
    "dress": "فستان",
    "jacket": "جاكيت",
    "jeans": "جينز",
    "knitwear": "ملابس تريكو",
    "shirt": "قميص",
    "shoes": "أحذية",
    "skirt": "جيبة",
    "t-shirt": "تيشيرت",
    "trousers": "بنطلون",
}

ARABIC_COLOR_NAMES = {
    "beige": "بيج",
    "black": "أسود",
    "blue": "أزرق",
    "burgundy": "نبيتي",
    "cream": "كريمي",
    "grey": "رمادي",
    "gray": "رمادي",
    "green": "أخضر",
    "navy": "كحلي",
    "olive": "زيتوني",
    "pink": "وردي",
    "red": "أحمر",
    "tan": "جملي",
    "white": "أبيض",
    "yellow": "أصفر",
}

_ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def localize_product_name(name: str, arabic: bool) -> str:
    if not arabic:
        return name
    # Unknown source names are shown verbatim in parentheses as a safe identity
    # fallback; they are never dynamically translated or used for lookup.
    return ARABIC_PRODUCT_NAMES.get(name, f"قطعة من الكتالوج ({name})")


def localize_category(category: str, arabic: bool) -> str:
    if not arabic:
        return category
    return ARABIC_CATEGORY_NAMES.get(category.strip().lower(), f"فئة {category}")


def localize_color(color: str, arabic: bool) -> str:
    if not arabic:
        return color
    # Workbook values use a single color or a structured compound such as
    # "Navy / White", "Black / White Stripe", or "Yellow / Red".
    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        return ARABIC_COLOR_NAMES.get(token.casefold(), token)

    result = re.sub(r"\b[A-Za-z]+\b", replace_token, color)
    result = re.sub(r"\bColor-Block\b", "ألوان متباينة", result, flags=re.I)
    result = re.sub(r"\bStripe\b", "مخطط", result, flags=re.I)
    return result


def format_egp(amount: int | float | Decimal, arabic: bool) -> str:
    """Format a verified numeric EGP amount without rounding away catalog cents."""
    value = amount if isinstance(amount, Decimal) else Decimal(str(amount))
    numeric = f"{value:,.2f}".rstrip("0").rstrip(".")
    if not arabic:
        return f"{numeric} EGP"
    return f"{numeric.replace(',', '٬').translate(_ARABIC_DIGITS)} جنيه"


def format_piasters(piasters: int, arabic: bool) -> str:
    # Catalog prices are whole EGP values stored internally as integer piasters.
    return format_egp(int(piasters) // 100, arabic)


def format_integer(value: int, arabic: bool) -> str:
    text=f"{int(value):,}"
    return text.replace(",", "٬").translate(_ARABIC_DIGITS) if arabic else text
