"""Demo clothing catalog. Prices and stock are catalog facts, never model output."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import cycle
from typing import Iterable


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    description: str
    category: str
    color: str
    sizes: list[str]
    price_egp: int
    stock: int
    arabic_aliases: str
    data_source: str = "KAYAN demo catalog — illustrative, not live retailer inventory"

    def searchable_text(self) -> str:
        return " ".join(
            [self.name, self.description, self.category, self.color, " ".join(self.sizes), self.arabic_aliases]
        ).lower()


FAMILIES = [
    ("Essential Cotton Crew T-Shirt", "t-shirt", "Soft combed-cotton crew-neck everyday tee with a regular fit.", ["S", "M", "L", "XL", "XXL"], 399, "تي شيرت قطن كاجوال"),
    ("Relaxed Oversized T-Shirt", "t-shirt", "Heavyweight cotton oversized t-shirt with dropped shoulders.", ["S", "M", "L", "XL"], 499, "تي شيرت اوفر سايز واسع"),
    ("Oxford Button-Down Shirt", "shirt", "Breathable oxford shirt for office and smart-casual outfits.", ["S", "M", "L", "XL", "XXL"], 899, "قميص اوكسفورد رسمي"),
    ("Linen Resort Shirt", "shirt", "Lightweight linen-blend short-sleeve shirt for warm weather.", ["S", "M", "L", "XL"], 799, "قميص كتان صيفي"),
    ("Merino V-Neck Sweater", "knitwear", "Fine-gauge merino-blend V-neck layer with soft hand feel.", ["S", "M", "L", "XL"], 1299, "بلوفر صوف برقبة V"),
    ("Cable Knit Cardigan", "knitwear", "Button-front cable cardigan for layering over tees and shirts.", ["S", "M", "L", "XL"], 1399, "كارديجان حريمي تريكو"),
    ("Classic Denim Jacket", "jacket", "Mid-weight denim trucker jacket with chest pockets and metal buttons.", ["S", "M", "L", "XL"], 1599, "جاكيت جينز"),
    ("Water-Resistant Puffer Jacket", "outerwear", "Insulated puffer with a high collar and light rain protection.", ["S", "M", "L", "XL", "XXL"], 2299, "جاكيت مبطن شتوي"),
    ("Tailored Blazer", "blazer", "Single-breasted stretch blazer for meetings and formal occasions.", ["S", "M", "L", "XL"], 2499, "بليزر رسمي"),
    ("French Terry Hoodie", "hoodie", "Brushed French-terry pullover hoodie with kangaroo pocket.", ["S", "M", "L", "XL", "XXL"], 1099, "هودي سويت شيرت"),
    ("Straight Fit Jeans", "jeans", "Comfort-stretch straight-leg denim with five-pocket construction.", ["30", "32", "34", "36", "38"], 1199, "جينز رجالي مستقيم"),
    ("Wide-Leg Trousers", "trousers", "High-rise wide-leg trousers with a clean drape and side pockets.", ["S", "M", "L", "XL"], 1199, "بنطلون واسع حريمي"),
    ("Slim Chino Pants", "trousers", "Cotton-twill slim chino for business casual styling.", ["30", "32", "34", "36", "38"], 999, "بنطلون تشينو"),
    ("Pleated Midi Skirt", "skirt", "Flowing pleated midi skirt with an elasticated waist.", ["S", "M", "L", "XL"], 949, "جيب ميدي بليسيه"),
    ("Satin Slip Dress", "dress", "Bias-cut satin midi dress for evening wear.", ["XS", "S", "M", "L", "XL"], 1499, "فستان ساتان سهرة"),
    ("Cotton Shirt Dress", "dress", "Belted cotton-poplin shirt dress with a collared neckline.", ["S", "M", "L", "XL"], 1399, "فستان قميص قطن"),
    ("Active Performance Leggings", "activewear", "High-rise squat-proof stretch leggings with a phone pocket.", ["XS", "S", "M", "L", "XL"], 899, "ليجنز رياضي"),
    ("Training Zip Jacket", "activewear", "Lightweight full-zip training jacket with breathable panels.", ["S", "M", "L", "XL"], 1199, "جاكيت رياضي بسوستة"),
    ("Leather-Look Sneakers", "shoes", "Low-profile everyday sneakers with cushioned insole.", ["39", "40", "41", "42", "43", "44"], 1799, "كوتشي سنيكرز"),
    ("Canvas Tote Bag", "accessories", "Reusable structured canvas tote for work and shopping.", ["One Size"], 499, "شنطة توت قماش"),
]
COLORS = [("Black", "اسود"), ("White", "ابيض"), ("Navy", "كحلي"), ("Beige", "بيج"), ("Olive", "زيتي")]


def build_catalog() -> list[Product]:
    products: list[Product] = []
    stock_cycle = cycle([3, 7, 12, 18, 25, 0, 9, 14, 31, 5])
    for family_index, (name, category, description, sizes, base_price, ar) in enumerate(FAMILIES, start=1):
        for color_index, (color, ar_color) in enumerate(COLORS, start=1):
            products.append(Product(
                sku=f"KYN-{family_index:02d}-{color_index:02d}",
                name=f"{color} {name}",
                description=f"{description} Color: {color}.",
                category=category,
                color=color,
                sizes=sizes,
                price_egp=base_price + color_index * 20,
                stock=next(stock_cycle),
                arabic_aliases=f"{ar} {ar_color}",
            ))
    assert len(products) == 100
    return products


def catalog_rows(products: Iterable[Product]) -> list[dict]:
    return [asdict(product) for product in products]


def catalog_dataframe(products: Iterable[Product]):
    """Return the exact catalog connected to ranking in a review/export-friendly shape."""
    import pandas as pd

    rows = []
    for product in products:
        rows.append({
            "Product ID": product.sku,
            "Name": product.name,
            "Category": product.category,
            "Description": product.description,
            "Color": product.color,
            "Sizes": ", ".join(product.sizes),
            "Price (EGP)": product.price_egp,
            "Stock availability": "In stock" if product.stock > 0 else "Out of stock",
            "Stock units": product.stock,
            "Arabic search aliases": product.arabic_aliases,
            "Demo / synthetic": "Yes — illustrative demo catalog, not live retailer inventory",
            "Data source": product.data_source,
        })
    return pd.DataFrame(rows)
