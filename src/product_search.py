"""Product-level XLM-R relevance ranking for the academic search demo.

Candidate products and their displayed facts are derived only from the selected
workbook catalog. Variants are grouped before inference so a product cannot
occupy several ranks merely because it has several sizes or colors.
"""
from __future__ import annotations

from collections import defaultdict

from .catalog import Product
from .conversation_intelligence import arabic_fold, route


# Keep the existing Product Search relevance formula explicit and shared by
# ranking, display, and regression checks. This is separate from the unrelated
# conversational retrieval blend in SalesAgent.search.
def product_search_score(probabilities: dict[str, float]) -> float:
    return (probabilities["E"] + 0.2 * probabilities["S"])


def _variant_matches_explicit_facets(variant, decision, query: str) -> bool:
    if decision.categories and variant.category.casefold() not in {value.casefold() for value in decision.categories}:
        return False
    variant_colors = {arabic_fold(part) for part in variant.color.split("/")}
    if decision.colors and not variant_colors.intersection({arabic_fold(value) for value in decision.colors}):
        return False
    if decision.size and arabic_fold(variant.size) != arabic_fold(decision.size):
        return False
    if decision.color_tone == "light" and not variant_colors.intersection({"white", "beige", "cream", "tan", "pink", "yellow"}):
        return False
    if decision.color_tone == "dark" and not variant_colors.intersection({"black", "navy", "blue", "olive", "burgundy", "grey", "gray"}):
        return False
    if decision.max_price_egp is not None and variant.price_egp > decision.max_price_egp:
        return False
    pattern_aliases = {
        "stripe": ("striped", "stripe", "مخطط", "مخططة", "خطوط"),
        "color-block": ("color-block", "color block", "مكس", "متعدد الالوان"),
        "floral": ("floral", "زهري", "ورد"),
    }
    folded_pattern = arabic_fold(variant.pattern)
    for canonical, aliases in pattern_aliases.items():
        if any(arabic_fold(alias) in arabic_fold(query) for alias in aliases):
            if canonical == "stripe" and folded_pattern not in {"stripe", "striped"}:
                return False
            if canonical == "color-block" and folded_pattern != "color-block":
                return False
            if canonical == "floral" and folded_pattern != "floral":
                return False
    if decision.occasion:
        occasion_aliases = {
            "beach": ("beach", "resort", "بحر", "شاطئ"),
            "work": ("work", "office", "business", "مقابلة شغل", "الشغل"),
            "wedding": ("wedding", "event", "فرح", "عرس"),
            "travel": ("travel", "trip", "سفر"),
            "sports": ("sports", "gym", "رياضة", "تمرين"),
        }
        aliases = occasion_aliases.get(decision.occasion, (decision.occasion,))
        folded_occasion = arabic_fold(variant.occasion)
        if not any(arabic_fold(alias) in folded_occasion for alias in aliases):
            return False
    return True


def product_candidates(variants, only_in_stock: bool = True, query: str = "") -> list[dict]:
    """Aggregate real workbook variant rows into one classifier product each."""
    grouped: dict[str, list] = defaultdict(list)
    all_grouped: dict[str, list] = defaultdict(list)
    for variant in variants:
        all_grouped[variant.product_id].append(variant)
        if not only_in_stock or variant.in_stock:
            grouped[variant.product_id].append(variant)

    decision = route(query)
    has_facets = bool(decision.categories or decision.colors or decision.size or
                      decision.max_price_egp is not None or decision.occasion or decision.color_tone)
    candidates = []
    for product_id, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row.color.casefold(), row.size.casefold(), row.variant_id))
        representative = rows[0]
        colors = sorted({row.color for row in rows}, key=str.casefold)
        sizes = sorted({row.size for row in rows}, key=lambda value: (not value.isdigit(), value.casefold()))
        prices = sorted({row.price_egp for row in rows})
        # The classifier sees catalog text plus true product facets. It never
        # sees generated labels, and the query/product pair is tokenized by the
        # same XLM-R pair-input path used by predict_detailed().
        description = " ".join(filter(None, (
            representative.description,
            f"Colors: {', '.join(colors)}.",
            f"Sizes: {', '.join(sizes)}.",
            f"Material: {representative.material}.",
            f"Fit: {representative.fit}.",
            f"Pattern: {representative.pattern}.",
            f"Occasion: {representative.occasion}.",
        )))
        facet_matches = ([row for row in all_grouped[product_id]
                          if _variant_matches_explicit_facets(row, decision, query)] if has_facets else [])
        model_product = Product(
            sku=product_id,
            name=representative.name,
            description=description,
            category=representative.category,
            color=" / ".join(colors),
            sizes=sizes,
            price_egp=prices[0],
            stock=sum(row.stock_qty for row in rows),
            arabic_aliases=f"{representative.category} {representative.occasion}",
            data_source="Workbook-backed KAYAN DEMO catalog",
        )
        candidates.append({
            "product_id": product_id,
            "product": model_product,
            "variants": rows,
            "colors": colors,
            "sizes": sizes,
            "prices_egp": prices,
            "stock_qty": sum(row.stock_qty for row in rows),
            "image_path": representative.image_path,
            "representative": representative,
            "facet_match": (bool(facet_matches) if has_facets else None),
            "in_stock_facet_match": (any(row.in_stock for row in facet_matches) if has_facets else None),
        })
    return candidates


def rank_product_candidates(query: str, candidates: list[dict], model) -> list[dict]:
    """Score every product with actual checkpoint probabilities, then stable sort."""
    if not candidates:
        return []
    predictions = model.predict_detailed(query, [item["product"] for item in candidates])
    if len(predictions) != len(candidates):
        raise RuntimeError("XLM-R returned a different number of classifications than products")
    ranked = []
    for candidate, prediction in zip(candidates, predictions):
        probabilities = prediction["probabilities"]
        if set(probabilities) != {"E", "S", "C", "I"}:
            raise ValueError("XLM-R probabilities are missing an ESCI class")
        item = dict(candidate)
        item["prediction"] = prediction
        item["ranking_score"] = product_search_score(probabilities)
        ranked.append(item)
    return sorted(ranked, key=lambda item: (-item["ranking_score"], item["product_id"]))
