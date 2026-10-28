"""Conversation planner grounded in the workbook-backed KAYAN demo catalog.

This module deliberately keeps durable preferences, the active shopping task and
the products displayed in the last answer as separate state.  It does not call a
language model to manufacture catalog facts: every displayed price, colour, size
and stock value is resolved from ``DemoVariant`` first.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
from typing import Callable

from .conversation_intelligence import CATEGORY_ALIASES, COLOR_ALIASES, arabic_fold, extract_aliases, has_signal, route
from .demo_catalog import DemoCatalog, DemoVariant, valid_price_piasters
from .presentation import ARABIC_CATEGORY_NAMES, ARABIC_COLOR_NAMES, ARABIC_PRODUCT_NAMES, format_egp, localize_category, localize_color, localize_product_name


LOG = logging.getLogger("kayan.demo_conversation")

def _display_name(variant: DemoVariant, arabic: bool) -> str:
    return localize_product_name(variant.name, arabic)


def _display_color(value: str, arabic: bool) -> str:
    return localize_color(value, arabic)


def _setup_log(path: Path | None) -> None:
    if not path or LOG.handlers:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)


def _arabic(message: str) -> bool:
    return bool(re.search(r"[\u0600-\u06ff]", message or ""))


def _digits(value: str) -> str:
    return (value or "").translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))


def _category_values(message: str) -> list[str]:
    return [key for key, forms in CATEGORY_ALIASES.items() if has_signal(message, *forms)]


def _size(message: str) -> str | None:
    folded = arabic_fold(message)
    match = re.search(r"(?:مقاس|size)\s*(xxl|xl|xs|[sml]|\d{2})\b", folded, re.I)
    if match:
        return match.group(1).upper()
    # Egyptian spoken size variants are only interpreted when the word is in a
    # size context, so ordinary prose cannot accidentally become a size filter.
    if has_signal(message, "ميديم", "medium"):
        return "M"
    if has_signal(message, "لارج", "large"):
        return "L"
    if has_signal(message, "سمول", "small"):
        return "S"
    return None


def _budget(message: str) -> int | None:
    text = _digits(arabic_fold(message))
    match = re.search(r"(?:اقل\s+من|تحت|under|below|less\s+than)\s*(?:egp|جنيه)?\s*(\d{2,6})", text, re.I)
    if not match:
        maximum = re.search(r"(?:بحد\s+اقصى|بحد\s+أقصى|حد\s+اقصى|max(?:imum)?|up\s+to|ب)\s*(?:egp|جنيه)?\s*(\d{2,6})", text, re.I)
        return int(maximum.group(1)) if maximum else None
    return max(0, int(match.group(1)) - 1)  # strict less-than constraint


def _price_range(message: str) -> tuple[int, int] | None:
    """Parse inclusive price ranges without treating either bound as a ceiling-only query."""
    text = _digits(arabic_fold(message))
    patterns = (
        r"(?:من|from)\s*(?:egp\s*|جنيه\s*)?(\d{2,6})\s*(?:(?:egp|جنيه)\s*)?(?:إلى|الى|الي|لـ?|لحد|حتى|to|through|until|[-–])\s*(?:egp\s*|جنيه\s*)?(\d{2,6})",
        r"\bbetween\s*(?:egp\s*)?(\d{2,6})\s*(?:and|to|[-–])\s*(?:egp\s*)?(\d{2,6})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            return (low, high) if low <= high else (high, low)
    return None


def _ordinal(message: str) -> int | None:
    text = arabic_fold(message)
    if has_signal(text, "اخر واحد", "اخر اختيار", "last one", "last option"):
        return -1
    if has_signal(text, "التاني", "الثاني", "second", "2"):
        return 1
    if has_signal(text, "التالت", "الثالث", "third", "3"):
        return 2
    if has_signal(text, "الاول", "أول اختيار", "اول اختيار", "اول واحد", "first", "1"):
        return 0
    return None


def _requested_count(message: str) -> int | None:
    text = _digits(arabic_fold(message))
    match = re.search(r"(?:list|ليست|قائمة|جدول|table)\s*(\d+)", text, re.I)
    if match:
        return max(1, min(10, int(match.group(1))))
    if has_signal(message, "خمس", "خمسة", "five") and has_signal(message, "ليست", "قائمة", "جدول", "list", "table", "حاجات", "منتجات", "هات", "اديني", "اعرض", "show"):
        return 5
    return None


def _is_global_cheapest_request(message: str) -> bool:
    """A catalog-wide price sort, distinct from a relative cheaper follow-up."""
    folded = arabic_fold(message)
    cheap_signal = has_signal(message, "ارخص", "أرخص", "cheapest", "lowest price", "أقل سعر") or bool(
        re.search(r"(?:^|\s)(?:ل)?(?:اقل|ارخص)(?:\s|$)", folded, re.I)
    )
    if not cheap_signal:
        return False
    # Do not use fuzzy short-token matching here: e.g. "دي" can match the
    # suffix of "اديني" and incorrectly turn a global list into a comparison.
    relative_reference = bool(re.search(
        r"(?:من\s*(?:ده|دي|هذا|هذه|المنتج|القطعة)|(?:ده|دي)\s*(?:المنتج|القطعة|الاختيار)|"
        r"منه|منها|\bthan\s+(?:this|that|the one)\b|reference product)", folded, re.I
    ))
    return not relative_reference and (
        _requested_count(message) is not None
        or has_signal(message, "عندكم", "عندك", "الكتالوج", "catalog", "products", "حاجات", "منتجات", "كل المنتجات")
    )


def _is_yes(message: str) -> bool:
    return arabic_fold(message) in {"ايوه", "ايوة", "اه", "أه", "نعم", "yes", "yes please", "ايوه ياريت", "اه ياريت"}


def _is_price_question(message: str) -> bool:
    return has_signal(message, "سعر", "السعر", "اسعار", "أسعار", "الأسعار", "اسعارهم", "أسعارهم", "بكام", "كام", "price", "prices", "their prices", "cost")


def _is_sizes_question(message: str) -> bool:
    return has_signal(message, "مقاسات", "المقاسات", "مقاساتهم", "sizes", "their sizes", "size available")


def _is_colors_question(message: str) -> bool:
    return has_signal(message, "الوان ايه", "ألوان ايه", "ايه الالوان", "ايه الألوان", "ما الالوان", "اللون", "ألوانه إيه", "الوانه ايه", "ايه ألوانه", "الوان متاحة", "الألوان المتاحة", "ايه الوان متاحه", "الوانهم", "الألوان", "الوان", "their colors", "colors", "what colors", "available colors")


def _is_image_question(message: str) -> bool:
    return has_signal(message, "صورة", "صورته", "صورتها", "صور", "image", "photo", "picture")


def _is_plural_image_question(message: str) -> bool:
    folded=arabic_fold(message)
    return bool(re.search(r"صور\s*(?:هم|ها|ليهم|لهم|ليها|لها)|صور\s+(?:المنتجات|الاختيارات|كلهم)|\b(?:their|all\s+their)\s+(?:photos|images)\b|\bshow\s+all\s+images\b",folded))


def _is_plural_list_followup(message: str, state: "DemoConversationState", product_named: bool = False) -> bool:
    """Whether a detail question refers to the full previously shown list."""
    if len(state.displayed_variant_ids) < 2 or _ordinal(message) is not None:
        return False
    if product_named:
        return False
    # A newly specified list/filter is a new search, even if it mentions plural
    # details such as images, prices and sizes. Do not consume it as a follow-up
    # to the previous product list.
    if (_requested_count(message) is not None or _budget(message) is not None or _price_range(message) is not None
            or _is_global_cheapest_request(message) or _category_values(message)
            or (extract_aliases(message, COLOR_ALIASES) and _explicit_product_request(message))):
        return False
    folded = arabic_fold(message)
    if re.search(r"(?:^|\s)(?:ده|دي|دا|هذا|هذه|it|this|that|this one|that one)(?:$|\s)", folded, re.I):
        return False
    explicit_plural = bool(re.search(
        r"(?:اسعارهم|أسعارهم|مقاساتهم|الوانهم|صورهم|صورها|صور\s*(?:هم|ها|ليهم|لهم|ليها|لها)|صور\s+(?:المنتجات|الاختيارات|كلهم)|"
        r"their\s+(?:prices|sizes|colors|photos|images)|prices\s+of\s+(?:these|them)|"
        r"sizes\s+of\s+(?:these|them)|all\s+(?:of\s+)?(?:these|them)|(?:دول|دول\s+كلهم))",
        folded, re.I))
    # A combined details request after a list is plural by context even if the
    # customer omits an explicit plural pronoun (e.g. “المقاسات والأسعار”).
    combined_details = sum((
        _is_price_question(message), _is_sizes_question(message),
        _is_colors_question(message), _is_image_question(message),
    )) >= 2
    return explicit_plural or combined_details


def _pending_size(message: str) -> str | None:
    size = _size(message)
    if size:
        return size
    folded = _digits(message).strip()
    match = re.fullmatch(r"(?:المقاس\s*)?(XXL|XL|XS|[SML]|\d{2})(?:\s*(?:و|and|,|/)?\s*.*)?", folded, re.I)
    return match.group(1).upper() if match else None


def _is_add_request(message: str) -> bool:
    return has_signal(message, "ضيف", "اضيف", "أضيف", "السلة", "سله", "cart", "selected products", "المنتجات المختارة")


def _explicit_product_request(message: str) -> bool:
    return bool(_category_values(message) or has_signal(message, "عندكم", "عندك", "عايز", "عاوز", "محتاج", "اديني", "وريني", "هات", "جدول", "ليست", "قائمة", "show", "list", "table", "looking"))


@dataclass
class DemoConversationState:
    language: str = "en"
    favorite_colors: list[str] = field(default_factory=list)
    style_preferences: list[str] = field(default_factory=list)
    active: dict = field(default_factory=dict)
    displayed_product_ids: list[str] = field(default_factory=list)
    displayed_variant_ids: list[str] = field(default_factory=list)
    displayed_price_evidence: list[dict] = field(default_factory=list)
    display_fields: list[str] = field(default_factory=lambda: ["name"])
    focused_product_id: str | None = None
    focused_variant_id: str | None = None
    selected_color: str | None = None
    selected_size: str | None = None
    pending_action: dict = field(default_factory=dict)

    @classmethod
    def from_storage(cls, value: dict | None) -> "DemoConversationState":
        payload = (value or {}).get("demo_conversation", value or {})
        fields = {name: payload.get(name, default) for name, default in {
            "language": "en",
            "favorite_colors": [], "style_preferences": [], "active": {},
            "displayed_product_ids": [], "displayed_variant_ids": [], "displayed_price_evidence": [], "display_fields": ["name"],
            "focused_product_id": None, "focused_variant_id": None, "selected_color": None,
            "selected_size": None, "pending_action": {},
        }.items()}
        # Existing customer rows may hold the legacy agent state.  Treat every
        # incompatible field as absent instead of allowing old strings to break
        # a restored customer session.
        for name in ("favorite_colors", "style_preferences", "displayed_product_ids", "displayed_variant_ids", "displayed_price_evidence", "display_fields"):
            if not isinstance(fields[name], list):
                fields[name] = []
        for name in ("active", "pending_action"):
            if not isinstance(fields[name], dict):
                fields[name] = {}
        if fields["language"] not in {"ar", "en"}:
            fields["language"] = "en"
        for name in ("focused_product_id", "focused_variant_id", "selected_color", "selected_size"):
            if fields[name] is not None and not isinstance(fields[name], str):
                fields[name] = None
        return cls(**fields)

    def storage(self) -> dict:
        return {"demo_conversation": asdict(self)}


class DemoConversationEngine:
    """Route a customer turn and return a catalog-grounded response plan."""

    def __init__(self, catalog: DemoCatalog, ranker: Callable[[str, list[DemoVariant], int], list[DemoVariant]], log_path: Path | None = None, qwen: Callable | None = None):
        self.catalog, self.ranker, self.qwen = catalog, ranker, qwen
        _setup_log(log_path)

    @staticmethod
    def product_name(variant: DemoVariant, arabic: bool) -> str:
        return _display_name(variant, arabic)

    @staticmethod
    def color_name(color: str, arabic: bool) -> str:
        return _display_color(color, arabic)

    def _phrase(self, message: str, approved: str, arabic: bool) -> tuple[str, str]:
        """Use Qwen only to phrase fact-free replies; never generate catalog facts."""
        if not self.qwen:
            return approved, "not_invoked:no_qwen_callback"
        try:
            answer, used, reason = self.qwen(message, approved, arabic)
            return (answer if used else approved), ("used" if used else f"fallback:{reason}")
        except Exception as exc:
            return approved, f"fallback:{type(exc).__name__}"

    def _log(self, message: str, intent: str, state: DemoConversationState, candidate_count: int, qwen: str, rejection: str = "") -> None:
        # No raw customer messages in developer logs.  The digest lets an
        # engineer correlate one browser turn with a log entry without storing
        # text that can include personal details.
        event = {
            "message_sha256_12": sha256(message.encode("utf-8")).hexdigest()[:12],
            "message_length": len(message), "intent": intent, "parsed_intent": intent,
            "active": state.active, "focused_product_id": state.focused_product_id,
            "displayed_product_ids": state.displayed_product_ids,
            "displayed_variant_ids": state.displayed_variant_ids,
            "selected_image_variant_ids": state.displayed_variant_ids if _is_image_question(message) else [],
            "candidate_count": candidate_count,
            "rendered_response_type": ("compact_product_cards" if state.displayed_variant_ids and
                                       (_is_image_question(message) or intent == "product_list_followup" or candidate_count)
                                       else "plain_text"),
            "plain_text_fallback": True,
            "qwen": qwen, "rejection": rejection, "catalog": "workbook_demo",
            "catalog_sheet": self.catalog.validation.get("sheet"),
            "catalog_variant_count": self.catalog.validation.get("unique_variant_ids"),
            "catalog_product_count": self.catalog.validation.get("unique_product_ids"),
        }
        LOG.info(json.dumps(event, ensure_ascii=False, sort_keys=True))

    def _variants(self, constraints: dict) -> list[DemoVariant]:
        # Prices appear in customer product cards. Fail closed when a row has
        # no valid positive numeric catalog price, rather than ranking it as free.
        candidates = [v for v in self.catalog.variants if v.in_stock and valid_price_piasters(v.price_piasters)]
        categories = constraints.get("categories") or ([] if not constraints.get("category") else [constraints["category"]])
        if categories:
            candidates = [v for v in candidates if v.category.lower() in {x.lower() for x in categories}]
        color = constraints.get("color")
        if color:
            candidates = [v for v in candidates if color.lower() in v.color.lower()]
        pattern = constraints.get("pattern")
        if pattern:
            candidates = [v for v in candidates if pattern.lower() in v.pattern.lower()]
        # Occasion is a soft catalog-suitability signal, not a literal product
        # attribute filter. It is scored in _occasion_candidates so a wedding
        # request can use verified Event/Smart-casual items even when no record
        # is literally tagged "Wedding".
        size = constraints.get("size")
        if size:
            candidates = [v for v in candidates if v.size.upper() == size.upper()]
        budget = constraints.get("max_price_egp")
        if budget is not None:
            candidates = [v for v in candidates if v.price_piasters <= int(budget) * 100]
        price_range = constraints.get("price_range_egp")
        if price_range:
            low, high = map(int, price_range)
            candidates = [v for v in candidates if low * 100 <= v.price_piasters <= high * 100]
        mixed = constraints.get("mixed")
        if mixed:
            candidates = [v for v in candidates if "/" in v.color or "color-block" in v.pattern.lower()]
        return candidates

    def _log_search_trace(self, message: str, constraints: dict, retrieved: list[DemoVariant],
                          eligible: list[DemoVariant], state: DemoConversationState,
                          response: str, occasion_rejected: list[DemoVariant] | None = None) -> None:
        mode = ("new_global_price_sort" if constraints.get("sort_by") == "price_asc" else
                "new_price_range_search" if constraints.get("price_range_egp") else
                "new_occasion_search" if constraints.get("occasion") else "new_product_search")
        rejected: dict[str, list[str]] = {}
        for variant in self.catalog.variants:
            reasons = []
            if not variant.in_stock:
                reasons.append("no_positive_demo_stock")
            if not valid_price_piasters(variant.price_piasters):
                reasons.append("invalid_or_non_positive_price")
            categories = constraints.get("categories") or ([] if not constraints.get("category") else [constraints["category"]])
            if categories and variant.category.lower() not in {str(value).lower() for value in categories}:
                reasons.append("category_mismatch")
            if constraints.get("color") and str(constraints["color"]).lower() not in variant.color.lower():
                reasons.append("color_mismatch")
            if constraints.get("pattern") and str(constraints["pattern"]).lower() not in variant.pattern.lower():
                reasons.append("pattern_mismatch")
            if constraints.get("size") and variant.size.upper() != str(constraints["size"]).upper():
                reasons.append("size_mismatch")
            if (valid_price_piasters(variant.price_piasters) and constraints.get("max_price_egp") is not None
                    and variant.price_piasters > int(constraints["max_price_egp"]) * 100):
                reasons.append("above_max_price")
            price_range = constraints.get("price_range_egp")
            if (valid_price_piasters(variant.price_piasters) and price_range and
                    not int(price_range[0]) * 100 <= variant.price_piasters <= int(price_range[1]) * 100):
                reasons.append("outside_price_range")
            if constraints.get("mixed") and not ("/" in variant.color or "color-block" in variant.pattern.lower()):
                reasons.append("not_mixed_color")
            if reasons:
                rejected[variant.variant_id] = reasons
        event = {
            "event": "search_trace", "message_sha256_12": sha256(message.encode("utf-8")).hexdigest()[:12],
            "parsed_intent": route(message, bool(state.active)).kind,
            "search_decision": mode, "active_filters": constraints,
            "pre_occasion_candidate_product_ids": list(dict.fromkeys(v.product_id for v in retrieved)),
            "candidate_product_ids": list(dict.fromkeys(v.product_id for v in eligible)),
            "candidate_variant_ids": [v.variant_id for v in eligible],
            "occasion_rejected_product_ids": list(dict.fromkeys(v.product_id for v in (occasion_rejected or []))),
            "constraint_rejection_reasons": rejected,
            "displayed_product_ids": state.displayed_product_ids,
            "displayed_variant_ids": state.displayed_variant_ids,
            "price_evidence": state.displayed_price_evidence,
            "selected_image_variant_ids": (state.displayed_variant_ids if _is_image_question(message) else []),
            "rendered_response_type": "compact_product_cards" if eligible else "plain_text",
            "plain_text_present": bool(response.strip()),
            "catalog": "workbook_demo",
        }
        LOG.info(json.dumps(event, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _occasion_score(variant: DemoVariant, occasion: str) -> int:
        tags = {arabic_fold(part.strip()) for part in re.split(r"[;,/]", variant.occasion) if part.strip()}
        category = arabic_fold(variant.category)
        fit = arabic_fold(variant.fit)
        material = arabic_fold(variant.material)
        name = arabic_fold(variant.name)
        seasons = {arabic_fold(part.strip()) for part in re.split(r"[;,/]", variant.season) if part.strip()}
        if occasion == "wedding":
            score = 6 * bool(tags & {"event", "formal", "smart casual", "wedding"})
            score += 2 * (category in {"blazer", "dress", "shirt", "trousers", "skirt"})
            score += 2 * (fit in {"tailored", "structured"})
            score += 1 * ("evening" in name)
            return score if score >= 3 else 0
        if occasion == "formal":
            score = 6 * bool(tags & {"event", "formal", "smart casual", "work"})
            score += 2 * (category in {"blazer", "dress", "shirt", "trousers", "skirt"})
            score += 2 * (fit in {"tailored", "structured"})
            return score if score >= 3 else 0
        if occasion == "beach":
            return 6 if tags & {"beach", "resort"} else 0
        if occasion == "sports":
            score = 6 * ("sports" in tags)
            score += 4 * (category == "activewear")
            score += 2 * (fit == "athletic" or "performance" in material)
            return score if score >= 4 else 0
        if occasion == "work":
            score = 5 * bool(tags & {"work", "office", "business"})
            score += 2 * (category in {"blazer", "dress", "shirt", "trousers"})
            score += 2 * (fit in {"tailored", "structured"})
            return score if score >= 3 else 0
        if occasion == "casual":
            score = 5 * bool(tags & {"casual", "everyday", "streetwear"})
            score += 1 * (category in {"t-shirt", "shirt", "jeans", "trousers", "dress", "skirt", "shoes"})
            return score if score >= 3 else 0
        if occasion == "summer":
            score = 5 * bool(tags & {"beach", "resort"})
            score += 3 * bool(seasons & {"summer", "spring; summer", "spring"})
            score += 1 * any(term in material for term in ("linen", "cotton voile", "combed cotton"))
            return score if score >= 3 else 0
        if occasion == "winter":
            score = 5 * bool(tags & {"winter", "cold weather"})
            score += 3 * bool(seasons & {"autumn; winter", "autumn", "winter"})
            score += 3 * (category in {"knitwear", "outerwear", "jacket"})
            score += 2 * any(term in material for term in ("merino", "wool", "fleece"))
            return score if score >= 3 else 0
        return 0

    @staticmethod
    def _occasion_note(variant: DemoVariant, occasion: str | None, arabic: bool) -> str:
        if not occasion:
            return ""
        tags = {arabic_fold(part.strip()) for part in re.split(r"[;,/]", variant.occasion) if part.strip()}
        seasons = {arabic_fold(part.strip()) for part in re.split(r"[;,/]", variant.season) if part.strip()}
        if occasion == "wedding" and tags & {"event", "formal", "smart casual"}:
            return "المنتج ده متصنّف للمناسبات في بيانات الكتالوج، وقصته/نوعه ظاهرين في التفاصيل." if arabic else "This item is tagged for events in the catalog; its cut and type are shown in the details."
        if occasion == "beach" and tags & {"beach", "resort"}:
            return "المنتج ده متصنّف للبحر أو المنتجع في بيانات الكتالوج." if arabic else "This item is tagged for beach or resort use in the catalog."
        if occasion == "sports" and ("sports" in tags or variant.category.lower() == "activewear"):
            return "ده منتج رياضي حسب تصنيف الكتالوج." if arabic else "This is cataloged as an athletic item."
        if occasion == "work" and "work" in tags:
            return "المنتج ده متصنّف للشغل في بيانات الكتالوج." if arabic else "This item is tagged for work in the catalog."
        if occasion == "casual" and tags & {"casual", "everyday", "streetwear"}:
            return "المنتج ده متصنّف للاستخدام الكاجوال في بيانات الكتالوج." if arabic else "This item is tagged for casual wear in the catalog."
        if occasion == "summer" and ("summer" in seasons or "spring; summer" in seasons or tags & {"beach", "resort"}):
            return "الخامة أو الموسم المسجّل في الكتالوج مناسب للأجواء الدافئة." if arabic else "The catalog lists this material or season for warmer weather."
        if occasion == "winter" and ("autumn; winter" in seasons or "winter" in seasons):
            return "الموسم المسجّل للقطعة في الكتالوج هو الخريف والشتاء." if arabic else "The catalog lists this item for autumn and winter."
        return ""

    def _occasion_candidates(self, variants: list[DemoVariant], occasion: str | None) -> list[DemoVariant]:
        if not occasion:
            return variants
        scored = [(self._occasion_score(variant, occasion), index, variant)
                  for index, variant in enumerate(variants)]
        suitable = [(score, index, variant) for score, index, variant in scored if score > 0]
        suitable.sort(key=lambda item: (-item[0], item[1]))
        return [variant for _score, _index, variant in suitable]

    @staticmethod
    def _groups(variants: list[DemoVariant]) -> list[list[DemoVariant]]:
        grouped: dict[str, list[DemoVariant]] = {}
        for v in variants:
            grouped.setdefault(v.product_id, []).append(v)
        return list(grouped.values())

    @staticmethod
    def _presentation(state: DemoConversationState, fields: list[str] | None = None,
                      variant_ids: list[str] | None = None) -> dict:
        return {
            "variant_ids": list(variant_ids if variant_ids is not None else state.displayed_variant_ids),
            "fields": list(dict.fromkeys(["name", *(fields if fields is not None else state.display_fields)])),
            "language": state.language,
        }

    def _show_products(self, message: str, state: DemoConversationState, constraints: dict, arabic: bool) -> tuple[str, int]:
        retrieved = self._variants(constraints)
        candidates = self._occasion_candidates(retrieved, constraints.get("occasion"))
        eligible_ids = {v.variant_id for v in candidates}
        occasion_rejected = [v for v in retrieved if v.variant_id not in eligible_ids]
        if not candidates:
            # A failed/new request invalidates the previous result list. This
            # prevents a later plural image/detail follow-up from reviving
            # unrelated products or their images.
            state.displayed_product_ids = []
            state.displayed_variant_ids = []
            state.displayed_price_evidence = []
            state.focused_product_id = None
            state.focused_variant_id = None
            state.selected_color = None
            state.selected_size = None
            state.display_fields = ["name"]
            detail = "within your budget" if constraints.get("max_price_egp") is not None else "with those requirements"
            if arabic:
                budget_text=format_egp(int(constraints["max_price_egp"]),True) if constraints.get("max_price_egp") is not None else ""
                reply = f"للأسف ملقتش منتج متاح في كتالوج الـDEMO بالمواصفات دي{(' وفي حدود '+budget_text if budget_text else '')}. تحب أوسّع البحث؟"
            else:
                reply = f"I could not find an in-stock demo product {detail}. Would you like to broaden the search?"
            self._log_search_trace(message, constraints, retrieved, candidates, state, reply, occasion_rejected)
            return reply, 0
        # An explicit catalog-wide cheapest request is a price sort, not a
        # semantic relevance query. Sort the eligible variants by catalog price
        # first, then collapse to distinct products by their cheapest eligible
        # variant. Other requests keep the trained XLM-R ranking path.
        cheapest = bool(constraints.get("sort_by") == "price_asc")
        ranked = (sorted(candidates, key=lambda v: (v.price_piasters, v.product_id, v.variant_id))
                  if cheapest else self.ranker(message, candidates, max(24, len(candidates))))
        grouped: dict[str, list[DemoVariant]] = {}
        for v in ranked:
            grouped.setdefault(v.product_id, []).append(v)
        count = int(constraints.get("requested_count") or (5 if cheapest else 3))
        groups = list(grouped.values())[:count]
        state.displayed_product_ids = [group[0].product_id for group in groups]
        state.displayed_variant_ids = [group[0].variant_id for group in groups]
        evidence_reason = ("lowest_valid_eligible_variant" if cheapest else
                           "within_inclusive_price_range" if constraints.get("price_range_egp") else
                           "valid_catalog_variant")
        state.displayed_price_evidence = [{
            "product_id": group[0].product_id,
            "variant_id": group[0].variant_id,
            "price_piasters": group[0].price_piasters,
            "eligible": True,
            "reason": evidence_reason,
        } for group in groups]
        primary = groups[0][0]
        state.focused_product_id, state.focused_variant_id = primary.product_id, primary.variant_id
        state.selected_color, state.selected_size = primary.color, primary.size
        state.pending_action = {"kind": "product_display", "primary_variant_id": primary.variant_id}
        requested_details = set(constraints.get("details", []))
        fields = ["name"]
        if "images" in requested_details: fields.append("image")
        if "prices" in requested_details: fields.append("price")
        if "sizes" in requested_details: fields.append("sizes")
        if "colors" in requested_details: fields.append("colors")
        elif constraints.get("color"): fields.append("color")
        if "category" in requested_details: fields.append("category")
        state.display_fields = fields
        lines = []
        for i, group in enumerate(groups, 1):
            v = group[0]
            display_group = [item for item in group if item.color == v.color] if cheapest else group
            colors = ", ".join(_display_color(c,arabic) for c in sorted({x.color for x in display_group}))
            base = f"{i}. {_display_name(v,arabic)}"
            if "prices" in requested_details:
                # The displayed amount belongs to this exact selected variant,
                # which is also the variant that passed price eligibility.
                price_text = format_egp(v.price_egp, arabic)
                base += f" — {price_text}"
            if "colors" in requested_details or constraints.get("color"):
                base += f" — {colors}"
            if "sizes" in requested_details:
                size_text = ", ".join(sorted({x.size for x in display_group}, key=lambda value: (len(value), value)))
                base += f" — المقاسات: {size_text}" if arabic else f" — sizes: {size_text}"
            if "category" in requested_details:
                category = localize_category(v.category, arabic)
                base += f" — {category}"
            lines.append(base)
        intro = (("ده الاختيار المطابق:" if len(groups) == 1 else "دي الاختيارات المطابقة:") if arabic else
                 f"I found {len(groups)} matching option{'s' if len(groups) != 1 else ''}:")
        suitability_note = self._occasion_note(groups[0][0], constraints.get("occasion"), arabic)
        if suitability_note:
            intro += "\n" + suitability_note
        suffix = ""
        if constraints.get("requested_count") and len(groups) < constraints["requested_count"]:
            suffix = f"\nالمتاح {len(groups)} فقط من المنتجات المطابقة." if arabic else f"\nOnly {len(groups)} distinct matching products are available."
        reply = intro + "\n" + "\n".join(lines) + suffix
        self._log_search_trace(message, constraints, retrieved, candidates, state, reply, occasion_rejected)
        return reply, len(candidates)

    def _list_details(self, message: str, state: DemoConversationState, arabic: bool,
                      product_named: bool = False) -> tuple[str, dict] | None:
        """Format an explicit plural follow-up against the same ordered SKUs."""
        if not _is_plural_list_followup(message, state, product_named):
            return None
        variants = [self.catalog.get(variant_id) for variant_id in state.displayed_variant_ids]
        variants = [variant for variant in variants if variant and variant.in_stock]
        if len(variants) < 2:
            return None
        fields = ["name"]
        wants_image = _is_image_question(message)
        wants_price = _is_price_question(message)
        wants_sizes = _is_sizes_question(message)
        wants_colors = _is_colors_question(message)
        if wants_image: fields.append("image")
        if wants_price: fields.append("price")
        if wants_sizes: fields.append("sizes")
        if wants_colors: fields.append("colors")
        lines = []
        for index, selected in enumerate(variants, 1):
            parts = [f"{index}. {_display_name(selected, arabic)}"]
            if wants_price:
                parts.append(format_egp(selected.price_egp, arabic))
            if wants_sizes:
                sizes = sorted({v.size for v in self.catalog.variants
                                if v.product_id == selected.product_id and v.color == selected.color and v.in_stock},
                               key=lambda value: (len(value), value))
                parts.append(("المقاسات " if arabic else "Sizes ") + ", ".join(sizes))
            if wants_colors:
                colors = sorted({v.color for v in self.catalog.variants if v.product_id == selected.product_id and v.in_stock})
                parts.append(("الألوان " if arabic else "Colors ") + ", ".join(_display_color(c, arabic) for c in colors))
            lines.append(" — ".join(parts))
        prefix = "دي نفس الاختيارات اللي عرضتهالك:" if arabic else "Here are the same products I showed you:"
        fields_to_show = fields
        if not fields_to_show or fields_to_show == ["name"]:
            fields_to_show = ["name"]
        state.display_fields = fields_to_show
        state.pending_action = {"kind": "product_list_followup", "variant_ids": [v.variant_id for v in variants]}
        missing_images = [v for v in variants if wants_image and (not v.image_path or not v.image_path.is_file())]
        if missing_images:
            prefix += (" بعض الصور التوضيحية مش متاحة حاليًا." if arabic else " Some illustrative images are currently unavailable.")
        reply = prefix + "\n" + "\n".join(lines)
        presentation = self._presentation(state, fields_to_show, [v.variant_id for v in variants])
        return reply, presentation

    def _focused_variants(self, state: DemoConversationState) -> list[DemoVariant]:
        if state.focused_product_id:
            return [v for v in self.catalog.variants if v.product_id == state.focused_product_id]
        return []

    def _match_product_name(self, message: str) -> str | None:
        """Resolve reviewed English/Arabic display names to catalog Product IDs."""
        def fold_name(value: str) -> str:
            return arabic_fold(value).replace("تيشيرت", "تيشرت").replace("تي شيرت", "تيشرت")
        folded = fold_name(message)
        products: dict[str, DemoVariant] = {}
        for variant in self.catalog.variants:
            products.setdefault(variant.product_id, variant)
        matches: list[tuple[int, str]] = []
        for product_id, variant in products.items():
            aliases = (variant.name, _display_name(variant, True))
            for alias in aliases:
                key = fold_name(alias)
                if len(key) >= 5 and key in folded:
                    matches.append((len(key), product_id))
        if not matches:
            return None
        longest = max(length for length, _product_id in matches)
        best = list(dict.fromkeys(product_id for length, product_id in matches if length == longest))
        return best[0] if len(best) == 1 else None

    def _select_displayed(self, message: str, state: DemoConversationState) -> DemoVariant | None:
        order = _ordinal(message)
        product_id = self._match_product_name(message)
        if product_id is None and order == -1 and state.displayed_product_ids:
            product_id = state.displayed_product_ids[-1]
        elif product_id is None and order is not None and order >= 0 and order < len(state.displayed_product_ids):
            product_id = state.displayed_product_ids[order]
        elif product_id is None and state.focused_product_id:
            product_id = state.focused_product_id
        if not product_id:
            return None
        requested_colors = extract_aliases(message, COLOR_ALIASES)
        requested_size = _size(message)
        # A pronoun such as “ده” refers to the exact variant presented as the
        # primary selection.  Do not reselect another colour just because the
        # product has several variant rows in the workbook.
        focused = self.catalog.get(state.focused_variant_id or "")
        if (focused and focused.product_id == product_id and focused.in_stock
                and not requested_colors and not requested_size):
            return focused
        variants = [v for v in self.catalog.variants if v.product_id == product_id and v.in_stock]
        if requested_colors:
            variants = [v for v in variants if any(c in v.color.lower() for c in requested_colors)]
        elif focused and focused.product_id == product_id:
            # Changing only a size keeps the displayed colour stable.
            variants = [v for v in variants if v.color == focused.color]
        if requested_size:
            variants = [v for v in variants if v.size.upper() == requested_size]
        return variants[0] if variants else None

    def _facts(self, message: str, state: DemoConversationState, arabic: bool) -> tuple[str | None, str | None]:
        wants_price, wants_sizes, wants_colors, wants_image = _is_price_question(message), _is_sizes_question(message), _is_colors_question(message), _is_image_question(message)
        if not (wants_price or wants_sizes or wants_colors or wants_image):
            return None, None
        selected = self._select_displayed(message, state)
        if not selected:
            return ("اختار منتج من الاختيارات المعروضة الأول عشان أقولك بياناته بدقة." if arabic else "Please select one displayed product first so I can give its exact catalog details."), "missing_product_reference"
        # Resolve variants from the product actually referenced in this turn;
        # an ordinal can point somewhere other than the previous primary item.
        variants = [v for v in self.catalog.variants if v.product_id == selected.product_id]
        if not variants:
            return ("اختار منتج من الاختيارات المعروضة الأول عشان أقولك بياناته بدقة." if arabic else "Please select one displayed product first so I can give its exact catalog details."), "missing_product_reference"
        # Use the exact referenced product's product_id, then report only facts
        # present in the catalog; a single size/color selection narrows it.
        colors = sorted({v.color for v in variants if v.in_stock})
        variants_for_color = [v for v in variants if v.color == selected.color and v.in_stock]
        sizes = sorted({v.size for v in variants_for_color}, key=lambda value: (len(value), value))
        parts = []
        if wants_price:
            prices = sorted({v.price_egp for v in variants if v.in_stock})
            exact_variant = bool(extract_aliases(message, COLOR_ALIASES) or _size(message)
                                 or state.pending_action.get("kind") == "added_to_cart")
            if len(prices) > 1 and not exact_variant:
                price_text = (f"من {format_egp(prices[0],True)} لحد {format_egp(prices[-1],True)}" if arabic else
                              f"{format_egp(prices[0],False)}–{format_egp(prices[-1],False)}")
                parts.append(f"سعر {_display_name(selected,True)} بيتراوح {price_text}" if arabic else f"{selected.name} ranges from {price_text}")
            else:
                parts.append(f"سعر {_display_name(selected,True)} هو {format_egp(selected.price_egp, True)}" if arabic else f"{selected.name} costs {format_egp(selected.price_egp, False)}")
        if wants_sizes:
            parts.append(f"المقاسات المتاحة للون {_display_color(selected.color,True)}: {', '.join(sizes)}" if arabic else f"Available sizes in {selected.color}: {', '.join(sizes)}")
        if wants_colors:
            parts.append(f"الألوان المتاحة: {', '.join(_display_color(c,True) for c in colors)}" if arabic else f"Available colors: {', '.join(colors)}")
        if wants_image:
            if _is_plural_image_question(message):
                shown = [self.catalog.get(variant_id) for variant_id in state.displayed_variant_ids]
                shown = [item for item in shown if item and item.in_stock]
                names = [_display_name(item, arabic) for item in shown]
                if names:
                    parts.append(("دي صور " + "، ".join(names)) if arabic else ("Here are images of " + ", ".join(names)))
                else:
                    parts.append("دي صور المنتجات اللي عرضتهالك." if arabic else "Here are images of the products I showed you.")
            else:
                # Keep a useful plain-text transcript alongside the compact
                # card. The card itself carries the subtle demo-image notice.
                parts.append((f"دي صورة {_display_name(selected, True)}." if arabic else
                              f"Here is the image of {selected.name}."))
        state.focused_product_id = selected.product_id
        state.focused_variant_id = selected.variant_id
        state.selected_color, state.selected_size = selected.color, selected.size
        state.pending_action = {"kind": "fact_followup", "primary_variant_id": selected.variant_id}
        return ". ".join(parts), None

    def _category_overview(self, categories: list[str], arabic: bool) -> str | None:
        """Brief category availability answer without unsolicited variant facts."""
        candidates = [v for v in self.catalog.variants if v.in_stock and v.category.lower() in {c.lower() for c in categories}]
        if not candidates:
            return None
        names = []
        for variant in candidates:
            if variant.name not in names:
                names.append(variant.name)
            if len(names) == 4:
                break
        if arabic:
            localized = [_display_name(next(v for v in candidates if v.name == name),True) for name in names]
            types = "، ".join(dict.fromkeys(localized))
            return f"أيوه، عندنا {types}. تحب تشوف نوع معين؟"
        return f"Yes, we have {', '.join(names)}. Would you like to see a particular type?"

    def respond(self, message: str, stored: dict | None) -> tuple[str, dict, dict]:
        state = DemoConversationState.from_storage(stored)
        # Short replies such as a numeric shoe size contain no language signal.
        # Keep the language of the active conversation so a pending action
        # does not unexpectedly switch to English.
        if _arabic(message):
            state.language = "ar"
        elif re.search(r"[A-Za-z]", message):
            state.language = "en"
        arabic = state.language == "ar"
        decision = route(message, bool(state.active))
        explicit_colors = list(extract_aliases(message, COLOR_ALIASES))
        favorite_statement = has_signal(message, "بحب", "favorite", "love") and bool(explicit_colors)
        if favorite_statement:
            for color in explicit_colors:
                if color not in state.favorite_colors:
                    state.favorite_colors.append(color)

        # A short affirmative is meaningful only as an answer to a pending offer.
        if _is_yes(message) and state.pending_action.get("kind") == "offer_alternatives":
            proposal = dict(state.pending_action)
            constraints = dict(proposal.get("constraints") or state.active)
            relax = proposal.get("relax", "color")
            if relax == "color":
                constraints.pop("color", None)
            elif relax == "size":
                constraints.pop("size", None)
            elif relax == "budget":
                constraints.pop("max_price_egp", None)
            elif relax == "category":
                constraints.pop("categories", None)
            else:
                state.pending_action = {"kind": "ask_attribute", "reason": "clarify_broaden"}
                reply = "تحب أوسّع البحث بإلغاء اللون، ولا المقاس، ولا حدود السعر؟" if arabic else "Which constraint should I relax: color, size, or budget?"
                self._log(message, "clarify_broaden", state, 0, "not_invoked:pending_action_clarification")
                return reply, state.storage(), {"add_variant_id": None}
            state.pending_action = {}
            # Never execute an accepted broadening as the same failed search.
            if constraints == state.active:
                state.pending_action = {"kind": "ask_attribute", "reason": "clarify_broaden"}
                reply = "تحب أوسّع البحث بإلغاء اللون، ولا المقاس، ولا حدود السعر؟" if arabic else "Which constraint should I relax: color, size, or budget?"
                self._log(message, "clarify_broaden", state, 0, "not_invoked:pending_action_clarification")
                return reply, state.storage(), {"add_variant_id": None}
            reply, count = self._show_products(message, state, constraints, arabic)
            state.active = constraints
            self._log(message, "accept_alternatives", state, count, "not_invoked:catalog_fact_route")
            return reply, state.storage(), {"add_variant_id": None}

        if state.pending_action.get("kind") == "choose_size_for_cart":
            size = _pending_size(message)
            requested_colors = extract_aliases(message, COLOR_ALIASES)
            product_id = state.pending_action.get("product_id")
            color = requested_colors[-1] if requested_colors else state.pending_action.get("color")
            if size:
                chosen = next((v for v in self.catalog.variants if v.product_id == product_id and v.color.lower() == str(color).lower() and v.size.upper() == size.upper() and v.in_stock), None)
            else:
                chosen = None
            if chosen:
                state.focused_variant_id, state.selected_color, state.selected_size = chosen.variant_id, chosen.color, chosen.size
                state.pending_action = {"kind": "added_to_cart", "primary_variant_id": chosen.variant_id}
                reply = f"تمت إضافة {_display_name(chosen,True)} — {_display_color(chosen.color,True)}/{chosen.size} للسلة." if arabic else f"Added {chosen.name} — {chosen.color}/{chosen.size} to the cart."
                self._log(message, "add_to_cart_size", state, 1, "not_invoked:catalog_fact_route")
                return reply, state.storage(), {"add_variant_id": chosen.variant_id}
            if size and requested_colors:
                reply = "اللون والمقاس دول مش متاحين مع بعض للمنتج ده. مش هبدّلهم تلقائيًا؛ تحب تختار تركيبة تانية؟" if arabic else "That exact color and size combination is unavailable. I will not substitute it automatically; would you like another combination?"
                self._log(message, "add_to_cart_size", state, 0, "not_invoked:catalog_fact_route", "exact_variant_missing")
                return reply, state.storage(), {"add_variant_id": None}
            if requested_colors and not size:
                matches = [v for v in self.catalog.variants if v.product_id == product_id and v.color.lower() == requested_colors[-1].lower() and v.in_stock]
                if not matches:
                    reply = "اللون ده مش متاح للمنتج اللي اخترته. قولّي لون ومقاس تاني لو حابب." if arabic else "That color is unavailable for the selected product. Tell me another color and size if you like."
                    self._log(message, "add_to_cart_color", state, 0, "not_invoked:catalog_fact_route", "pending_color_unavailable")
                    return reply, state.storage(), {"add_variant_id": None}
                color = requested_colors[-1]
                state.pending_action["color"] = color
                state.selected_color = color
                available = sorted({v.size for v in matches}, key=lambda value: (len(value), value))
                reply = f"تمام، اللون {_display_color(color,True)} متاح. اختار مقاس: {', '.join(available)}." if arabic else f"That color is available. Choose a size: {', '.join(available)}."
                self._log(message, "add_to_cart_color", state, len(matches), "not_invoked:catalog_fact_route")
                return reply, state.storage(), {"add_variant_id": None}
            if not size:
                reply = (f"اختار مقاس للون {_display_color(str(color),True)}، زي L أو M." if arabic else f"Choose a size for {color}, such as L or M.")
                self._log(message, "add_to_cart_size", state, 0, "not_invoked:awaiting_size", "size_not_parsed")
                return reply, state.storage(), {"add_variant_id": None}

        if decision.kind == "memory_question":
            colors = state.favorite_colors
            visible_colors=", ".join(_display_color(color,True) for color in colors)
            reply = (f"أيوه، فاكر إنك بتحب: {visible_colors}." if colors else "لسه ما قلتليش لونك المفضل.") if arabic else (f"Yes — your favorite colors are: {', '.join(colors)}." if colors else "You have not told me a favorite color yet.")
            self._log(message, decision.kind, state, 0, "not_invoked:memory_route")
            return reply, state.storage(), {"add_variant_id": None}

        if favorite_statement and not _explicit_product_request(message):
            visible_colors=", ".join(_display_color(color,True) for color in explicit_colors)
            reply=(f"تمام، هفتكر إنك بتحب {visible_colors}." if arabic else f"Got it—I’ll remember that you like {', '.join(explicit_colors)}.")
            self._log(message,"preference_update",state,0,"not_invoked:memory_route")
            return reply,state.storage(),{"add_variant_id":None}

        named_product_id = self._match_product_name(message)
        named_product_variants = [v for v in self.catalog.variants if v.product_id == named_product_id and v.in_stock] if named_product_id else []
        if named_product_variants:
            state.focused_product_id = named_product_id
            current_variant = self.catalog.get(state.focused_variant_id or "")
            if not current_variant or current_variant.product_id != named_product_id:
                state.focused_variant_id = named_product_variants[0].variant_id
                state.selected_color, state.selected_size = named_product_variants[0].color, named_product_variants[0].size
            if state.pending_action.get("kind") == "choose_size_for_cart" and state.pending_action.get("product_id") != named_product_id:
                state.pending_action = {}
            state.active = {"categories": [named_product_variants[0].category]}
            asks_fact = (_is_price_question(message) or _is_sizes_question(message) or _is_colors_question(message) or _is_image_question(message))
            if not asks_fact and not _is_add_request(message):
                state.pending_action = {"kind": "product_selected", "product_id": named_product_id}
                product_label = _display_name(named_product_variants[0], arabic)
                reply = (f"اختيار حلو! خلّينا في {product_label}. تحب تعرف ألوانه ومقاساته ولا تشوف صورته؟" if arabic else
                         f"Good choice. Let’s look at the {named_product_variants[0].name}. Would you like its colors and sizes, or a picture?")
                self._log(message, "product_selection", state, len(named_product_variants), "not_invoked:catalog_fact_route")
                return reply, state.storage(), {"add_variant_id": None}

        list_details = self._list_details(message, state, arabic, product_named=bool(named_product_id))
        if list_details:
            reply, presentation = list_details
            self._log(message, "product_list_followup", state, len(presentation["variant_ids"]), "not_invoked:catalog_fact_route")
            return reply, state.storage(), {"add_variant_id": None, "presentation": presentation}

        ordinal = _ordinal(message)
        if ordinal is not None and not (_is_price_question(message) or _is_sizes_question(message) or _is_colors_question(message) or _is_image_question(message)) and not _is_add_request(message):
            selected = self._select_displayed(message, state)
            if selected:
                state.focused_product_id, state.focused_variant_id = selected.product_id, selected.variant_id
                state.selected_color, state.selected_size = selected.color, selected.size
                state.pending_action = {"kind": "product_selected", "product_id": selected.product_id}
                reply = (f"تمام، اختيارك هو {_display_name(selected, arabic)}." if arabic else
                         f"Sure, your selection is {selected.name}.")
                self._log(message, "product_selection", state, 1, "not_invoked:catalog_fact_route")
                return reply, state.storage(), {"add_variant_id": None, "presentation": self._presentation(state, ["name"], [selected.variant_id])}

        # Exact factual references always win over generic search routing.
        # “I want something under X” includes the word price in Arabic but is a
        # new constrained search, never a question about the old product.
        image_context_followup = False
        if _is_image_question(message) and state.displayed_variant_ids and not (
                explicit_colors or _size(message) or _budget(message) is not None or _requested_count(message)):
            shown_categories = {variant.category.lower() for variant_id in state.displayed_variant_ids
                                if (variant := self.catalog.get(variant_id)) is not None}
            requested_categories = {category.lower() for category in _category_values(message)}
            image_context_followup = not requested_categories or bool(requested_categories & shown_categories)
        starts_catalog_request = bool(
            decision.occasion or _category_values(message) or decision.mixed_color_request
            or _requested_count(message)
            or (explicit_colors and _explicit_product_request(message))
        ) and not image_context_followup
        fact_reply, fact_rejection = (self._facts(message, state, arabic)
                                      if _budget(message) is None and (not starts_catalog_request or (named_product_id and (_is_price_question(message) or _is_sizes_question(message) or _is_colors_question(message) or _is_image_question(message))))
                                      else (None, None))
        if fact_reply:
            self._log(message, "product_facts", state, 1 if not fact_rejection else 0, "not_invoked:catalog_fact_route", fact_rejection or "")
            selected = self._select_displayed(message, state)
            if selected and _is_image_question(message):
                image_ids = list(state.displayed_variant_ids) if _is_plural_image_question(message) else [selected.variant_id]
                fields = ["image"]
                if _is_price_question(message): fields.append("price")
                if _is_sizes_question(message): fields.append("sizes")
                if _is_colors_question(message): fields.append("colors")
                return fact_reply, state.storage(), {"add_variant_id": None,
                    "image_variant_ids": image_ids or [selected.variant_id],
                    "presentation": self._presentation(state, fields, image_ids or [selected.variant_id])}
            return fact_reply, state.storage(), {"add_variant_id": None}

        if _is_add_request(message):
            selected = self._select_displayed(message, state)
            if not selected:
                reply = "اختار منتج من الاختيارات المعروضة الأول عشان أضيفه للسلة." if arabic else "Please choose one of the displayed products first so I can add it to the cart."
                self._log(message, "add_to_cart", state, 0, "not_invoked:missing_product_reference", "missing_product_reference")
                return reply, state.storage(), {"add_variant_id": None}
            variants = [v for v in self.catalog.variants if v.product_id == selected.product_id and v.in_stock and v.color == selected.color]
            requested_size = _size(message)
            if requested_size:
                variants = [v for v in variants if v.size.upper() == requested_size]
            if not variants:
                reply = "اللون أو المقاس المطلوب غير متاح للمنتج ده في الكتالوج." if arabic else "That color or size is not available for this product in the catalog."
                self._log(message, "add_to_cart", state, 0, "not_invoked:variant_unavailable", "variant_unavailable")
                return reply, state.storage(), {"add_variant_id": None}
            available_sizes = sorted({v.size for v in variants}, key=lambda value: (len(value), value))
            if not requested_size and len(available_sizes) > 1:
                reply = f"تمام، اختار المقاس الأول للون {_display_color(selected.color,True)}: {', '.join(available_sizes)}." if arabic else f"Sure—choose a size for {selected.color} first: {', '.join(available_sizes)}."
                state.pending_action = {"kind": "choose_size_for_cart", "product_id": selected.product_id, "color": selected.color}
                self._log(message, "add_to_cart", state, len(variants), "not_invoked:awaiting_size")
                return reply, state.storage(), {"add_variant_id": None}
            chosen = variants[0]
            state.focused_variant_id, state.selected_color, state.selected_size = chosen.variant_id, chosen.color, chosen.size
            state.pending_action = {"kind": "added_to_cart", "primary_variant_id": chosen.variant_id}
            self._log(message, "add_to_cart", state, 1, "not_invoked:catalog_fact_route")
            return (f"تمت إضافة {_display_name(chosen,True)} — {_display_color(chosen.color,True)}/{chosen.size} للسلة." if arabic else f"Added {chosen.name} — {chosen.color}/{chosen.size} to the cart."), state.storage(), {"add_variant_id": chosen.variant_id}

        if decision.kind == "fit_advice":
            reply = ("الـOversized أوسع وحرّكته أريح، والـRegular Fit أقرب لشكل الجسم. اختار اللي تحس فيه براحة؛ الطول والوزن لوحدهم مش كفاية عشان أحدد مقاس دقيق." if arabic else "Oversized is roomier and more relaxed; regular fit sits closer to the body. Choose what feels comfortable—height and weight alone are not enough to determine an exact size.")
            reply, qwen_status = self._phrase(message, reply, arabic)
            self._log(message, decision.kind, state, 0, qwen_status)
            return reply, state.storage(), {"add_variant_id": None}

        # ``striped`` is a supported workbook pattern (the older legacy router
        # classified it as unsupported); a suit is still not a recorded category.
        unsupported = tuple(term for term in decision.unsupported_catalog_terms if term != "striped")
        if unsupported:
            term = unsupported[0]
            reply = (f"الكتالوج الحالي لا يسجل {term} كصفة/فئة قابلة للتأكد منها، لذلك لا أقدر أوعدك بتوفره. تحب بديل من الفئات المسجلة؟" if arabic else f"The current catalog cannot verify {term} as a searchable attribute/category, so I cannot confirm availability. Would you like a supported alternative?")
            self._log(message, decision.kind, state, 0, "not_invoked:unsupported_route")
            return reply, state.storage(), {"add_variant_id": None}

        # A one-word "colour" after we asked what to change has its own useful
        # response rather than replaying the same clarification.
        if _is_colors_question(message) and state.pending_action.get("kind") == "ask_attribute":
            state.pending_action = {"kind": "offer_color_choice"}
            variants = self._variants(state.active)
            values = sorted({v.color for v in variants})
            reply = (f"الألوان المتاحة للطلب الحالي: {', '.join(values)}. تحب أي لون؟" if values else "مفيش ألوان متاحة للمواصفات الحالية. تحب نوسّع البحث؟") if arabic else (f"Available colors for this request: {', '.join(values)}. Which color would you like?" if values else "No colors are available for the current requirements. Would you like to broaden the search?")
            self._log(message, "color_choice", state, len(variants), "not_invoked:catalog_fact_route")
            return reply, state.storage(), {"add_variant_id": None}

        # Occasion (notably work interview / beach) starts a new shopping task;
        # persistent favorites survive but stale color/category filters do not.
        global_cheapest = _is_global_cheapest_request(message)
        price_range = _price_range(message)
        explicit_list = _requested_count(message) is not None and _explicit_product_request(message)
        starts_new = bool(global_cheapest or price_range or explicit_list or decision.occasion or decision.category or unsupported or decision.mixed_color_request or _budget(message) is not None or (favorite_statement and _explicit_product_request(message)))
        if starts_new:
            state.displayed_product_ids = []
            state.displayed_variant_ids = []
            state.displayed_price_evidence = []
            state.focused_product_id = None
            state.focused_variant_id = None
            state.selected_color = None
            state.selected_size = None
            state.display_fields = ["name"]
            # A pending yes/no or cart clarification belongs to the old task.
            state.pending_action = {}
        constraints = {} if starts_new else dict(state.active)
        if global_cheapest:
            constraints = {"sort_by": "price_asc", "requested_count": _requested_count(message) or 5}
        if price_range:
            constraints["price_range_egp"] = list(price_range)
        if decision.occasion:
            constraints["occasion"] = decision.occasion
            if decision.occasion == "work":
                constraints["categories"] = ["shirt", "blazer", "dress"]
        categories = _category_values(message)
        if categories:
            constraints["categories"] = categories
            constraints.pop("occasion", None)
        if explicit_colors:
            constraints["color"] = explicit_colors[-1]
        size = _size(message)
        if size:
            constraints["size"] = size
        budget = _budget(message)
        if budget is not None:
            constraints["max_price_egp"] = budget
        elif not global_cheapest and has_signal(message, "ارخص", "أرخص", "cheaper", "less expensive") and state.focused_product_id:
            reference = self._select_displayed(message, state)
            if reference:
                # “cheaper” is always strictly cheaper than the precise product
                # in focus—not merely a low price elsewhere in the catalog.
                constraints["max_price_egp"] = reference.price_egp - 1
                constraints.setdefault("categories", [reference.category])
        if decision.mixed_color_request:
            constraints["mixed"] = True
        if has_signal(message, "مخطط", "مخططة", "striped", "stripe"):
            constraints["pattern"] = "stripe"
        requested_count = _requested_count(message)
        if requested_count:
            constraints["requested_count"] = requested_count
        details = []
        if _is_image_question(message): details.append("images")
        if _is_sizes_question(message): details.append("sizes")
        if _is_price_question(message) or has_signal(message, "أسعار", "اسعار", "prices"):
            details.append("prices")
        if _is_colors_question(message): details.append("colors")
        if has_signal(message, "الفئة", "التصنيف", "category"):
            details.append("category")
        if details: constraints["details"] = details
        # A catalog-wide cheapest request is meaningful only if the verified
        # prices used for sorting are visible beside the returned products.
        if global_cheapest and "prices" not in details:
            details.append("prices")
            constraints["details"] = details
        # A price-range query is inherently asking to compare the prices that
        # passed the filter; include that verified value in each result.
        if price_range and "prices" not in details:
            details.append("prices")
            constraints["details"] = details
        if decision.relax_constraints:
            constraints.pop("color", None); constraints.pop("categories", None); constraints.pop("category", None)

        searchish = (global_cheapest or price_range is not None or decision.kind in {"product_search", "product_change", "workwear_search", "occasion_search", "mixed_color_request"} or (bool(decision.occasion) and decision.kind != "occasion_advice") or (_explicit_product_request(message) and bool(explicit_colors)) or (has_signal(message, "مخطط", "مخططة", "striped", "stripe") and _explicit_product_request(message)))
        category_question = bool(categories) and has_signal(message, "عندكم", "عندك", "هل في", "هل فيه", "do you have")
        if category_question and not explicit_colors and not size and budget is None and price_range is None and not global_cheapest and not details and not _requested_count(message) and not has_signal(message, "مخطط", "مخططة", "striped", "stripe"):
            overview = self._category_overview(categories, arabic)
            if overview:
                state.active = {"categories": categories}
                visible=[]; seen=set()
                for variant in self.catalog.variants:
                    if variant.in_stock and variant.category.lower() in {c.lower() for c in categories} and variant.product_id not in seen:
                        seen.add(variant.product_id); visible.append(variant)
                    if len(visible) >= 4:
                        break
                state.displayed_product_ids=[v.product_id for v in visible]
                state.displayed_variant_ids=[v.variant_id for v in visible]
                if visible:
                    state.focused_product_id=visible[0].product_id
                    state.focused_variant_id=visible[0].variant_id
                    state.selected_color,state.selected_size=visible[0].color,visible[0].size
                state.display_fields = ["name"]
                state.pending_action = {}
                self._log(message, "category_overview", state, 0, "not_invoked:catalog_fact_route")
                return overview, state.storage(), {"add_variant_id": None}
        if searchish:
            state.active = constraints
            reply, count = self._show_products(message, state, constraints, arabic)
            if count == 0 and explicit_colors:
                # Store the proposal and its exact search scope so an affirmative
                # can execute the proposed relaxation without losing category,
                # size, or budget constraints.
                state.pending_action = {"kind": "offer_alternatives", "missing_color": explicit_colors[-1],
                                        "relax": "color", "constraints": dict(constraints)}
            self._log(message, decision.kind, state, count, "not_invoked:catalog_fact_route")
            return reply, state.storage(), {"add_variant_id": None,
                "image_variant_ids": list(state.displayed_variant_ids) if "images" in details else [],
                "presentation": self._presentation(state) if count else None}

        if decision.kind in {"occasion_advice", "style_advice"}:
            # A newly mentioned occasion starts a fresh shopping task even if
            # the customer is asking for advice rather than requesting a list.
            # Retaining the previous task here made a later work/wedding
            # question inherit (and re-show) an unrelated beach/color filter.
            if decision.occasion:
                previous_occasion = state.active.get("occasion")
                if previous_occasion != decision.occasion:
                    state.active = {"occasion": decision.occasion}
                    state.pending_action = {}
                    state.displayed_product_ids = []
                    state.displayed_variant_ids = []
                    state.focused_product_id = None
                    state.focused_variant_id = None
                    state.selected_color = None
                    state.selected_size = None
            if decision.occasion == "casual":
                reply = "للخروجة، تيشيرت أو قميص مريح مع جينز اختيار بسيط ومرتب. بتحب اللوك كاجوال ولا أشيك شوية؟" if arabic else "For going out, a comfortable T-shirt or shirt with jeans is an easy polished choice. Do you prefer casual or smarter?"
            elif decision.occasion == "wedding":
                reply = "للفرح اختار قطعة مرتبة ومريحة زي قميص أو بليزر، وخلّي الألوان هادئة. تحب ستايل كلاسيك ولا كاجوال؟"
            elif decision.occasion == "beach":
                reply = "للبحر اختار خامة خفيفة ومريحة. تحب أدوّر لك في كتالوج الـDEMO على اختيارات مناسبة؟"
            else:
                reply = "أقدر أساعدك تنسّق ستايل مناسب. رايح فين وبتفضّل ستايل كلاسيك ولا كاجوال؟" if arabic else "I can help you plan a look. Where are you going, and do you prefer classic or casual?"
            reply, qwen_status = self._phrase(message, reply, arabic)
            self._log(message, decision.kind, state, 0, qwen_status)
            return reply, state.storage(), {"add_variant_id": None}

        # Qwen remains reserved for non-catalog free conversation.  Calling it
        # through the legacy agent is intentionally avoided here because its
        # older state machine can overwrite demo state; the log identifies this
        # transparent deterministic fallback rather than pretending Qwen spoke.
        reply = "أهلاً! قولّي بتدور على إيه أو محتاج رأي في ستايل؟" if arabic else "Hi! Tell me what you are looking for, or ask me for styling help."
        reply, qwen_status = self._phrase(message, reply, arabic)
        self._log(message, decision.kind, state, 0, qwen_status)
        return reply, state.storage(), {"add_variant_id": None}
