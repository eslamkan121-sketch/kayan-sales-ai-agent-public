"""Arabic-aware intent and slot extraction for the KAYAN sales conversation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher


COLOR_ALIASES = {
    "red": ("red", "احمر", "أحمر", "حمر", "احمار"),
    "black": ("black", "اسود", "أسود", "سود"),
    "white": ("white", "ابيض", "أبيض", "بيضا", "بيضاء", "بيضه"),
    "navy": ("navy", "كحلي", "كحلى"),
    "beige": ("beige", "بيج"),
    "olive": ("olive", "زيتي", "زيتى"),
    "blue": ("blue", "ازرق", "أزرق", "زرق"),
    "yellow": ("yellow", "اصفر", "أصفر", "الصفر", "الأصفر", "صفرا", "صفراء"),
    "pink": ("pink", "وردي", "وردى", "زهري", "زهريه", "زهريّة", "روز"),
    "purple": ("purple", "بنفسجي", "بنفسجى", "بنفسجية", "موف"),
    # Keep colors that are absent from today's catalog as explicit constraints.
    # A missing catalog color is meaningful customer intent, not a cue to guess.
    "green": ("green", "اخضر", "أخضر", "الخضر", "الأخضر", "خضرا", "خضراء"),
}
CATEGORY_ALIASES = {
    "t-shirt": ("t-shirt", "t shirt", "tshirt", "تيشيرت", "تيشرت", "تي شيرت"),
    "shirt": ("shirt", "قميص"), "hoodie": ("hoodie", "هودي"),
    "jeans": ("jeans", "جينز"), "trousers": ("trousers", "pants", "بنطلون"),
    "blazer": ("blazer", "بليزر"), "dress": ("dress", "فستان"),
    "jacket": ("jacket", "جاكيت"), "shoes": ("shoes", "sneakers", "كوتشي", "حذاء", "جزمة", "جزم", "احذية", "أحذية"),
    "knitwear": ("knitwear", "sweater", "cardigan", "بلوفر", "كارديجان"),
    "outerwear": ("outerwear", "puffer", "جاكيت مبطن", "معطف"),
    "skirt": ("skirt", "جيب", "تنورة"),
    "activewear": ("activewear", "leggings", "training", "ليجنز", "تريننج"),
    "accessories": ("accessories", "tote", "شنطة", "اكسسوار", "إكسسوار"),
}


def arabic_fold(text: str) -> str:
    text = text.lower().translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ؤ": "و", "ئ": "ي"}))
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\u0600-\u06ff]+", arabic_fold(text)))


def _similar(left: str, right: str) -> bool:
    return len(left) >= 4 and len(right) >= 4 and SequenceMatcher(None, left, right).ratio() >= 0.82


def has_signal(text: str, *signals: str) -> bool:
    folded = arabic_fold(text)
    word_set = tokens(text)
    for signal in signals:
        folded_signal = arabic_fold(signal)
        # English aliases must match word boundaries: substring matching made
        # "striped" look like a "trip" and "t-shirt" look like a shirt-only
        # request. Arabic still uses the substring path for attached clitics.
        if (re.search(rf"(?<![a-z0-9]){re.escape(folded_signal)}(?![a-z0-9])", folded)
                if folded_signal.isascii() else folded_signal in folded):
            return True
        if not folded_signal.isascii() and " " not in folded_signal and any(_similar(word, folded_signal) for word in word_set):
            return True
    return False


def extract_alias(text: str, aliases: dict[str, tuple[str, ...]]) -> str | None:
    for canonical, forms in aliases.items():
        if has_signal(text, *forms):
            return canonical
    return None


def extract_aliases(text: str, aliases: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Return every explicit slot value, in catalog-independent canonical form."""
    return tuple(canonical for canonical, forms in aliases.items() if has_signal(text, *forms))


@dataclass(frozen=True)
class IntentDecision:
    kind: str
    confidence: float
    color: str | None = None
    category: str | None = None
    categories: tuple[str, ...] = field(default_factory=tuple)
    size: str | None = None
    max_price_egp: int | None = None
    color_tone: str | None = None
    signals: tuple[str, ...] = field(default_factory=tuple)
    occasion: str | None = None
    colors: tuple[str, ...] = field(default_factory=tuple)
    mixed_color_request: bool = False
    requested_count: int | None = None
    unsupported_catalog_terms: tuple[str, ...] = field(default_factory=tuple)
    relax_constraints: bool = False


def route(message: str, has_active_request: bool = False) -> IntentDecision:
    folded = arabic_fold(message)
    colors = extract_aliases(message, COLOR_ALIASES)
    color = colors[0] if colors else None
    categories = tuple(key for key, forms in CATEGORY_ALIASES.items() if has_signal(message, *forms))
    if "t-shirt" in categories and "shirt" in categories:
        # The final ``shirt`` in ``t-shirt`` is not a second category. Retain
        # shirt only when it appears as a separate word (for example,
        # "t-shirt or shirt").
        if (not re.search(r"(?<![-a-z0-9])shirt(?![a-z0-9])", folded)
                and not has_signal(message, "قميص")):
            categories = tuple(value for value in categories if value != "shirt")
    category = categories[0] if categories else None
    size_match = re.search(r"(?:size|مقاس)\s*(xxl|xl|xs|[sml]|[0-9]{2})\b", folded, re.I)
    size = size_match.group(1).upper() if size_match else None
    budget_match = re.search(r"(?:under|below|less\s+than|تحت|اقل\s+من|ارخص\s+من)\s*(?:egp\s*)?(\d{2,5})", folded, re.I)
    # These forms are strict: “under 300” means a maximum catalog price of 299.
    budget = int(budget_match.group(1)) - 1 if budget_match else None
    light = has_signal(message, "فاتح", "فاتحه", "الوان فاتحه", "light color", "light colours")
    dark = has_signal(message, "غامق", "غامقه", "dark color", "dark colours")
    favorite = (has_signal(message, "بحب", "مفضل", "favorite", "love") and (color is not None or has_signal(message, "لون", "color")))
    recall = has_signal(message, "فاكر", "عارف", "عرف", "تذكر", "remember") and has_signal(message, "بحب", "لون", "color")
    fit = has_signal(message, "oversized", "oversize", "regular fit", "فرق", "difference") and has_signal(message, "oversized", "regular", "fit", "فرق", "difference")
    wedding = has_signal(message, "فرح", "عرس", "wedding")
    beach = has_signal(message, "بحر", "شاطئ", "beach", "resort")
    university = has_signal(message, "جامعة", "الجامعة", "university", "campus")
    travel = has_signal(message, "سفر", "مسافر", "travel", "trip")
    sports = has_signal(message, "رياضة", "رياضي", "رياضية", "جيم", "تمرين", "تريننج", "sports", "gym", "workout", "athletic")
    work_interview = has_signal(message, "مقابلة شغل", "مقابله شغل", "انترفيو", "interview", "job interview")
    work = work_interview or has_signal(message, "للشغل", "الشغل", "دوام", "مكتب", "office", "workwear")
    summer = has_signal(message, "صيف", "الصيف", "summer", "hot weather")
    winter = has_signal(message, "شتاء", "الشتا", "الشتاء", "winter", "cold weather")
    casual_outing = has_signal(message, "خروجة", "خروجه", "خروجات", "مشوار", "outing", "casual")
    formal = has_signal(message, "رسمي", "رسمية", "فورمال", "formal", "smart casual", "سهرة")
    occasion = ("work" if work else "wedding" if wedding else "beach" if beach else
                "university" if university else "travel" if travel else "sports" if sports else
                "summer" if summer else "winter" if winter else "casual" if casual_outing else
                "formal" if formal else None)
    alternatives = has_signal(message, "بديل", "بدايل", "بدائل", "حاجة تانية", "وريني", "مش عاجبني", "alternative", "another")
    colors_question = has_signal(message, "الوان تاني", "لون تاني", "الوان ايه", "ألوان ايه", "ايه الالوان", "ايه الألوان", "what colors", "other color", "different color")
    cheaper = has_signal(message, "ارخص", "أرخص", "cheaper", "less expensive")
    mixed_color_request = has_signal(message, "مكس", "mix", "multi color", "multicolor", "متعدد الالوان")
    unsupported_catalog_terms = tuple(term for term, forms in {
        "striped": ("مخطط", "مخططة", "striped"),
        "suit": ("بدله", "بدلة", "suit"),
    }.items() if has_signal(message, *forms))
    requested_count_match = re.search(r"(?:list|ليست|قائمة)\s*(\d+)", folded, re.I)
    requested_count = int(requested_count_match.group(1)) if requested_count_match else (
        5 if has_signal(message, "خمس", "خمسة", "five") and has_signal(message, "ليست", "قائمة", "list") else None
    )
    relax_constraints = has_signal(message, "اي لون", "أي لون", "any color") and has_signal(message, "اي نوع", "أي نوع", "any type")
    style = has_signal(message, "ستايل", "style", "كلاسيك", "classic", "يلبق", "يليق", "شايف", "اقترح", "قترح", "تنصح", "خروجه", "خروجة", "موعد عمل", "going out")
    # Shopping-request verbs must match explicitly. Fuzzy matching made words
    # such as "محتار" look like "محتاج", turning a styling question into a
    # catalog search. Keep typo tolerance for slots, but not this routing gate.
    shopping_verbs = ("عايز", "عاوز", "محتاج", "فيه", "عندكم", "عندك", "هل فيه", "هل عندك", "متاح", "هات", "اديني", "وريني", "ورّيني")
    product_ask = any(arabic_fold(verb) in folded for verb in shopping_verbs) or bool(
        re.search(r"(?<![a-z0-9])(?:show|need|looking|find|recommend|suggest)(?![a-z0-9])", folded, re.I))
    # "محتاج رأيي/مساعدة في الستايل" is a request for advice, not a catalog
    # query. A garment/category later in the sentence still routes to search.
    advice_request = ("محتاج رايي" in folded or "محتاج رايك" in folded or "محتاج مساعدة" in folded
                      or "need your opinion" in folded or "need advice" in folded)
    if advice_request:
        product_ask = False
    occasion_recommendation = bool(occasion) and (
        product_ask or has_signal(message, "اقترح", "قترح", "recommend", "suggest"))
    if advice_request:
        occasion_recommendation = False
    casual = has_signal(message, "شكرا", "شكرًا", "thanks", "thank you", "اهلا", "أهلا", "hello", "hi")

    if unsupported_catalog_terms:
        kind = "unsupported_catalog"
    elif mixed_color_request:
        kind = "mixed_color_request"
    elif recall:
        kind = "memory_question"
    # A customer can save a preference and request stock in the same message.
    # In that case product retrieval is needed, but extract_preferences still
    # records the favourite color before the search begins.
    elif favorite and not product_ask:
        kind = "preference_update"
    elif fit:
        kind = "fit_advice"
    elif work:
        # This opens a new task: it should not inherit a prior shirt/tee/color.
        kind = "workwear_search"
    elif occasion_recommendation:
        kind = "occasion_search"
    elif occasion:
        kind = "occasion_advice"
    elif colors_question:
        kind = "catalog_colors" if has_active_request or category else "style_advice"
    elif alternatives or cheaper:
        kind = "product_change" if has_active_request or category or color else "style_advice"
    elif category and (product_ask or color or size or budget):
        kind = "product_search"
    elif color and (product_ask or category or has_active_request):
        kind = "product_change" if has_active_request else "product_search"
    elif product_ask and has_active_request:
        kind = "product_change"
    elif style:
        kind = "style_advice"
    elif category or size or budget:
        kind = "product_search"
    elif casual:
        kind = "casual"
    else:
        kind = "general"
    signals = tuple(name for name, active in (("color", bool(color)), ("category", bool(category)), ("style", style), ("occasion", bool(occasion)), ("alternative", alternatives), ("memory", recall)) if active)
    return IntentDecision(kind, 0.92 if signals else 0.62, color, category, categories, size, budget, "light" if light else "dark" if dark else None, signals, occasion, colors, mixed_color_request, requested_count, unsupported_catalog_terms, relax_constraints)
