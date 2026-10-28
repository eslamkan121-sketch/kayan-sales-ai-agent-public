"""Candidate retrieval and XLM-R relevance reranking for the KAYAN prototype."""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .catalog import Product
from .conversation_intelligence import COLOR_ALIASES, IntentDecision, arabic_fold, route

DEV_LOG = Path(__file__).resolve().parents[1] / "outputs" / "conversation_dev.log"
DEV_LOG.parent.mkdir(exist_ok=True)
LOGGER = logging.getLogger("kayan.conversation")
if not LOGGER.handlers:
    _handler = logging.FileHandler(DEV_LOG, encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    LOGGER.addHandler(_handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

LABEL_NAMES = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}
FALLBACK_LABELS = {"E": 0.56, "S": 0.22, "C": 0.14, "I": 0.08}

COLOR_WORDS = ("black", "white", "navy", "beige", "olive", "red", "blue", "yellow", "green")
CATEGORY_WORDS = ("t-shirt", "shirt", "knitwear", "jacket", "outerwear", "blazer", "hoodie", "jeans", "trousers", "skirt", "dress", "activewear", "shoes", "accessories")
SENSITIVE_ACTION_TERMS = ("buy", "purchase", "checkout", "pay", "order", "discount", "refund", "charge", "اشتري", "ادفع", "اطلب", "خصم", "استرجاع")
ALTERNATIVE_TERMS = ("alternative", "alternatives", "another", "something else", "different color", "look for more",
                     "بديل", "بدايل", "بدائل", "حاجة تانية", "لون تاني", "مش عاجبني", "دور على")
ADVICE_TERMS = ("what do you recommend", "what do you think", "style me", "what suits me", "advise",
                "إنت شايف إيه", "انت شايف ايه", "تنصحني", "إيه اللي يليق", "ايه اللي يليق")
CHEAPER_TERMS = ("cheaper", "less expensive", "lower price", "أرخص", "ارخص")
MEMORY_TERMS = ("favorite color", "what color do i like", "do you remember", "بحب لون اي", "بحب اللون", "عارف انا بحب")
FIT_TERMS = ("oversized", "regular fit", "regular", "فرق بين", "الفرق بين", "اوفر سايز", "أوفر سايز", "oversize")
WEDDING_TERMS = ("wedding", "فرح", "عرس")
BEACH_TERMS = ("beach", "بحر", "شاطئ", "شاطى")
GREETING_TERMS = ("hello", "hi", "hey", "thanks", "thank you", "مرحبا", "اهلا", "أهلا", "شكرا", "شكرًا")

ARABIC_COLORS = {"Black": "أسود", "White": "أبيض", "Navy": "كحلي", "Beige": "بيج", "Olive": "زيتي", "Red": "أحمر", "Blue": "أزرق", "Yellow": "أصفر", "Green": "أخضر"}
ARABIC_FAMILIES = {
    "Essential Cotton Crew T-Shirt": "تي شيرت أساسي قطن برقبة دائرية",
    "Relaxed Oversized T-Shirt": "تي شيرت واسع مريح",
    "Oxford Button-Down Shirt": "قميص أوكسفورد بأزرار",
    "Linen Resort Shirt": "قميص كتان صيفي",
    "Merino V-Neck Sweater": "بلوفر ميرينو برقبة V",
    "Cable Knit Cardigan": "كارديجان تريكو بضفائر",
    "Classic Denim Jacket": "جاكيت جينز كلاسيك",
    "Water-Resistant Puffer Jacket": "جاكيت مبطن مقاوم للمياه",
    "Tailored Blazer": "بليزر مفصل",
    "French Terry Hoodie": "هودي فرنش تيري",
    "Straight Fit Jeans": "جينز بقصة مستقيمة",
    "Wide-Leg Trousers": "بنطلون واسع",
    "Slim Chino Pants": "بنطلون تشينو بقصة ضيقة",
    "Pleated Midi Skirt": "جيب ميدي بليسيه",
    "Satin Slip Dress": "فستان ساتان",
    "Cotton Shirt Dress": "فستان قميص قطن",
    "Active Performance Leggings": "ليجنز رياضي",
    "Training Zip Jacket": "جاكيت رياضي بسوستة",
    "Leather-Look Sneakers": "سنيكرز بشكل جلد",
    "Canvas Tote Bag": "شنطة توت كانفاس",
}


@dataclass
class ConversationState:
    """Customer preferences retained during one visible Gradio chat session."""

    color: str | None = None
    size: str | None = None
    category: str | None = None
    categories: list[str] = field(default_factory=list)
    max_price_egp: int | None = None
    in_stock_only: bool = True
    style: str | None = None
    fit: str | None = None
    occasion: str | None = None
    favorite_color: str | None = None
    favorite_colors: list[str] = field(default_factory=list)
    preferred_style: str | None = None
    preferred_fit: str | None = None
    active_context: str = "none"
    color_tone: str | None = None
    greeted: bool = False
    last_intent: str | None = None
    prefer_cheaper: bool = False
    recommended_skus: list[str] = field(default_factory=list)
    displayed_skus: list[str] = field(default_factory=list)
    primary_sku: str | None = None
    last_recommended_sku: str | None = None
    # The product used as the factual and price-comparison anchor.  This is
    # deliberately independent from mutable search constraints.
    reference_price_sku: str | None = None
    cheaper_than_price_egp: int | None = None
    selected_sku: str | None = None
    rejected_skus: list[str] = field(default_factory=list)
    journey_stage: str = "discovery"
    excluded_skus: list[str] = field(default_factory=list)
    pending_action: str | None = None


@dataclass(frozen=True)
class ApprovedFacts:
    """The only product facts permitted in a customer-facing recommendation."""

    product_name: str | None = None
    color: str | None = None
    size: str | None = None
    price_egp: int | None = None
    in_stock: bool | None = None


class TrustGate:
    """Blocks transactions and commercial changes from the conversational model."""

    @staticmethod
    def requires_approval(message: str) -> bool:
        lowered = normalize_query(message)
        # Discussing shopping or explicitly declining to buy is not a transaction.
        if any(phrase in arabic_fold(message) for phrase in ("مش عاوز اشتري", "مش عايز اشتري", "مش هشتري", "مش حاشتري", "dont want to buy", "do not want to buy")):
            return False
        return any(term in lowered for term in SENSITIVE_ACTION_TERMS)

    @staticmethod
    def refusal(arabic: bool) -> str:
        if arabic:
            return "أقدر أساعدك تقارن بين المنتجات، لكن الدفع والطلب والخصومات والاسترجاع وتعديل المخزون تحتاج موافقة من المسؤول."
        return "I can help you compare products, but payment, ordering, discounts, refunds, and stock changes need approval from an authorized team member."


class LocalQwenConversation:
    """Optional local Qwen3 phrasing layer. Product facts are supplied by KAYAN, never generated."""

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or os.getenv("KAYAN_QWEN_MODEL_PATH", "models/Qwen3-0.6B"))
        self.model = None
        self.tokenizer = None
        self.last_response_source = "not_used"
        self.last_generated_candidate = ""
        self.status = "Qwen3 is not installed locally; catalog-grounded template replies are active."
        self._attempt_load()

    def _attempt_load(self) -> None:
        if not self.model_path.is_dir():
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, local_files_only=True, torch_dtype="auto", low_cpu_mem_usage=True
            ).eval()
            self.status = f"Local Qwen3 conversation model loaded from {self.model_path}."
        except Exception as exc:
            self.model = self.tokenizer = None
            self.status = f"Local Qwen3 unavailable ({type(exc).__name__}); catalog-grounded template replies are active."

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def respond(self, message: str, approved_answer: str, arabic: bool,
                facts: ApprovedFacts | None = None) -> tuple[str, bool, str]:
        """Let Qwen polish a fact-free approved reply, or return that reply unchanged.

        Product facts are intentionally never sent to the generative layer. This
        means an invalid model response cannot alter prices, stock, sizes, or
        catalog availability.
        """
        if not self.available:
            return approved_answer, False, "qwen_unavailable"
        try:
            import torch
            system = (
                "You are a friendly clothing-store assistant. Rewrite the approved reply in the customer's language "
                "so it sounds natural, concise, and warm. Preserve its meaning. Do not add any product, price, size, "
                "color, stock, availability, discount, product ID, payment, or order facts. Return only the reply."
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Customer: {message}\nApproved reply: {approved_answer}"}]
            # Disable Qwen3's long internal reasoning mode for a responsive retail chat.
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=72, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            self.last_generated_candidate = response
            forbidden = ("price", "egp", "stock", "available", "size", "black", "white", "navy", "beige", "olive", "red", "blue", "in stock",
                         "سعر", "مقاس", "متاح", "مخزون", "اسود", "أسود", "أبيض", "ابيض", "كحلي", "بيج", "زيتي", "أحمر", "احمر", "حمر", "ازرق", "أزرق")
            has_arabic = bool(re.search(r"[\u0600-\u06FF]", response))
            has_latin = bool(re.search(r"[A-Za-z]", response))
            if not response or len(response) > 360:
                return approved_answer, False, "empty_or_overlong"
            if (arabic and (not has_arabic or has_latin)) or (not arabic and has_arabic):
                return approved_answer, False, "language_mismatch"
            approved_folded = arabic_fold(approved_answer)
            response_folded = arabic_fold(response)
            exact_anchors = [
                term for term in ("الاحمر", "الاسود", "الابيض", "الكحلي", "البيج", "الزيتي", "الازرق", "للفرح", "للبحر")
                if term in approved_folded
            ]
            if any(anchor not in response_folded for anchor in exact_anchors):
                return approved_answer, False, "lost_approved_anchor"
            # Styling terms may be translated naturally in Arabic. Requiring
            # the English label verbatim rejected fluent Qwen output such as
            # "القصة الواسعة" / "القصة العادية" despite preserving meaning.
            translated_anchors = {
                "oversized": ("oversized", "القصة الواسعة", "القصة الواسع", "القصة الاوسع", "واسع", "واسعة", "اوسع"),
                "regular fit": ("regular fit", "القصة العادية", "القصة العادي", "قصة عادية", "عادي", "عادية", "اقرب للجسم"),
            }
            for anchor, equivalents in translated_anchors.items():
                if anchor in approved_folded and not any(arabic_fold(value) in response_folded for value in equivalents):
                    return approved_answer, False, "lost_approved_anchor"
            # Qwen receives no product facts. If it invents any, the deterministic
            # response remains the sole customer-visible source of those facts.
            # A fact word already present in the approved text is safe to paraphrase.
            if facts is None:
                if any(token in response_folded and token not in approved_folded for token in forbidden):
                    return approved_answer, False, "unapproved_product_fact"
                approved_numbers = re.findall(r"\d+", approved_folded)
                if any(number not in approved_numbers for number in re.findall(r"\d+", response_folded)):
                    return approved_answer, False, "unapproved_product_fact"
            return response, True, "accepted"
        except Exception:
            return approved_answer, False, "qwen_exception"


def normalize_query(text: str) -> str:
    replacements = {
        "جاكيت شتوي مبطن": "puffer jacket", "جاكيت شتوي": "puffer jacket",
        "جاكيت مبطن": "puffer jacket",
        "تي شيرت": "t shirt", "تيشيرت": "t shirt", "تيشرت": "t shirt", "تيشيرتت": "t shirt", "تيشرتت": "t shirt",
        "الأسود": "black", "الاسود": "black", "اسود": "black", "أسود": "black",
        "الأبيض": "white", "الابيض": "white", "ابيض": "white", "أبيض": "white",
        "الأحمر": "red", "الاحمر": "red", "أحمر": "red", "احمر": "red", "حمر": "red", "احمار": "red",
        "الأزرق": "blue", "الازرق": "blue", "أزرق": "blue", "ازرق": "blue", "زرق": "blue",
        "كحلي": "navy", "بيج": "beige", "زيتي": "olive", "قميص": "shirt",
        "جاكيت": "jacket", "بنطلون": "trousers", "جينز": "jeans", "فستان": "dress",
        "هودي": "hoodie", "كوتشي": "sneakers", "حذاء": "shoes", "رياضي": "activewear",
        "مقاس": "size", "ميديم": " size M ", "ميديوم": " size M ", "مديم": " size M ", "medium": " size M ",
        "سمول": " size S ", "لارج": " size L ", "إكس لارج": " size XL ", "اكس لارج": " size XL ",
        "خروجة البحر": "beach", "خروجه البحر": "beach", "فسحة على البحر": "beach", "للبحر": "beach", "بحر": "beach", "شاطئ": "beach",
        "مخطط": "stripe", "مقلم": "stripe", "عايز": "want", "عاوز": "want", "تحت": "under",
        "شتوي": "winter insulated puffer", "مبطن": "insulated puffer",
    }
    result = text.lower().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
    for source, target in replacements.items():
        result = result.replace(source, target)
    result = result.replace("t shirt", "t-shirt")
    return re.sub(r"\s+", " ", result).strip()


def is_arabic(text: str) -> bool:
    """Treat a message as Arabic when it contains Arabic letters, including Egyptian Arabic."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def localized_product_name(product: Product, arabic: bool) -> str:
    if not arabic:
        return product.name
    for english_name, translated_name in ARABIC_FAMILIES.items():
        if product.name.endswith(english_name):
            return f"{translated_name} {ARABIC_COLORS.get(product.color, product.color)}"
    return product.name


def localized_sizes(sizes: list[str], arabic: bool) -> str:
    if not arabic:
        return ", ".join(sizes)
    names = {"XS": "إكس سمول", "S": "سمول", "M": "ميديم", "L": "لارج", "XL": "إكس لارج", "XXL": "إكس إكس لارج"}
    return "، ".join(names.get(size.upper(), size) for size in sizes)


@dataclass(frozen=True)
class ConversationIntent:
    kind: str = "general"
    alternatives: bool = False
    advice: bool = False
    cheaper: bool = False
    different_color: bool = False


def detect_intent(message: str) -> ConversationIntent:
    normalized = normalize_query(message)
    lowered = message.lower()
    alternatives = any(term in lowered or term in normalized for term in ALTERNATIVE_TERMS)
    advice = any(term in lowered or term in normalized for term in ADVICE_TERMS)
    cheaper = any(term in lowered or term in normalized for term in CHEAPER_TERMS)
    different_color = "لون تاني" in lowered or "different color" in normalized or "another color" in normalized
    has_category = any(category in normalized for category in CATEGORY_WORDS)
    has_size = bool(re.search(r"(?:size|مقاس)\s*(xxl|xl|xs|[sml]|[0-9]{2})\b", normalized, re.I))
    has_budget = bool(re.search(r"(?:under|below|less than|تحت)\s*(\d{2,5})", normalized, re.I))
    favorite_statement = ("بحب اللون" in lowered or "i love" in lowered or "my favorite" in lowered)
    memory_question = (any(term in lowered or term in normalized for term in MEMORY_TERMS) or ("عارف" in lowered and "بحب" in lowered)) and not favorite_statement
    fit_explanation = any(term in lowered or term in normalized for term in FIT_TERMS) and ("فرق" in lowered or "difference" in normalized or "regular" in normalized)
    occasion_advice = any(term in lowered or term in normalized for term in WEDDING_TERMS + BEACH_TERMS)
    greeting = any(term == lowered.strip() or lowered.strip().startswith(term + " ") for term in GREETING_TERMS)
    if memory_question:
        kind = "memory"
    elif fit_explanation:
        kind = "fit_explanation"
    elif occasion_advice:
        kind = "occasion_advice"
    elif alternatives or cheaper or different_color:
        kind = "product_change"
    elif has_category or has_size or has_budget:
        kind = "product_search"
    elif advice:
        kind = "styling_advice"
    elif greeting:
        kind = "casual"
    elif favorite_statement:
        kind = "preference"
    else:
        kind = "general"
    return ConversationIntent(kind=kind, alternatives=alternatives, advice=advice, cheaper=cheaper, different_color=different_color)


@dataclass
class ModelStatus:
    loaded: bool
    checkpoint: str | None
    message: str
    ndcg5: float | None = None


class XLMRRelevanceModel:
    """Loads the fully trained state dictionary, when present, without retraining."""

    def __init__(self, checkpoint_path: str | Path | None):
        self.status = ModelStatus(False, None, "Checkpoint not found; lexical retrieval fallback is active.")
        self.model = None
        self.tokenizer = None
        self.id2label = {0: "E", 1: "S", 2: "C", 3: "I"}
        self.max_length = 192
        if checkpoint_path:
            self._load(Path(checkpoint_path))

    def _load(self, path: Path) -> None:
        if not path.is_file():
            self.status = ModelStatus(False, str(path), "Checkpoint file is unavailable; lexical fallback is active.")
            return
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
            payload = torch.load(path, map_location="cpu", weights_only=False)
            config_data = payload["config"]
            labels = config_data["label2id"]
            normalized_labels = {str(key).upper(): int(value) for key, value in labels.items()}
            if set(normalized_labels) != {"E", "S", "C", "I"} or set(normalized_labels.values()) != set(range(4)):
                raise ValueError("Checkpoint label2id must map E/S/C/I exactly once to logits 0..3")
            self.id2label = {value: key for key, value in normalized_labels.items()}
            self.max_length = int(config_data.get("max_length", 192))
            # The export carries the tokenizer/config from the original pinned Kaggle XLM-R
            # mount.  Prefer those local files, so inference is reproducible and has no
            # internet dependency.  The public base is only a documented fallback for a
            # user-supplied checkpoint without its companion tokenizer assets.
            assets = path.parent
            source = str(assets) if (assets / "tokenizer.json").is_file() and (assets / "config.json").is_file() else "FacebookAI/xlm-roberta-base"
            self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=source != "FacebookAI/xlm-roberta-base")
            config = AutoConfig.from_pretrained(source, num_labels=len(labels), local_files_only=source != "FacebookAI/xlm-roberta-base")
            # This prototype also runs on small local machines.  Creating the usual
            # CPU model first would momentarily duplicate this 1.11 GB checkpoint.
            # Meta construction plus ``assign=True`` attaches the verified tensors
            # directly and avoids that temporary memory spike.
            with torch.device("meta"):
                self.model = AutoModelForSequenceClassification.from_config(config)
            self.model.load_state_dict(payload["model"], strict=True, assign=True)
            # ``assign`` intentionally leaves non-persistent positional buffers on
            # the meta device.  Recreate those tiny buffers on CPU; all learned
            # parameters above still reference the checkpoint tensors directly.
            embeddings = self.model.roberta.embeddings
            embeddings.position_ids = torch.arange(
                config.max_position_embeddings, dtype=torch.long
            ).expand((1, -1))
            embeddings.token_type_ids = torch.zeros(
                (1, config.max_position_embeddings), dtype=torch.long
            )
            self.model.eval()
            metrics = payload.get("metrics", {}).get("ranking", {}).get("macro", {})
            self.status = ModelStatus(True, str(path), "Trained XLM-R checkpoint loaded.", metrics.get("ndcg5"))
        except Exception as exc:  # Preserve a usable UI even on machines without model dependencies.
            self.model = self.tokenizer = None
            self.status = ModelStatus(False, str(path), f"Checkpoint could not load: {type(exc).__name__}; lexical fallback is active.")

    def predict_detailed(self, query: str, products: list[Product]) -> list[dict]:
        """Return actual checkpoint logits, softmax confidence, and label argmax.

        Unlike ``predict`` this method never returns demo fallback probabilities:
        an unavailable model yields an empty list, so the UI cannot imply that a
        lexical fallback is a trained classifier prediction.
        """
        if not self.model or not self.tokenizer or not self.status.loaded:
            return []
        import torch
        result: list[dict] = []
        with torch.inference_mode():
            for start in range(0, len(products), 2):
                group = products[start:start + 2]
                batch = self.tokenizer(
                    [query] * len(group), [product.searchable_text() for product in group],
                    truncation=True, max_length=self.max_length, padding=True, return_tensors="pt",
                )
                logits_tensor = self.model(**batch).logits
                if logits_tensor.ndim != 2 or logits_tensor.shape[-1] != 4:
                    raise ValueError(f"Expected four ESCI logits; received shape {tuple(logits_tensor.shape)}")
                probabilities = torch.softmax(logits_tensor, dim=-1).cpu().numpy()
                logits = logits_tensor.cpu().numpy()
                for logit_row, probability_row in zip(logits, probabilities):
                    named_logits = {self.id2label[index]: float(value) for index, value in enumerate(logit_row)}
                    named_probabilities = {self.id2label[index]: float(value) for index, value in enumerate(probability_row)}
                    predicted = max(named_logits, key=named_logits.get)
                    result.append({
                        "logits": named_logits,
                        "probabilities": named_probabilities,
                        "predicted_label": predicted,
                        "confidence": named_probabilities[predicted],
                    })
        return result

    def predict(self, query: str, products: list[Product]) -> list[dict[str, float]]:
        if not self.model or not self.tokenizer:
            return [dict(FALLBACK_LABELS) for _ in products]
        import torch
        rows: list[dict[str, float]] = []
        # Tiny batches keep the prototype usable on an 8 GB laptop as well as a
        # larger server.  Retrieval has already reduced the catalog to candidates.
        with torch.inference_mode():
            for start in range(0, len(products), 2):
                group = products[start:start + 2]
                batch = self.tokenizer(
                    [query] * len(group), [product.searchable_text() for product in group],
                    truncation=True, max_length=self.max_length, padding=True, return_tensors="pt",
                )
                probabilities = torch.softmax(self.model(**batch).logits, dim=-1).cpu().numpy()
                rows.extend(
                    {self.id2label[index]: float(value) for index, value in enumerate(row)}
                    for row in probabilities
                )
        return rows


class SalesAgent:
    def __init__(self, products: Iterable[Product], checkpoint_path: str | Path | None = None,
                 qwen_path: str | Path | None = None):
        self.products = list(products)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([product.searchable_text() for product in self.products])
        self.model = XLMRRelevanceModel(checkpoint_path)
        self.conversation = LocalQwenConversation(qwen_path)
        self.last_approved_facts: ApprovedFacts | None = None
        self.last_validation_reason = "not_run"

    def retrieve(self, query: str, allowed_indices: list[int] | None = None, limit: int = 20) -> list[tuple[Product, float]]:
        normalized = normalize_query(query)
        scores = cosine_similarity(self.vectorizer.transform([normalized]), self.matrix)[0]
        # Catalog facets make a precise customer category/color request win over an
        # incidental shared word such as "cotton" or "shirt".
        for index, product in enumerate(self.products):
            if product.category.lower() in normalized:
                scores[index] += 1.5
            if product.color.lower() in normalized:
                scores[index] += 0.4
            # Some catalog families deliberately use a broader category (for example,
            # a puffer belongs to outerwear).  Preserve an explicit customer request
            # for a distinctive product family rather than rewarding generic "jacket".
            product_text = product.searchable_text().lower()
            for family in ("puffer", "blazer", "hoodie", "sneakers", "jeans", "dress", "t-shirt"):
                if family in normalized and family in product_text:
                    scores[index] += 2.0
        if allowed_indices is None:
            allowed_indices = list(range(len(self.products)))
        indices = sorted(allowed_indices, key=lambda index: scores[index], reverse=True)[:limit]
        return [(self.products[int(index)], float(scores[int(index)])) for index in indices]

    def _hard_filter_indices(self, state: ConversationState | None, only_in_stock: bool) -> list[int]:
        """Filter actual catalog facts before retrieval/XLM-R. No constraint is ever relaxed here."""
        state = state or ConversationState(in_stock_only=only_in_stock)
        required_stock = only_in_stock or state.in_stock_only
        indices: list[int] = []
        for index, product in enumerate(self.products):
            if required_stock and product.stock <= 0:
                continue
            if state.color and product.color.lower() != state.color:
                continue
            if state.color_tone == "light" and product.color not in {"White", "Beige"}:
                continue
            if state.color_tone == "dark" and product.color not in {"Black", "Navy", "Olive"}:
                continue
            if state.size and state.size.upper() not in {size.upper() for size in product.sizes}:
                continue
            allowed_categories = state.categories or ([state.category] if state.category else [])
            if allowed_categories and product.category.lower() not in allowed_categories:
                continue
            if state.max_price_egp is not None and product.price_egp > state.max_price_egp:
                continue
            if state.cheaper_than_price_egp is not None and product.price_egp >= state.cheaper_than_price_egp:
                continue
            indices.append(index)
        return indices

    def search(self, query: str, top_k: int = 8, only_in_stock: bool = True,
               state: ConversationState | None = None, exclude_skus: set[str] | None = None) -> list[dict]:
        allowed_indices = self._hard_filter_indices(state, only_in_stock)
        if exclude_skus:
            allowed_indices = [i for i in allowed_indices if self.products[i].sku not in exclude_skus]
        if not allowed_indices:
            return []
        candidates = self.retrieve(query, allowed_indices=allowed_indices)
        products = [product for product, _ in candidates]
        predictions = self.model.predict(normalize_query(query), products)
        ranked = []
        for (product, retrieval_score), probabilities in zip(candidates, predictions):
            relation = max(probabilities, key=probabilities.get)
            # Candidate retrieval keeps catalog match strong; XLM-R probability ranks the relation.
            score = 0.35 * retrieval_score + 0.65 * probabilities.get("E", 0.0) + 0.12 * probabilities.get("S", 0.0)
            ranked.append({
                "product": product,
                "retrieval_score": retrieval_score,
                "relation": LABEL_NAMES[relation],
                "relation_code": relation,
                "probabilities": probabilities,
                "rank_score": score,
            })
        if state and state.prefer_cheaper:
            return sorted(ranked, key=lambda item: (item["product"].price_egp, -item["rank_score"]))[:top_k]
        return sorted(ranked, key=lambda item: item["rank_score"], reverse=True)[:top_k]

    def extract_preferences(self, message: str, state: ConversationState | None = None,
                            intent: IntentDecision | None = None) -> ConversationState:
        updated = ConversationState(**({
            **state.__dict__,
            "recommended_skus": list(state.recommended_skus),
            "displayed_skus": list(state.displayed_skus),
            "excluded_skus": list(state.excluded_skus),
            "favorite_colors": list(state.favorite_colors),
            "categories": list(state.categories),
            "rejected_skus": list(state.rejected_skus),
        } if state else {}))
        intent = intent or route(message, updated.active_context == "shopping")
        normalized = normalize_query(message)
        if intent.kind == "unsupported_catalog":
            # Unsupported attributes/categories open a new request boundary;
            # never let a prior unavailable colour answer leak into it.
            updated.active_context = "shopping"
            updated.color = updated.category = updated.size = None
            updated.categories = []
            updated.color_tone = None
            updated.max_price_egp = None
            updated.cheaper_than_price_egp = None
            updated.prefer_cheaper = False
        # A new styling topic must not inherit an old product's color/category.
        if intent.kind in {"occasion_advice", "fit_advice", "style_advice"}:
            updated.active_context = "styling"
            updated.color = updated.category = updated.size = None
            updated.categories = []
            updated.max_price_egp = None
            updated.color_tone = None
        # Work-interview shopping is a fresh product task.  Retain durable
        # preferences, but clear constraints belonging to the prior item and
        # apply an explainable, bounded workwear category mapping.
        if intent.kind == "workwear_search":
            updated.active_context = "shopping"
            updated.color = updated.category = updated.size = None
            updated.categories = ["shirt", "blazer"]
            updated.max_price_egp = None
            updated.color_tone = None
            updated.prefer_cheaper = False
            updated.excluded_skus = []
            updated.rejected_skus = []
            updated.occasion = "work"
        elif intent.kind == "occasion_search":
            # Product-oriented occasion requests are new shopping tasks, not
            # refinements of the last SKU. Map only to explainable catalog
            # categories and leave unsupported attributes unspecified.
            updated.active_context = "shopping"
            updated.color = updated.category = updated.size = None
            updated.max_price_egp = None
            updated.color_tone = None
            updated.prefer_cheaper = False
            updated.excluded_skus = []
            updated.rejected_skus = []
            updated.occasion = intent.occasion
            occasion_categories = {
                "beach": ["shirt", "t-shirt"], "sports": ["activewear", "shoes"],
                "travel": ["hoodie", "jacket", "trousers"], "university": ["t-shirt", "hoodie", "jeans"],
            }
            updated.categories = occasion_categories.get(intent.occasion or "", [])
        elif intent.occasion:
            updated.occasion = intent.occasion
        if intent.color:
            updated.color = intent.color
        if intent.color_tone:
            updated.color_tone = intent.color_tone
        favorite_statement = intent.kind == "preference_update" or (
            bool(intent.color) and any(marker in arabic_fold(message) for marker in ("بحب", "favorite", " i love", "احب"))
        )
        if favorite_statement and intent.colors:
            # Favorite colors are durable customer preferences, unlike the
            # active single-color product filter above.
            updated.favorite_color = intent.colors[-1]
            updated.favorite_colors = list(dict.fromkeys(updated.favorite_colors + list(intent.colors)))
        if intent.kind == "product_change" and intent.color is None and any(
            marker in arabic_fold(message) for marker in ("لون تاني", "الوان تاني", "other color", "different color")
        ):
            updated.color = None
        # Naming a different category starts a fresh item request unless the
        # customer explicitly carries a colour forward in the same message.
        prior_categories = set(state.categories or ([state.category] if state and state.category else [])) if state else set()
        if intent.category and intent.color is None and prior_categories and intent.category not in prior_categories:
            updated.color = updated.size = None
            updated.color_tone = None
            updated.max_price_egp = None
            updated.cheaper_than_price_egp = None
            updated.prefer_cheaper = False
            updated.excluded_skus = []
            updated.rejected_skus = []
        if intent.categories:
            updated.categories = list(intent.categories)
        if intent.category:
            updated.category = intent.category
            updated.active_context = "shopping"
        if intent.size:
            updated.size = intent.size
        if intent.max_price_egp is not None:
            # A standalone budget starts a new shopping task. It must not use
            # the previous item’s color/category and then claim an old SKU is
            # within a new, incompatible budget.
            if intent.relax_constraints or (not intent.color and not intent.category and not intent.categories):
                updated.color = updated.category = updated.size = None
                updated.categories = []
                updated.color_tone = None
                updated.cheaper_than_price_egp = None
                updated.prefer_cheaper = False
            updated.max_price_egp = intent.max_price_egp
        if any(term in normalized for term in ("available", "in stock", "متاح")):
            updated.in_stock_only = True
        if any(term in normalized for term in ("oversized", "oversize", "اوفر سايز", "واسع")):
            updated.fit = "oversized"
        elif any(term in normalized for term in ("slim", "سليم", "مظبوط")):
            updated.fit = "slim"
        if any(term in normalized for term in ("casual", "كاجوال")):
            updated.style = "casual"
            updated.preferred_style = "casual"
        elif any(term in normalized for term in ("classic", "formal", "كلاسيك", "رسمي")):
            updated.style = "classic"
            updated.preferred_style = "classic"
        # A specific routed occasion (for example “خروجة البحر”) wins over a
        # generic substring such as “خروج”, which would otherwise overwrite it.
        if intent.occasion is None:
            if any(term in normalized for term in ("work", "office", "شغل", "مكتب", "عمل", "موعد عمل")):
                updated.occasion = "work"
            elif any(term in normalized for term in ("outing", "going out", "خروج", "سهرة")):
                updated.occasion = "outing"
            elif any(term in normalized for term in ("occasion", "event", "مناسبة")):
                updated.occasion = "occasion"
        if intent.kind == "product_change" and intent.max_price_egp is None and any(marker in arabic_fold(message) for marker in ("ارخص", "أرخص", "cheaper")):
            updated.prefer_cheaper = True
            reference_sku = ((state.reference_price_sku or state.primary_sku or state.last_recommended_sku) if state else None)
            reference = next((product for product in self.products if product.sku == reference_sku), None)
            if reference:
                updated.reference_price_sku = reference.sku
                updated.cheaper_than_price_egp = reference.price_egp
                ceiling = reference.price_egp - 1
                updated.max_price_egp = min(updated.max_price_egp, ceiling) if updated.max_price_egp is not None else ceiling
        explicit_alternative = any(marker in arabic_fold(message) for marker in (
            "بديل", "بدايل", "بدائل", "حاجة تانية", "حاجه تانية", "مش عاجب", "مش مناسب",
            "different", "alternative", "another",
        ))
        if intent.kind == "product_change" and explicit_alternative and updated.recommended_skus:
            updated.excluded_skus = list(dict.fromkeys(updated.excluded_skus + updated.recommended_skus))
        updated.last_intent = intent.kind
        return updated

    @staticmethod
    def _unavailable_message(state: ConversationState, arabic: bool) -> str:
        if state.cheaper_than_price_egp is not None:
            # This method is static for ordinary no-match messages; the price
            # ceiling itself is still a verified state value.
            if arabic:
                return f"مفيش اختيار متاح يطابق المواصفات ويكون أرخص من {state.cheaper_than_price_egp} جنيه. تحب توسّع اللون أو النوع؟"
            return f"No available catalog product meets those requirements and costs less than {state.cheaper_than_price_egp} EGP. Would you like to broaden the color or product type?"
        if state.max_price_egp is not None:
            limit = state.max_price_egp + 1
            if arabic:
                return f"مفيش منتج متاح في الكتالوج الحالي سعره أقل من {limit} جنيه. تحب تزود الميزانية أو تحدد نوع؟"
            return f"No available catalog product costs less than {limit} EGP. Would you like to raise the budget or choose a product type?"
        if arabic:
            color = {"red": "أحمر", "black": "أسود", "white": "أبيض", "navy": "كحلي", "beige": "بيج", "olive": "زيتي", "blue": "أزرق", "yellow": "أصفر", "green": "أخضر"}.get(state.color or "", "")
            details = f" باللون {color}" if color else ""
            return f"للأسف مفيش منتج متاح{details} يطابق المواصفات المطلوبة في الكتالوج الحالي. تحب أدوّر لك على لون تاني أو بديل؟"
        details = f" in {state.color}" if state.color else ""
        return f"Sorry, no available catalog product{details} meets those requirements. Would you like me to look for another color or an alternative?"

    @staticmethod
    def _style_question(arabic: bool) -> str:
        return ("تحب ستايل كاجوال ولا كلاسيك؟ وعاوزها للخروج ولا للشغل؟"
                if arabic else "Do you prefer casual or classic style, and is it for going out or for work?")

    @staticmethod
    def _styling_note(state: ConversationState, arabic: bool) -> str:
        if state.occasion == "work":
            return "القميص أو البليزر بيكون اختيار مرتب للشغل." if arabic else "A shirt or blazer is a polished choice for work."
        if state.occasion == "outing":
            return "التيشيرت أو الهودي مع جينز اختيار مريح للخروج." if arabic else "A T-shirt or hoodie with jeans is an easy option for going out."
        if state.fit == "oversized":
            return "القصّة الواسعة هتديك لوك كاجوال ومريح." if arabic else "An oversized fit gives a relaxed, casual look."
        if state.fit == "slim":
            return "القصّة السليم بتدي شكل مرتب ومحدد." if arabic else "A slim fit gives a neater, more tailored look."
        return ""

    @staticmethod
    def _alternative_state(state: ConversationState) -> ConversationState:
        """First fallback: retain category/size/budget/availability, relax color only."""
        return replace(state, color=None, color_tone=None, recommended_skus=list(state.recommended_skus), excluded_skus=list(state.excluded_skus))

    def _record_recommendations(self, state: ConversationState, results: list[dict]) -> None:
        # Conversation copy names one primary product.  Keep that explicit so a
        # later singular “its price/sizes” cannot drift to a reranked candidate.
        primary = results[0]["product"].sku if results else None
        if primary:
            shown_skus = [item["product"].sku for item in results]
            state.recommended_skus = list(dict.fromkeys(state.recommended_skus + shown_skus))
            state.displayed_skus = shown_skus
            state.primary_sku = primary
            state.last_recommended_sku = primary
            state.reference_price_sku = primary
            state.cheaper_than_price_egp = None
            state.pending_action = "change_attribute"

    @staticmethod
    def _catalog_context(results: list[dict]) -> str:
        return "\n".join(
            f"{item['product'].sku}: {item['product'].name}; category={item['product'].category}; "
            f"color={item['product'].color}; sizes={', '.join(item['product'].sizes)}; "
            f"price={item['product'].price_egp} EGP; availability={'in stock' if item['product'].stock else 'out of stock'}; "
            f"relation={item['relation']}"
            for item in results[:4]
        )

    @staticmethod
    def _approved_product_facts(product: Product, state: ConversationState) -> ApprovedFacts:
        return ApprovedFacts(product_name=product.name, color=product.color,
                             size=state.size, price_egp=product.price_egp,
                             in_stock=product.stock > 0)

    @staticmethod
    def _validate_catalog_reply(reply: str, facts: ApprovedFacts) -> tuple[bool, str]:
        """Validate the rendered factual portion before it reaches the customer."""
        if not facts.product_name or facts.in_stock is not True:
            return False, "missing_or_out_of_stock_fact"
        if ("جنيه" in reply or "EGP" in reply) and str(facts.price_egp) not in reply:
            return False, "price_not_in_approved_facts"
        if facts.size and "المقاسات المتاحة" not in reply and "Available sizes:" not in reply:
            local_size = localized_sizes([facts.size], "المقاس" in reply)
            if local_size not in reply and facts.size not in reply:
                return False, "size_not_in_approved_facts"
        return True, "approved_facts_rendered"

    @staticmethod
    def _favorite_color_name(color: str | None, arabic: bool) -> str:
        if not color:
            return ""
        english = color.capitalize()
        if not arabic:
            return english
        return {"Red": "الأحمر", "Black": "الأسود", "White": "الأبيض", "Navy": "الكحلي", "Beige": "البيج", "Olive": "الزيتي", "Blue": "الأزرق", "Yellow": "الأصفر", "Green": "الأخضر"}.get(english, english)

    def _polish_non_catalog_reply(self, query: str, approved: str, arabic: bool) -> tuple[str, str, str]:
        """Use Qwen only where no catalog fact can be distorted; retain safe fallback otherwise."""
        reply, used_qwen, reason = self.conversation.respond(query, approved, arabic)
        self.conversation.last_response_source = "qwen" if used_qwen else "guarded_fallback"
        return reply, self.conversation.last_response_source, reason

    @staticmethod
    def _audit(intent: IntentDecision, before: ConversationState, after: ConversationState,
               source: str, reason: str, searched: bool) -> None:
        tracked = ("color", "size", "category", "max_price_egp", "color_tone", "occasion", "favorite_color", "active_context")
        changed = [name for name in tracked if getattr(before, name) != getattr(after, name)]
        # Do not write raw customer messages or values; this is developer-only
        # behavioral telemetry for repeatable debugging.
        LOGGER.info(json.dumps({"intent": intent.kind, "confidence": intent.confidence,
                                 "signals": intent.signals, "state_changes": changed,
                                 "searched_catalog": searched, "response_source": source,
                                 "rejection_reason": reason}, ensure_ascii=False))

    def _memory_reply(self, state: ConversationState, arabic: bool) -> str:
        colors = state.favorite_colors or ([state.favorite_color] if state.favorite_color else [])
        if colors:
            names = [self._favorite_color_name(color, arabic) for color in colors]
            listed = " و".join(names) if arabic else " and ".join(names)
            return f"أيوه، فاكر إن اللون {listed} من ألوانك المفضلة." if arabic else f"Yes — you told me your favorite colors are {listed}."
        return "لسه ما قلتليش لونك المفضل. تحب تقولي؟" if arabic else "You have not told me a favorite color yet. Would you like to share it?"

    @staticmethod
    def _fit_reply(arabic: bool) -> str:
        if arabic:
            return "الـOversized أوسع من الجسم ويدي راحة ولوك كاجوال. الـRegular Fit متوازن: لا واسع قوي ولا لازق. الاختيار حسب اللوك اللي بتحبه، مش حسب رقم على الميزان."
        return "Oversized is roomier and more relaxed; regular fit is balanced, neither loose nor tight. Choose by the look and comfort you like, not a number on a scale."

    @staticmethod
    def _occasion_reply(state: ConversationState, query: str, arabic: bool) -> str:
        if route(query).occasion == "casual":
            return ("للخروجة، تيشيرت أو قميص مريح مع جينز اختيار بسيط ومرتب. بتحب اللوك كاجوال ولا أشيك شوية؟"
                    if arabic else "For going out, a comfortable T-shirt or shirt with jeans is an easy polished choice. Do you prefer casual or smarter?")
        if state.occasion == "wedding":
            state.occasion = "wedding"
            if arabic:
                return "ألف مبروك مقدمًا! للفرح اختار لوك مرتب زي قميص بقصة نظيفة مع بنطلون مناسب، أو بليزر لو المناسبة رسمية. طولك ووزنك مش كفاية عشان أحدد مقاس؛ جرّب المقاس اللي مرتاح فيه. بتحب اللوك كاجوال شيك ولا رسمي؟"
            return "Congratulations in advance! For a wedding, a clean-cut shirt with smart trousers works well, or a blazer if it is formal. Height and weight alone are not enough to choose a size. Do you prefer smart-casual or formal?"
        if state.occasion == "beach":
            if arabic:
                return "للبحر خليك في قماش خفيف وتهوية كويسة، زي قميص كتان أو تيشيرت مريح مع شورت مناسب. تحب الألوان الهادية ولا لون جريء؟"
            return "For the beach, choose breathable lightweight pieces such as a linen shirt or a comfortable T-shirt with suitable shorts. Do you prefer neutral or bold colors?"
        if state.occasion == "university":
            return ("للجامعة، اختار قطعة مريحة وسهلة التنسيق زي تيشيرت أو هودي مع جينز. لو تحب أدوّر في الكتالوج، قولّي المقاس واللون اللي تفضله."
                    if arabic else "For university, choose comfortable, easy-to-style pieces such as a tee or hoodie with jeans. Tell me your preferred size and color if you want catalog options.")
        if state.occasion == "travel":
            return ("للسفر، الأولوية للراحة والطبقات الخفيفة. جاكيت خفيف أو هودي مع بنطلون مريح اختيار عملي. تحب أدوّر لك على خيارات؟"
                    if arabic else "For travel, prioritize comfort and light layers. A light jacket or hoodie with comfortable trousers is practical. Would you like catalog options?")
        if state.occasion == "sports":
            return ("للرياضة، اختار قطع مخصصة للحركة والتهوية. أقدر أدوّر لك على القطع الرياضية المتاحة لو تحدد المقاس أو اللون."
                    if arabic else "For sports, choose items designed for movement and breathability. I can look for available activewear if you share a size or color.")
        if arabic:
            return "قولّي المناسبة والستايل اللي تحبه، وأنا أساعدك تختار لوك مناسب."
        return "Tell me the occasion and I can help you choose an appropriate outfit."

    @staticmethod
    def _preference_reply(state: ConversationState, arabic: bool) -> str:
        colors = state.favorite_colors or ([state.favorite_color] if state.favorite_color else [])
        names = [SalesAgent._favorite_color_name(color, arabic) for color in colors]
        color = " و".join(names) if arabic else " and ".join(names)
        if arabic:
            return f"تمام، هفتكر إن اللون {color} من ألوانك المفضلة." if color else "تمام، هفتكر تفضيلاتك."
        return f"Got it — I’ll remember that {color} is among your favorite colors." if color else "Got it — I’ll remember that preference."

    @staticmethod
    def _general_reply(arabic: bool) -> str:
        return "أكيد، قولّي محتاج مساعدة في اختيار ستايل ولا مقارنة بين قطعتين؟" if arabic else "Of course. Would you like help choosing a style or comparing two pieces?"

    def _product_reply(self, query: str, state: ConversationState, intent: IntentDecision, arabic: bool) -> tuple[str, list[dict]]:
        folded_query = arabic_fold(query)
        is_alternative = intent.kind == "product_change" and any(marker in folded_query for marker in (
            "بديل", "بدايل", "بدائل", "حاجة تانية", "حاجه تانية", "مش عاجب", "مش مناسب",
            "different", "alternative", "another",
        ))
        exclude_skus = set(state.excluded_skus) | set(state.rejected_skus)
        requested_count = max(1, min(intent.requested_count or 1, 5))
        results = self.search(query, top_k=requested_count if intent.requested_count else 8,
                              state=state, exclude_skus=exclude_skus)
        fallback_used = False
        # Never replace an explicit requested color with another color.  Color
        # relaxation is permitted only once the customer asks for alternatives
        # or broadens the search without naming a new color.
        allow_color_relaxation = (
            intent.color is None
            and intent.kind == "product_change"
            and any(marker in arabic_fold(query) for marker in (
                "بديل", "بدايل", "بدائل", "حاجة تانية", "حاجه تانية",
                "حاجة مختلفة", "حاجه مختلفة", "لون تاني", "الوان تاني",
                "alternative", "another",
            ))
        )
        if not results and allow_color_relaxation:
            alternate_state = self._alternative_state(state)
            results = self.search(query, state=alternate_state, exclude_skus=exclude_skus)
            fallback_used = bool(results)
        if not results:
            state.pending_action = "alternative_offer"
            return self._unavailable_message(state, arabic), []

        # A normal chat reply names one product; expose only that one as a
        # displayed identity. Explicit list requests are the sole multi-item
        # conversation response and record every named SKU in order.
        shown_results = results[:requested_count] if intent.requested_count else results[:1]
        chosen = shown_results[0]
        product = chosen["product"]
        self._record_recommendations(state, shown_results)
        price_requested = bool(re.search(r"(?:under|below|less than|تحت)\s*\d+", normalize_query(query))) or any(
            term in query.lower() for term in ("price", "cost", "سعر", "اسعار", "أسعار", "الاسعار", "الأسعار", "بكام")
        )
        sizes_requested = any(term in arabic_fold(query) for term in ("المقاسات", "مقاسات", "المقاس", "sizes", "available sizes"))
        requested_color = self._favorite_color_name(state.color, arabic)
        name = localized_product_name(product, arabic)
        selected_size = localized_sizes([state.size], arabic) if state.size else ""
        if intent.requested_count:
            count = len(shown_results)
            if arabic:
                lines = [f"{index}. {localized_product_name(item['product'], True)}" for index, item in enumerate(shown_results, start=1)]
                reply = f"لقيت {count} منتج مطابق للطلب:" + "\n" + "\n".join(lines)
                if count < requested_count:
                    reply += f"\nالكتالوج فيه {count} فقط يطابق المواصفات، ومش هكرّر منتجات." 
            else:
                lines = [f"{index}. {item['product'].name}" for index, item in enumerate(shown_results, start=1)]
                reply = f"I found {count} matching product{'s' if count != 1 else ''}:\n" + "\n".join(lines)
                if count < requested_count:
                    reply += f"\nOnly {count} catalog item{'s' if count != 1 else ''} match, so I have not duplicated any products."
            self.last_approved_facts = self._approved_product_facts(product, state)
            self.last_validation_reason = "catalog_grounded_requested_list"
            self.conversation.last_response_source = "catalog_grounded"
            return reply, shown_results
        if arabic:
            # Once an earlier result was rejected, do not repeat the same
            # availability explanation. The preceding reply already made the
            # color mismatch clear; this turn should simply move the journey
            # forward with a different, catalog-grounded option.
            if is_alternative:
                opening = f"في بديل مختلف: {name}."
            elif fallback_used:
                opening = f"للأسف اللون {requested_color} مش متوفر بالمواصفات دي، لكن عندي بديل مناسب: {name}."
            else:
                opening = f"أنسب اختيار لطلبك هو {name}."
            detail = f" متوفر بالمقاس {selected_size}." if state.size else ""
            if price_requested:
                detail += f" سعره {product.price_egp} جنيه."
            if sizes_requested and not state.size:
                detail += f" المقاسات المتاحة: {localized_sizes(product.sizes, True)}."
            follow = " تحب أدوّر على بديل تاني؟" if is_alternative else " تحب لون تاني ولا تفضّل ستايل مختلف؟"
        else:
            if is_alternative:
                opening = f"Here is a different alternative: {name}."
            elif fallback_used:
                opening = f"The {requested_color} option is unavailable with those details, but I found a suitable alternative: {name}."
            else:
                opening = f"A strong match for your request is {name}."
            detail = f" It is available in your selected size, {selected_size}." if state.size else ""
            if price_requested:
                detail += f" It is {product.price_egp} EGP."
            if sizes_requested and not state.size:
                detail += f" Available sizes: {localized_sizes(product.sizes, False)}."
            follow = " Would you like another alternative?" if is_alternative else " Would you prefer another color or style?"
        reply = opening + detail + follow
        facts = self._approved_product_facts(product, state)
        valid, reason = self._validate_catalog_reply(reply, facts)
        self.last_approved_facts = facts
        self.last_validation_reason = reason
        if not valid:
            # This should not normally be reached because facts are rendered from
            # the same product object. It prevents a future NLG refactor from
            # accidentally exposing unapproved catalog claims.
            return ("لا أقدر أؤكد تفاصيل المنتج دلوقتي." if arabic else "I cannot confirm the product details right now."), []
        self.conversation.last_response_source = "catalog_grounded"
        return reply, shown_results

    def _reference_reply(self, query: str, state: ConversationState, arabic: bool,
                         product: Product | None = None) -> tuple[str, list[dict]]:
        """Answer price/stock only for a verified catalog identity.

        ``product`` is supplied for an explicit reference such as “the black
        trousers”; otherwise the last displayed SKU is used.  This prevents a
        price question from entering generic retrieval and silently becoming a
        claim about a substitute.
        """
        product = product or next((item for item in self.products if item.sku == (state.primary_sku or state.last_recommended_sku)), None)
        if product is None:
            return ("مش قادر أحدد المنتج اللي تقصده، قولّي اسمه أو ابعتهولي تاني." if arabic else "I cannot identify that product; please share its name again."), []
        folded = arabic_fold(query)
        asks_price = any(term in folded for term in ("بكام", "سعر", "اسعار", "price", "cost"))
        asks_stock = any(term in folded for term in ("متوفر", "متاح", "available", "مقاس", "sizes", "المقاسات"))
        asks_sizes = any(term in folded for term in ("مقاس كام", "المقاسات", "مقاسات", "sizes", "what sizes", "which sizes"))
        asks_color = any(term in folded for term in ("اللون", "لونه", "لون ايه", "color"))
        size_match = re.search(r"(?:size|مقاس)\s*(xxl|xl|xs|[sml])\b", normalize_query(query), re.I)
        size = size_match.group(1).upper() if size_match else state.size
        parts: list[str] = []
        if asks_price:
            parts.append(f"سعر {localized_product_name(product, True)} هو {product.price_egp} جنيه." if arabic else f"{product.name} is {product.price_egp} EGP.")
        if asks_stock:
            if asks_sizes:
                if arabic:
                    parts.append(f"المقاسات المتاحة: {localized_sizes(product.sizes, True)}.")
                else:
                    parts.append(f"Available sizes: {localized_sizes(product.sizes, False)}.")
            elif size:
                available = product.stock > 0 and size in {value.upper() for value in product.sizes}
                if arabic:
                    parts.append(f"ومقاس {localized_sizes([size], True)} {'متوفر' if available else 'مش متوفر'}.")
                else:
                    parts.append(f"Size {size} is {'available' if available else 'not available'}.")
            elif arabic:
                parts.append("المنتج متاح حاليًا." if product.stock > 0 else "المنتج غير متاح حاليًا.")
            else:
                parts.append("It is currently in stock." if product.stock > 0 else "It is currently unavailable.")
        if asks_color:
            if arabic:
                parts.append(f"لونه {ARABIC_COLORS.get(product.color, product.color)}.")
            else:
                parts.append(f"Its color is {product.color}.")
        if not parts:
            parts.append("تقصد تعرف السعر ولا المقاس المتاح؟" if arabic else "Would you like the price or available sizes?")
        self.last_approved_facts = self._approved_product_facts(product, state)
        self.last_validation_reason = "verified_product_identity_resolved"
        return " ".join(parts), [{"product": product}]

    def _resolve_explicit_product_identity(self, state: ConversationState,
                                           intent: IntentDecision) -> Product | None:
        """Resolve an explicitly named/color-filtered product without ranking.

        Prefer a product the customer has actually seen.  This makes a later
        question such as “سعر البنطلون الاسود بكام” resolve to the original
        black-trousers SKU even if a colour-browsing turn occurred afterward.
        """
        category = intent.category
        color = intent.color
        for sku in reversed(state.recommended_skus):
            product = next((item for item in self.products if item.sku == sku), None)
            if product and (not category or product.category.lower() == category) and (not color or product.color.lower() == color):
                return product
        matches = [product for product in self.products
                   if (not category or product.category.lower() == category)
                   and (not color or product.color.lower() == color)]
        if len(matches) == 1:
            return matches[0]
        return None

    def _available_colors_reply(self, state: ConversationState, arabic: bool) -> tuple[str, list[dict]]:
        """List actual available catalogue colours without replacing selection."""
        categories = state.categories or ([state.category] if state.category else [])
        products = [product for product in self.products
                    if product.stock > 0 and (not categories or product.category.lower() in categories)]
        colors = list(dict.fromkeys(product.color for product in products))
        if not colors:
            return ("مفيش ألوان متاحة بالمواصفات دي دلوقتي." if arabic else "There are no available colors for those requirements right now."), []
        if arabic:
            localized = "، ".join(ARABIC_COLORS.get(color, color) for color in colors)
            subject = "للبناطيل" if categories == ["trousers"] else "للمواصفات دي"
            return f"الألوان المتاحة {subject}: {localized}. تحب تشوف لون معيّن؟", []
        return f"Available colors for these requirements: {', '.join(colors)}. Would you like to see a particular color?", []

    def _last_product(self, state: ConversationState) -> Product | None:
        return next((product for product in self.products if product.sku == (state.primary_sku or state.last_recommended_sku)), None)

    def _details_reply(self, state: ConversationState, arabic: bool) -> tuple[str, list[dict]]:
        product = self._last_product(state)
        if not product:
            return ("قولّي اسم المنتج أو خلّيني أرشح لك حاجة الأول." if arabic else "Tell me the product name, or let me recommend one first."), []
        if arabic:
            reply = f"{localized_product_name(product, True)}: {product.arabic_aliases}. لونه {ARABIC_COLORS.get(product.color, product.color)} ومتوافر حاليًا." if product.stock else f"{localized_product_name(product, True)}: {product.arabic_aliases}. لكنه غير متاح حاليًا."
        else:
            reply = f"{product.name}: {product.description} It is {'in stock' if product.stock else 'currently unavailable'}."
        self.last_validation_reason = "last_product_details_resolved"
        return reply, [{"product": product}]

    def _comparison_reply(self, state: ConversationState, arabic: bool) -> tuple[str, list[dict]]:
        skus = state.recommended_skus[-2:]
        products = [product for product in self.products if product.sku in skus]
        if len(products) < 2:
            return ("لسه محتاج أوريك اختيارين عشان أقارن بينهم." if arabic else "I need to show you two options before I can compare them."), []
        first, second = products
        if arabic:
            reply = (f"{localized_product_name(first, True)} لونه {ARABIC_COLORS.get(first.color, first.color)} وسعره {first.price_egp} جنيه، "
                     f"أما {localized_product_name(second, True)} فسعره {second.price_egp} جنيه. "
                     "اختار حسب اللون والستايل اللي مرتاح له؛ أقدر أساعدك تحسم بينهم.")
        else:
            reply = (f"{first.name} is {first.price_egp} EGP, while {second.name} is {second.price_egp} EGP. "
                     "Choose based on the color and style you prefer; I can help you decide.")
        self.last_validation_reason = "comparison_grounded"
        return reply, [{"product": first}, {"product": second}]

    def customer_reply(self, query: str, state: ConversationState | None = None) -> tuple[str, list[dict], ConversationState]:
        arabic = is_arabic(query)
        if TrustGate.requires_approval(query):
            return TrustGate.refusal(arabic), [], state or ConversationState()
        before = ConversationState(**({**state.__dict__, "recommended_skus": list(state.recommended_skus),
                                       "displayed_skus": list(state.displayed_skus),
                                       "excluded_skus": list(state.excluded_skus), "favorite_colors": list(state.favorite_colors),
                                       "categories": list(state.categories), "rejected_skus": list(state.rejected_skus)} if state else {}))
        intent = route(query, before.active_context != "none")
        updated = self.extract_preferences(query, before, intent)

        folded_query = arabic_fold(query)
        references_last_product = bool(updated.primary_sku or updated.last_recommended_sku) and any(
            term in folded_query for term in ("ده", "دي", "التيشيرت ده", "القميص ده", "سعره", "مقاساته", "لونه", "this one", "that one", "its price", "its sizes")
        )
        asks_factual_detail = any(term in folded_query for term in ("بكام", "سعر", "اسعار", "متوفر", "متاح", "مقاس", "المقاسات", "price", "available", "size"))
        asks_price = any(term in folded_query for term in ("بكام", "سعر", "اسعار", "price", "cost"))
        # An explicit budget is a fresh hard-filtered shopping request, even
        # when its wording contains “price”. It must never be routed as a
        # factual follow-up to the old SKU.
        if references_last_product and asks_factual_detail and intent.max_price_egp is None:
            reply, results = self._reference_reply(query, updated, arabic)
            self._audit(intent, before, updated, "catalog_grounded", self.last_validation_reason, searched=False)
            return reply, results, updated
        # Explicit identities (“سعر البنطلون الأسود بكام”) are factual
        # questions, not new product searches. Resolve the identity from the
        # visible recommendation history before permitting a price/stock claim.
        if asks_price and intent.category and intent.color:
            product = self._resolve_explicit_product_identity(updated, intent)
            if product:
                reply, results = self._reference_reply(query, updated, arabic, product=product)
                self._audit(intent, before, updated, "catalog_grounded", self.last_validation_reason, searched=False)
                return reply, results, updated
            reply = ("مش قادر أحدد منتج واحد بالمواصفات دي، قولّي الاسم أو اختار من المنتجات اللي ظهرت لك." if arabic
                     else "I cannot identify one catalog product from those details; please share its name or choose one of the shown products.")
            self._audit(intent, before, updated, "identity_guard", "explicit_product_identity_ambiguous", searched=False)
            return reply, [], updated

        # Shopping-journey actions are resolved before generic advice so feedback,
        # comparison, and product details retain the exact product the customer saw.
        if references_last_product and any(term in folded_query for term in ("اختار", "هاخد", "هختار", "select", "choose")):
            updated.selected_sku = updated.last_recommended_sku
            updated.journey_stage = "selected"
            reply = "اختيار موفق. لو تحب أقولك السعر أو أتأكد من المقاس، أنا معاك." if arabic else "Great choice. I can share the price or check your size whenever you like."
            self._audit(intent, before, updated, "journey", "last_product_selected", searched=False)
            return reply, [], updated
        if any(term in folded_query for term in ("تفاصيل", "الخامة", "الوصف", "details", "material", "description")) and updated.last_recommended_sku:
            reply, results = self._details_reply(updated, arabic)
            self._audit(intent, before, updated, "catalog_grounded", self.last_validation_reason, searched=False)
            return reply, results, updated
        if any(term in folded_query for term in ("قارن", "الفرق بين", "مين احسن", "compare", "which is better")) and not any(term in folded_query for term in ("oversized", "regular fit", "اوفر", "اوفرسايز")):
            reply, results = self._comparison_reply(updated, arabic)
            self._audit(intent, before, updated, "catalog_grounded", self.last_validation_reason, searched=False)
            return reply, results, updated
        rejected = any(term in folded_query for term in ("مش عاجب", "مش مناسب", "حاجة مختلفة", "حاجه مختلفة", "غيره", "غيرها", "different"))
        if rejected and updated.last_recommended_sku:
            updated.rejected_skus = list(dict.fromkeys(updated.rejected_skus + [updated.last_recommended_sku]))
            updated.excluded_skus = list(dict.fromkeys(updated.excluded_skus + [updated.last_recommended_sku]))
            updated.active_context = "shopping"
            reply, results = self._product_reply(query, updated, IntentDecision("product_change", 0.9), arabic)
            updated.journey_stage = "exploring"
            self._audit(intent, before, updated, "catalog_grounded", self.last_validation_reason, searched=True)
            return reply, results, updated
        def non_catalog(approved: str) -> tuple[str, list[dict], ConversationState]:
            reply, source, reason = self._polish_non_catalog_reply(query, approved, arabic)
            self._audit(intent, before, updated, source, reason, searched=False)
            return reply, [], updated

        if intent.kind == "unsupported_catalog":
            terms = set(intent.unsupported_catalog_terms)
            if arabic:
                labels = []
                if "striped" in terms:
                    labels.append("التيشيرت المخطط")
                if "suit" in terms:
                    labels.append("البدلة")
                requested = " أو ".join(labels)
                approved = f"الكتالوج الحالي مش بيسجّل {requested} كمنتج أو صفة قابلة للبحث، فمش هقدر أؤكد توفره. تحب بديل من الأنواع الموجودة؟"
            else:
                requested = " and ".join(sorted(terms))
                approved = f"The current catalog does not represent {requested} as a searchable verified category or attribute, so I cannot confirm availability. Would you like a supported alternative?"
            return non_catalog(approved)

        # “Mixed” is a product attribute request, not a license to blend two
        # remembered favorite colors. The current catalog represents one color
        # per SKU, so answer honestly without a fabricated search result.
        if intent.kind == "mixed_color_request":
            updated.color = None
            updated.active_context = "shopping"
            approved = ("الكتالوج الحالي مفيهوش منتج متسجل بأكتر من لون. تحب أدوّر لك على الأصفر أو الأزرق كل لون لوحده؟"
                        if arabic else "The current catalog has no verified multi-color product. Would you like me to look for yellow or blue separately?")
            return non_catalog(approved)

        # A short affirmative is meaningful only if it answers our immediately
        # pending offer.  It is never treated as blanket consent elsewhere.
        affirmative = folded_query in {"ايوه", "ايوة", "اه", "اها", "ياريت", "ايوه ياريت", "yes", "yes please", "sure"}
        if before.pending_action == "alternative_offer" and affirmative:
            updated.color = None
            updated.pending_action = "alternative_scope"
            approved = ("أكيد. تحب نغيّر اللون ولا النوع؟" if arabic else "Certainly. Would you like to change the color or the product type?")
            return non_catalog(approved)
        if before.pending_action == "alternative_scope" and folded_query in {"اللون", "لون", "color"}:
            updated.color = None
            updated.pending_action = "color_choice"
            reply, results = self._available_colors_reply(updated, arabic)
            self._audit(intent, before, updated, "catalog_grounded", "pending_color_choice", searched=False)
            return reply, results, updated

        # Shipping, delivery, and order execution are not connected to this
        # prototype.  Keep this separate from TrustGate's sensitive-action
        # refusal so customers get an honest integration limitation.
        if any(term in folded_query for term in ("شحن", "توصيل", "عنوان", "shipping", "delivery", "tracking")):
            approved = ("حالياً ما عنديش نظام شحن أو تتبع طلبات متصل بالكتالوج. أقدر أساعدك تختار منتج، وبعدها فريق المتجر يكمّل معاك."
                        if arabic else "This prototype is not connected to shipping, delivery, or order tracking. I can help choose a product, then a store team member can take over.")
            return non_catalog(approved)

        # The chat currently presents one named recommendation per turn. Never
        # guess an ordinal reference to a product that was not actually named.
        if any(term in folded_query for term in ("التاني", "الثاني", "رقم 2", "second one", "the second")):
            approved = ("عشان أحدد المنتج التاني بدقة، قولّي اسمه أو اختاره من نتائج البحث الظاهرة عندك."
                        if arabic else "To identify the second product accurately, please tell me its name or select it from the visible search results.")
            return non_catalog(approved)

        if intent.kind == "catalog_colors":
            reply, results = self._available_colors_reply(updated, arabic)
            self._audit(intent, before, updated, "catalog_grounded", "available_colors_from_catalog", searched=False)
            return reply, results, updated

        if any(term in folded_query for term in ("سينما", "cinema", "movie")):
            updated.occasion = "cinema"
            updated.active_context = "styling"
            approved = "للسينما خليك في لوك مريح ومرتب: تيشيرت أو قميص خفيف مع جينز مناسب. بتحب اللوك يكون هادي ولا فيه لون جريء؟" if arabic else "For the cinema, go for a comfortable polished look: a tee or light shirt with suitable jeans. Do you prefer a neutral or bolder color?"
            return non_catalog(approved)

        # Direct-answer intents deliberately bypass catalog retrieval, preventing
        # an earlier product request from hijacking a new conversation topic.
        if intent.kind == "memory_question":
            return non_catalog(self._memory_reply(updated, arabic))
        if intent.kind == "fit_advice":
            return non_catalog(self._fit_reply(arabic))
        if intent.kind == "occasion_advice":
            return non_catalog(self._occasion_reply(updated, query, arabic))
        if intent.kind == "preference_update":
            return non_catalog(self._preference_reply(updated, arabic))
        if intent.kind == "casual":
            folded = arabic_fold(query)
            if any(term in folded for term in ("شكرا", "شكرًا", "thanks", "thank you")):
                approved = "العفو، تحت أمرك." if arabic else "You’re welcome — happy to help."
            else:
                approved = "أهلاً بيك، قولّي محتاج إيه؟" if arabic else "Hello — how can I help?"
            return non_catalog(approved)
        if intent.kind == "style_advice":
            if updated.color_tone == "light":
                approved = "لو بتحب الألوان الفاتحة، الأبيض والبيج اختيارات سهلة وشيك. تحبها في تيشيرت ولا قميص؟" if arabic else "For light colors, white and beige are easy, polished choices. Would you prefer a T-shirt or a shirt?"
            elif updated.color_tone == "dark":
                approved = "الألوان الغامقة زي الكحلي والأسود عملية ومرتبة. المناسبة خروجة ولا شغل؟" if arabic else "Dark colors such as navy and black are versatile and polished. Is this for going out or for work?"
            elif updated.occasion == "work" and updated.style == "classic":
                approved = "للشغل بستايل كلاسيك، القميص المرتب أو البليزر مع بنطلون مناسب اختيار ممتاز. لو تحب أوريك المتاح منهم قولّي." if arabic else "For a classic work look, a polished shirt or blazer with suitable trousers is a strong choice. I can show available options if you like."
            elif updated.occasion == "work":
                approved = "بما إن المناسبة شغل، أقدر أرتب لك لوك مناسب. بتحب ستايل كلاسيك ولا كاجوال مرتب؟" if arabic else "Since it is for work, I can help build a suitable look. Do you prefer classic or smart-casual?"
            elif updated.occasion == "outing":
                approved = "للخروجة، تيشيرت أو قميص مريح مع جينز اختيار بسيط ومرتب. بتحب اللوك كاجوال ولا أشيك شوية؟" if arabic else "For going out, a comfortable T-shirt or shirt with jeans is an easy polished choice. Do you prefer casual or smarter?"
            else:
                approved = self._style_question(arabic)
            return non_catalog(approved)
        if intent.kind == "general":
            if updated.active_context == "shopping":
                approved = "تمام، قولي تحب تغيّر اللون ولا المقاس ولا الستايل؟" if arabic else "Sure — would you like to change the color, size, or style?"
            else:
                approved = "أكيد، قولي المناسبة أو الستايل اللي في بالك وأنا أساعدك تختار." if arabic else "Of course — tell me the occasion or style you have in mind and I’ll help you choose."
            return non_catalog(approved)

        # Product search/change is the only path that touches catalog retrieval or XLM-R.
        updated.active_context = "shopping"
        if updated.style == "classic" and updated.occasion == "work" and not updated.categories:
            updated.categories = ["shirt", "blazer"]
        reply, results = self._product_reply(query, updated, intent, arabic)
        self.conversation.last_response_source = "catalog_grounded"
        self._audit(intent, before, updated, "catalog_grounded", self.last_validation_reason, searched=True)
        return reply, results, updated

    def status_text(self) -> str:
        status = self.model.status
        metric = f" Best validation NDCG@5: {status.ndcg5:.5f}." if status.ndcg5 is not None else ""
        return status.message + metric + " " + self.conversation.status
