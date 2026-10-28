"""KAYAN Sales Agent with a separately labelled workbook-backed demo builder."""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from functools import lru_cache
from datetime import datetime
from html import escape
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo
from uuid import uuid4

import gradio as gr
import pandas as pd

from src.agent import ConversationState, SalesAgent, is_arabic, normalize_query
from src.catalog import Product, build_catalog, catalog_dataframe
from src.conversation_intelligence import arabic_fold
from src.demo_catalog import DemoCatalog, DemoCatalogValidationError, load_demo_catalog
from src.demo_conversation import DemoConversationEngine
from src.product_search import product_candidates, rank_product_candidates
from src.presentation import format_egp, format_integer, format_piasters, localize_category, localize_color, localize_product_name
from src.session_store import CustomerSessionStore

ROOT = Path(__file__).resolve().parent


def _source_fingerprint() -> str:
    """Fingerprint the loaded application sources for local build diagnostics."""
    digest = hashlib.sha256()
    source_files = [ROOT / "app.py", *sorted((ROOT / "src").glob("*.py"))]
    for source_file in source_files:
        digest.update(source_file.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(source_file.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
        digest.update(b"\0")
    return digest.hexdigest()[:16]


BUILD_FINGERPRINT = _source_fingerprint()
STARTED_AT = datetime.now(ZoneInfo("Africa/Cairo")).isoformat(timespec="seconds")


def log_build_identity() -> None:
    """Print a non-sensitive identity line for the process serving this build."""
    print(
        "KAYAN_BUILD "
        + json.dumps(
            {
                "project_path": str(ROOT),
                "source_fingerprint": BUILD_FINGERPRINT,
                "pid": os.getpid(),
                "started_at": STARTED_AT,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    # Log before workbook, SQLite, and model initialization so operators can
    # identify the exact source build even if startup later fails or is slow.
    log_build_identity()


EXPORT_DIR = Path(os.getenv("KAYAN_EXPORT_DIR", str(ROOT / "outputs"))); EXPORT_DIR.mkdir(exist_ok=True)
DAY_REPORT_DIR = Path(os.getenv("KAYAN_DAY_REPORT_DIR", str(EXPORT_DIR / "end_of_day"))); DAY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = os.getenv("KAYAN_CHECKPOINT", str(ROOT / "models" / "best.pt"))
LEGACY_CATALOG = build_catalog(); LEGACY_DF = catalog_dataframe(LEGACY_CATALOG)
LEGACY_XLSX = EXPORT_DIR / "KAYAN_Clothing_Catalog.xlsx"
with pd.ExcelWriter(LEGACY_XLSX, engine="openpyxl") as writer:
    LEGACY_DF.to_excel(writer, sheet_name="Legacy XLM-R Catalog", index=False)

DEMO_WORKBOOK = ROOT / "catalog_preview_review" / "outputs" / "KAYAN_catalog_color_variants_NEW.xlsx"
DEMO_ASSETS = ROOT / "catalog_preview_review" / "preview_assets"
try:
    DEMO_CATALOG: DemoCatalog | None = load_demo_catalog(DEMO_WORKBOOK, DEMO_ASSETS)
    DEMO_ERROR = ""
except DemoCatalogValidationError as exc:
    DEMO_CATALOG, DEMO_ERROR = None, str(exc)

STORE = CustomerSessionStore(os.getenv("KAYAN_SESSION_DB", str(EXPORT_DIR / "kayan_customer_sessions.sqlite3")))
active = STORE.list_customers()
DEFAULT_CUSTOMER_ID = STORE.active_customer_id() or (active[0]["customer_id"] if active else None)
if DEFAULT_CUSTOMER_ID:
    STORE.set_active_customer(DEFAULT_CUSTOMER_ID)
AGENT = SalesAgent(LEGACY_CATALOG, CHECKPOINT, os.getenv("KAYAN_QWEN_MODEL_PATH", str(ROOT / "models" / "Qwen3-0.6B")))


def business_date() -> str: return datetime.now(ZoneInfo("Africa/Cairo")).date().isoformat()
def day() -> str:
    record=STORE.active_business_day()
    return record["day_id"] if record else (STORE.current_business_day() or {}).get("day_id", business_date())
def money(piasters: int, arabic: bool = False) -> str: return format_piasters(piasters, arabic)
def choices(): return [(f"{c['display_name']} ({c['customer_id'][-4:]})", c["customer_id"]) for c in STORE.list_customers()]
def removed_choices(): return [(f"{c['display_name']} — archived ({c['customer_id']})", c["customer_id"]) for c in STORE.removed_customers()]
def outfit_choices(customer_id: str): return [(o["display_name"], o["outfit_id"]) for o in STORE.list_outfits(customer_id)]
def first_outfit(customer_id: str):
    outfits = STORE.list_outfits(customer_id)
    # The persistence table predates the cart-only UI. Each customer now uses
    # one invisible cart record is used per active customer.
    return outfits[0]["outfit_id"] if outfits else STORE.create_outfit(customer_id, "Shopping cart")


def _feedback_signal(message: str) -> tuple[str, str | None] | None:
    text=arabic_fold(message or "").lower()
    reject_terms=("مش عاجبني","مش عاجباني","مش عاجب","مش مناسب","مش حلو","مش عايز ده","مش عاوز ده",
                  "غالي اوي","سعره غالي","غالي عليا","مش على ذوقي","مش ستايلي","too expensive","too pricey",
                  "wrong color","wrong size","not interested","don't like","do not like","not for me")
    if any(term in text for term in reject_terms):
        if any(term in text for term in ("غالي","سعره عالي","too expensive","too pricey")):reason="too_expensive"
        elif any(term in text for term in ("لون","color")):reason="wrong_color"
        elif any(term in text for term in ("مقاس","size")):reason="wrong_size"
        elif any(term in text for term in ("ستايل","ذوق","style")):reason="unwanted_style"
        else:reason="not_liked"
        return "rejected",reason
    accepted_terms=("هختار","هختاره","اختارته","هاخده","هاخدها","هشتريه","هشتريها","ضيفه للسلة","ضيفها للسلة",
                    "add to cart","i'll take","i will take","i'll choose","i will choose","i choose this")
    if any(term in text for term in accepted_terms):return "accepted",None
    positive_terms=("عجبني","عجباني","عاجبني","حلو جدا","حلو اوي","حلوه جدا","جميل جدا","مناسب لي",
                    "بحبه","i like","love it","looks good","beautiful","great choice")
    if any(term in text for term in positive_terms):return "positive",None
    return None


def _record_marketing_signals(customer_id: str, business_day_id: str, message: str,
                              prior_state: dict, updated_state: dict, action: dict, reply: str) -> None:
    customer=STORE.customer(customer_id)
    if not customer:return
    label=customer["display_name"]
    demo=updated_state.get("demo_conversation",{})
    active=demo.get("active",{}) if isinstance(demo,dict) else {}
    presentation=action.get("presentation") if isinstance(action,dict) else None
    # A product-search turn creates one customer/product interest signal even
    # if the same result set is rendered through images or factual follow-ups.
    if isinstance(presentation,dict):
        seen=set()
        for variant_id in presentation.get("variant_ids",[]):
            variant=DEMO_CATALOG.get(str(variant_id)) if DEMO_CATALOG else None
            if not variant or variant.product_id in seen:continue
            seen.add(variant.product_id)
            payload={"customer_id":customer_id,"customer_label":label,"product_id":variant.product_id,
                     "variant_id":variant.variant_id,"category":variant.category,"color":variant.color,
                     "size":variant.size,"request_context":{k:active[k] for k in
                     ("categories","category","color","size","max_price_egp","price_range_egp","occasion","pattern") if k in active}}
            STORE.publish_communication_event("sales.customer_interest_recorded","sales","marketing",
                business_day_id,payload,customer_id,
                interaction_key=f"interest:{customer_id}:{business_day_id}:{variant.product_id}")
    # Explicit feedback only. Merely viewing, selecting, asking a price, or
    # ignoring a suggestion does not become approval or rejection.
    signal=_feedback_signal(message)
    if signal:
        feedback_type,reason=signal
        prior=prior_state.get("demo_conversation",prior_state) if isinstance(prior_state,dict) else {}
        variant_id=(action.get("add_variant_id") if isinstance(action,dict) else None) or (
            prior.get("focused_variant_id") if isinstance(prior,dict) else None) or demo.get("focused_variant_id")
        variant=DEMO_CATALOG.get(str(variant_id)) if DEMO_CATALOG and variant_id else None
        if variant:
            payload={"customer_id":customer_id,"customer_label":label,"product_id":variant.product_id,
                     "variant_id":variant.variant_id,"feedback_type":feedback_type,"reason":reason,
                     "category":variant.category,"color":variant.color,"size":variant.size}
            STORE.publish_communication_event("sales.customer_feedback_recorded","sales","marketing",
                business_day_id,payload,customer_id,
                interaction_key=f"feedback:{customer_id}:{business_day_id}:{variant.product_id}:{variant.variant_id}:{feedback_type}:{reason or ''}")
    normalized_reply=arabic_fold(reply or "").lower()
    no_match=any(term in normalized_reply for term in ("مفيش منتج","ملقتش منتج","غير متاح","could not find","no matching","no in-stock"))
    displayed=(demo.get("displayed_variant_ids",[]) if isinstance(demo,dict) else [])
    if no_match and not displayed and isinstance(active,dict) and active:
        criteria={k:active[k] for k in ("categories","category","color","size","max_price_egp","price_range_egp","occasion","pattern","mixed") if k in active}
        if criteria:
            criteria_key=hashlib.sha256(json.dumps(criteria,sort_keys=True,ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
            STORE.publish_communication_event("sales.unmet_demand_recorded","sales","marketing",business_day_id,
                {"customer_id":customer_id,"customer_label":label,"criteria":criteria,
                 "request_hash":criteria_key},customer_id,
                interaction_key=f"unmet:{customer_id}:{business_day_id}:{criteria_key}")

@lru_cache(maxsize=64)
def _compact_image_data(path_text: str, modified_ns: int) -> str:
    """Serve a compact in-memory preview without altering catalog image files."""
    path = Path(path_text)
    if not path.is_file():
        return ""
    try:
        from PIL import Image
        with Image.open(path) as source:
            source.thumbnail((640, 640))
            if source.mode in {"RGBA", "LA"}:
                canvas = Image.new("RGB", source.size, "#f4f1eb")
                alpha = source.getchannel("A")
                canvas.paste(source.convert("RGB"), mask=alpha)
                source = canvas
            else:
                source = source.convert("RGB")
            buffer = BytesIO()
            source.save(buffer, format="JPEG", quality=78, optimize=True)
        return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception:
        # Preserve display support for an unexpected image format/library issue.
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def image_data(path: Path | None) -> str:
    if not path or not path.is_file(): return ""
    return _compact_image_data(str(path.resolve()), path.stat().st_mtime_ns)

def render_product_cards(record: dict) -> str:
    """Build a safe, responsive chat card grid from exact catalog Variant IDs."""
    if not DEMO_CATALOG:
        return ""
    fields = set(record.get("fields") or ["name"])
    arabic = record.get("language") == "ar"
    cards = []
    for variant_id in record.get("variant_ids", []):
        variant = DEMO_CATALOG.get(str(variant_id))
        if not variant:
            continue
        name = localize_product_name(variant.name, arabic)
        image = image_data(variant.image_path) if "image" in fields else ""
        picture = (f"<img src='{image}' alt='{escape(name)} — "
                  f"{escape(localize_color(variant.color, arabic))}' loading='lazy'>" if image else
                  ("<div class='kayan-card-image-missing'>الصورة التوضيحية غير متاحة</div>" if arabic else
                   "<div class='kayan-card-image-missing'>Illustrative image unavailable</div>")) if "image" in fields else ""
        details = []
        if "category" in fields:
            details.append(localize_category(variant.category, arabic))
        if "colors" in fields:
            colors = sorted({v.color for v in DEMO_CATALOG.variants if v.product_id == variant.product_id and v.in_stock})
            details.append(("الألوان: " if arabic else "Colors: ") + ", ".join(localize_color(c, arabic) for c in colors))
        elif "color" in fields:
            details.append(localize_color(variant.color, arabic))
        if "sizes" in fields:
            sizes = sorted({v.size for v in DEMO_CATALOG.variants if v.product_id == variant.product_id and v.color == variant.color and v.in_stock}, key=lambda value: (len(value), value))
            details.append(("المقاسات: " if arabic else "Sizes: ") + ", ".join(sizes))
        if "price" in fields:
            details.append(format_egp(variant.price_egp, arabic))
        detail_html = "".join(f"<div class='kayan-card-detail'>{escape(value)}</div>" for value in details)
        image_label = ("صورة توضيحية · DEMO" if arabic else "Illustrative DEMO image") if image else ""
        image_note = f"<small class='kayan-card-disclosure'>{image_label}</small>" if image else ""
        cards.append(
            "<article class='kayan-product-card'>" + picture +
            "<div class='kayan-card-copy'><b>" + escape(name) + "</b>" + detail_html + image_note + "</div></article>"
        )
    if not cards:
        return ""
    direction = "rtl" if arabic else "ltr"
    return (f"<div dir='{direction}' class='kayan-product-grid'>" + "".join(cards) + "</div>"
            "<style>.kayan-product-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:8px 0;width:100%}"
            ".kayan-product-card{overflow:hidden;border:1px solid #334155;border-radius:12px;background:#111a22;color:#f8fafc}"
            ".kayan-product-card img{display:block;width:100%;height:150px;object-fit:contain;background:#f4f1eb}"
            ".kayan-card-image-missing{height:100px;display:grid;place-items:center;color:#94a3b8}"
            ".kayan-card-copy{padding:10px;line-height:1.5}.kayan-card-detail{color:#cbd5e1;font-size:.92em;margin-top:3px}"
            ".kayan-card-disclosure{display:block;color:#aeb8c5;margin-top:6px;font-size:.8em}</style>")

def render_history(history):
    """Render readable transcript text and the same compact product-card UI.

    Older persisted turns used standalone ``gr.Image`` blocks. Convert those
    records to the shared card renderer by exact saved Variant ID so restored
    chats use the same presentation and retain catalog identity/provenance.
    """
    rendered=[]
    for saved in history or []:
        message={"role":saved.get("role","assistant"),"content":saved.get("content","")}
        cards = saved.get("_cards")
        image_records = saved.get("_images", [])
        if not cards and image_records:
            legacy_ids = [str(item.get("variant_id", "")) for item in image_records if item.get("variant_id")]
            if legacy_ids:
                last_user=next((turn.get("content","") for turn in reversed(rendered) if turn.get("role")=="user"),"")
                cards={"variant_ids":legacy_ids,"fields":["name","image"],
                       "language":"ar" if is_arabic(last_user) else "en"}
        if cards:
            card_html = render_product_cards(cards)
            if card_html:
                # Keep the complete, meaningful reply as a real text block.
                # Copy/export tools must never derive transcript text from an
                # HTML component (which some Gradio versions stringify as
                # "undefined").
                plain_text = str(message["content"] or "").strip()
                if not plain_text:
                    plain_text = "المنتجات الظاهرة في الصور:" if cards.get("language") == "ar" else "Products shown in the images:"
                # Preserve a readable accessible caption in Gradio's component
                # metadata while keeping the compact card visually unlabeled.
                caption = "صورة المنتج التوضيحية" if cards.get("language") == "ar" else "Illustrative product card"
                first_variant_id = next(iter(cards.get("variant_ids", [])), None)
                first_variant = DEMO_CATALOG.get(str(first_variant_id)) if first_variant_id and DEMO_CATALOG else None
                if first_variant:
                    product_caption = localize_product_name(first_variant.name, cards.get("language") == "ar")
                    color_caption = localize_color(first_variant.color, cards.get("language") == "ar")
                    caption = f"{product_caption} · {color_caption} · {caption}"
                message["content"] = [plain_text, gr.HTML(value=card_html, label=caption, show_label=False)]
        rendered.append(message)
    return rendered

def cards(variants, arabic: bool = False):
    if not variants:
        message="ملقتش منتجات مطابقة في كتالوج الـDEMO. الكتالوج القديم متاح كخيار منفصل." if arabic else "No matching demo variants. The legacy catalog remains available as fallback."
        return f"<div class='empty'>{escape(message)}</div>"
    html = []
    for v in variants:
        name=localize_product_name(v.name,arabic); color=localize_color(v.color,arabic); category=localize_category(v.category,arabic)
        price=format_egp(v.price_egp,arabic)
        stock=(f"مخزون تجريبي: {format_integer(v.stock_qty,True)} · صورة توضيحية" if arabic else f"Demo stock: {v.stock_qty} · illustrative image")
        image=image_data(v.image_path)
        picture=f"<img alt='Illustrative demo catalog image' src='{image}'/>" if image else "<div class='tiny-missing'>الصورة غير متاحة</div>"
        html.append(f"<div class='product-card'>{picture}<div><b>{escape(name)}</b><br>{escape(category)} · {escape(color)} · {escape(v.size)}<br><b>{escape(price)}</b><br><small>{escape(stock)}</small></div></div>")
    return "<div class='cards'>" + "".join(html) + "</div>"

def rank_demo_variants(query: str, only_in_stock: bool = True, limit: int = 12):
    """Retrieve workbook variants first, then classify/re-rank with trained XLM-R."""
    if not DEMO_CATALOG:
        return []
    normalized_query = normalize_query(query)
    candidates = DEMO_CATALOG.search(normalized_query, only_in_stock, limit=20)
    if not candidates:
        return []
    model_products = [
        Product(
            sku=v.variant_id, name=v.name, description=v.description, category=v.category,
            color=v.color, sizes=[v.size], price_egp=v.price_egp, stock=v.stock_qty,
            arabic_aliases=f"{v.color} {v.category} {v.occasion}",
            data_source="Workbook-backed KAYAN demo catalog",
        ) for v in candidates
    ]
    probabilities = AGENT.model.predict(normalized_query, model_products)
    scored = []
    for variant, probability in zip(candidates, probabilities):
        # XLM-R remains the relevance classifier. Lexical retrieval only bounds
        # the workbook candidates, while the trained E/S probabilities rank them.
        score = probability.get("E", 0.0) + 0.2 * probability.get("S", 0.0)
        scored.append((score, variant))
    return [variant for _score, variant in sorted(scored, key=lambda row: row[0], reverse=True)[:limit]]

def rank_demo_candidates(query: str, candidates, limit: int = 24):
    """Rank already hard-filtered workbook variants with the pinned XLM-R model."""
    if not candidates:
        return []
    normalized_query = normalize_query(query)
    model_products = [
        Product(sku=v.variant_id, name=v.name, description=v.description, category=v.category,
                color=v.color, sizes=[v.size], price_egp=v.price_egp, stock=v.stock_qty,
                arabic_aliases=f"{v.color} {v.category} {v.occasion}",
                data_source="Workbook-backed KAYAN demo catalog")
        for v in candidates
    ]
    probabilities = AGENT.model.predict(normalized_query, model_products)
    return [v for _score, v in sorted(
        ((p.get("E", 0.0) + 0.2 * p.get("S", 0.0), v) for v, p in zip(candidates, probabilities)),
        key=lambda row: row[0], reverse=True,
    )[:limit]]

DEMO_CONVERSATION = DemoConversationEngine(DEMO_CATALOG, rank_demo_candidates, EXPORT_DIR / "demo_conversation_dev.log", qwen=AGENT.conversation.respond) if DEMO_CATALOG else None

def demo_search(query: str, stock_only: bool, top_k: int = 10):
    if not DEMO_CATALOG:
        return "**Catalog error:** " + escape(DEMO_ERROR), [], pd.DataFrame(), pd.DataFrame()
    normalized_query=normalize_query(query).strip() or query.strip()
    # Product Search is a relevance-classification demonstration. Score every
    # eligible Product ID in this small workbook instead of pruning candidates
    # with lexical keyword overlap before XLM-R can evaluate them.
    candidates=product_candidates(DEMO_CATALOG.variants, only_in_stock=stock_only, query=normalized_query)
    if not candidates:
        message=("لا توجد منتجات في كتالوج الـDEMO تحقق اختيار المخزون الحالي." if is_arabic(query)
                 else "No products in the selected DEMO catalog meet the current stock setting.")
        return message,[],pd.DataFrame(),pd.DataFrame()
    arabic=is_arabic(query)
    ranked=[]
    model_status=AGENT.model.status
    if model_status.loaded:
        try:
            ranked=rank_product_candidates(normalized_query,candidates,AGENT.model)
            score_source="trained_xlmr"
        except Exception as exc:
            score_source=f"unavailable:{type(exc).__name__}"
            ranked=sorted(candidates,key=lambda item:item["product_id"])
    else:
        # Candidate retrieval remains usable, but no heuristic fallback is
        # presented as a classifier prediction or XLM-R relevance score.
        ranked=sorted(candidates,key=lambda item:item["product_id"])
        score_source="xlmr_unavailable"
    shown=ranked[:min(int(top_k or 10),10)]
    rows=[]; technical_rows=[]; gallery=[]
    for rank,item in enumerate(shown,1):
        v=item["representative"]; prices=item["prices_egp"]
        price_text=(format_egp(prices[0],arabic) if len(prices)==1 else
                    f"{format_egp(prices[0],arabic)} – {format_egp(prices[-1],arabic)}")
        prediction=item.get("prediction") if score_source=="trained_xlmr" else None
        row={"Rank":(rank if prediction else None),"Product ID":item["product_id"],
            "Product name":localize_product_name(v.name,arabic),
            "Category":localize_category(v.category,arabic),
            "Colors":", ".join(localize_color(color,arabic) for color in item["colors"]),
            "Available sizes":", ".join(item["sizes"]),
            ("السعر (جنيه)" if arabic else "Price (EGP)"):price_text,
            ("مخزون تجريبي" if arabic else "Demo stock qty"):item["stock_qty"],
            ("مطابقة مواصفات الطلب" if arabic else "Catalog facet match"):(
                ("مطابق ومتوفر بالمخزون التجريبي" if arabic else "In-stock facet match") if item["in_stock_facet_match"] else
                ("مطابق لكن بدون مخزون تجريبي" if arabic else "Facet match, out of stock") if item["facet_match"] else
                ("لا توجد مطابقة مؤكدة" if arabic else "Not verified") if item["facet_match"] is not None else
                ("لا توجد مواصفات محددة" if arabic else "No explicit facets")),
            ("متوقع ESCI" if arabic else "Predicted ESCI"):(prediction["predicted_label"] if prediction else "Not scored"),
            ("درجة الترتيب" if arabic else "Relevance score"):(round(item["ranking_score"],6) if prediction else None)}
        rows.append(row)
        if prediction:
            technical_rows.append({"Rank":rank,"Product ID":item["product_id"],
                **{f"P({label})":round(prediction["probabilities"][label],6) for label in ("E","S","C","I")},
                **{f"Logit {label}":round(prediction["logits"][label],6) for label in ("E","S","C","I")}})
        image_path=item["image_path"]
        if image_path and image_path.is_file():
            caption=(f"{row['Product name']} · {row['Colors']} · صورة توضيحية DEMO" if arabic else
                     f"{v.name} · {', '.join(item['colors'])} · illustrative DEMO image")
            gallery.append((str(image_path),caption))
    if score_source=="trained_xlmr":
        mapping=", ".join(f"{label}={index}" for index,label in sorted(AGENT.model.id2label.items()))
        metric=(f" Saved validation NDCG@5: {model_status.ndcg5:.5f}." if model_status.ndcg5 is not None else "")
        status=(f"**XLM-R checkpoint:** `{model_status.checkpoint}`.{metric} Scored {len(candidates)} distinct products from workbook `Preview variants`; showing Top {len(shown)}. "
                f"The table groups actual color/size variants; prices and stock are simulated DEMO data. "
                f"ESCI is the argmax of actual checkpoint logits (mapping: `{mapping}`); probabilities and logits are in the technical view. "
                "Input is an XLM-R query/product text pair; the existing Product Search ranking score is P(E) + 0.2 × P(S), not a guaranteed match. "
                "These are inferences on the new workbook, not the model's original Amazon ESCI validation results.")
    elif score_source.startswith("unavailable"):
        status=f"**XLM-R scoring failed ({score_source.split(':',1)[1]}).** Showing unranked catalog candidates only; no ESCI labels or ranking scores are fabricated."
    else:
        status=f"**XLM-R checkpoint unavailable:** `{model_status.checkpoint}`. Showing unranked catalog candidates only; no predictions or relevance scores are fabricated."
    status += f" Candidate pool: {len(candidates)} distinct Product IDs. Top-K only slices the same deterministic ranked set."
    exact_stock_matches=sum(bool(item["in_stock_facet_match"]) for item in candidates)
    if exact_stock_matches:
        status += f" Explicit in-stock catalog facet matches: {exact_stock_matches} product(s); model class remains a prediction."
    elif any(item["facet_match"] is not None for item in candidates):
        status += " No exact in-stock catalog match for the explicit request; the table still shows XLM-R relevance candidates, not exact substitutes."
    missing_images=sum(1 for item in shown if not item["image_path"] or not item["image_path"].is_file())
    if missing_images:
        status += f"\n\n{missing_images} result image(s) were unavailable; product matching and catalog facts remain unchanged."
    else:
        status += "\n\nImages are illustrative demo assets, not verified product photographs."
    return status,gallery,pd.DataFrame(rows),pd.DataFrame(technical_rows)

def outfit_language(outfit_id: str | None) -> bool:
    customer_id=STORE.outfit_record(outfit_id).get("customer_id") if outfit_id else None
    if not customer_id: return False
    _,state=STORE.conversation(customer_id)
    return state.get("demo_conversation",{}).get("language")=="ar"

def outfit_payload(outfit_id: str | None):
    record = STORE.outfit_record(outfit_id); selected=[]; arabic=outfit_language(outfit_id)
    if DEMO_CATALOG:
        for item in record["items"]:
            variant = DEMO_CATALOG.get(item["variant_id"])
            if variant: selected.append((item["slot_key"], variant, int(item.get("quantity", 1))))
    total = sum(v.price_piasters * qty for _, v, qty in selected)
    item_html=[]; remove=[]; rows=[]
    for slot, v, qty in selected:
        name=localize_product_name(v.name,arabic); color=localize_color(v.color,arabic); category=localize_category(v.category,arabic)
        qty_text=f"الكمية {format_integer(qty,True)} · للقطعة" if arabic else f"Qty {qty} · each"
        item_html.append(f"<div class='outfit-item'><span><b>{escape(name)}</b><br>{escape(category)} · {escape(color)} · {v.size} · {qty_text}<br>{money(v.price_piasters,arabic)}</span></div>")
        remove.append((f"{name} — {color}/{v.size}", slot)); rows.append({"Slot":slot,"Variant ID":v.variant_id,"Product":name,"Category":category,"Color":color,"Size":v.size,"Quantity":qty,"Unit price":format_egp(v.price_egp,arabic),"Line total":format_egp(v.price_egp*qty,arabic),"Demo stock":v.stock_qty})
    body="".join(item_html) or "<div class='empty'>No selected products yet. Search the demo catalog and add an exact variant.</div>"
    disclaimer="بيانات DEMO: الأسعار والمخزون تجريبية." if arabic else "DEMO catalog: prices and stock are simulated."
    total_label="الإجمالي" if arabic else "Total"
    heading="المنتجات المختارة" if arabic else "Selected products"
    html=f"<div class='cart-panel'><h3>{heading}</h3><small>{disclaimer}</small><div>{body}</div><div class='total'>{total_label}: {money(total,arabic)}</div></div>"
    return html, pd.DataFrame(rows), gr.Dropdown(choices=remove, value=None), money(total,arabic), selected

def switch_customer(customer_id):
    if not STORE.is_current_day_customer(customer_id): return [], {"customer_id":None}, "No active customer", gr.Dropdown(choices=[]), "", *outfit_payload(None)[:4], "No active customer in this business day."
    history,_=STORE.conversation(customer_id); outfit_id=first_outfit(customer_id); customer=STORE.customer(customer_id); html, table, remove, total, _=outfit_payload(outfit_id)
    return history,{"customer_id":customer_id},f"Active: {customer['display_name']} ({customer_id})",gr.Dropdown(choices=outfit_choices(customer_id),value=outfit_id),STORE.outfit_record(outfit_id)["display_name"],html,table,remove,total,"Customer session restored."

def new_customer(label):
    customer_id=STORE.create_customer(label); result=switch_customer(customer_id)
    return gr.Dropdown(choices=choices(),value=customer_id),"",*result

def end_customer(session):
    cid=(session or {}).get("customer_id")
    if cid and STORE.is_current_day_customer(cid): STORE.end_customer(cid,day())
    remaining=STORE.list_customers(); nxt=remaining[0]["customer_id"] if remaining else None
    result=switch_customer(nxt)
    return gr.Dropdown(choices=choices(),value=nxt),*result[:-1],"Customer completed and retained. Switched to an active customer."

def switch_customer_cart(customer_id):
    """Customer switcher for the cart-only right panel."""
    if not STORE.is_current_day_customer(customer_id):
        html, table, remove, total, _ = outfit_payload(None)
        return [], {"customer_id": None}, "No active customer", None, html, table, remove, total, "Completed customer cannot receive messages."
    STORE.set_active_customer(customer_id)
    history, _ = STORE.conversation(customer_id)
    cart_id = first_outfit(customer_id)
    html, table, remove, total, _ = outfit_payload(cart_id)
    customer = STORE.customer(customer_id)
    return render_history(history), {"customer_id": customer_id}, f"Active: {customer['display_name']} ({customer_id})", cart_id, html, table, remove, total, "Customer session restored."

def new_customer_cart(label, request_key):
    try:
        customer_id = STORE.create_customer_once(request_key, label)
    except RuntimeError as exc:
        return gr.Dropdown(choices=choices(),value=None), "", [], {"customer_id":None}, STORE.business_day_label(), None, *outfit_payload(None)[:4], str(exc), uuid4().hex
    result = switch_customer_cart(customer_id)
    return gr.Dropdown(choices=choices(), value=customer_id), "", *result, uuid4().hex

def restore_active_customer():
    """Restore the operator's selected customer after a page refresh."""
    customer_id = STORE.active_customer_id()
    if not customer_id:
        active_customers = STORE.list_customers()
        customer_id = active_customers[0]["customer_id"] if active_customers else None
        if customer_id:
            STORE.set_active_customer(customer_id)
    if not customer_id:
        return (gr.Dropdown(choices=choices(),value=None), [], {"customer_id":None}, "No customer in this business day.", None,
                *outfit_payload(None)[:4], "No customer in this business day. Start a new customer session.")
    return gr.Dropdown(choices=choices(), value=customer_id), *switch_customer_cart(customer_id)

def end_customer_cart(session):
    customer_id = (session or {}).get("customer_id")
    if customer_id and STORE.is_current_day_customer(customer_id):
        STORE.end_customer(customer_id, day())
    remaining = STORE.list_customers()
    next_id = remaining[0]["customer_id"] if remaining else None
    result = switch_customer_cart(next_id)
    message="Customer completed and retained. Switched to an active customer." if next_id else "Customer completed and retained. Start a new customer session when ready."
    return gr.Dropdown(choices=choices(), value=next_id), *result[:-1], message

def remove_customer_selected(session, selected_customer_id, active_label, confirmed):
    """Remove only the selected active customer after explicit ID confirmation."""
    # The dropdown is the operator's explicit target. Gradio may deliver its
    # change callback and the send/session state on adjacent event turns, so the
    # session State can briefly lag behind the selected dropdown value.
    cid = selected_customer_id
    fallback_id = (session or {}).get("customer_id")
    def result(customer_id, status):
        switched=list(switch_customer_cart(customer_id))
        switched[-1]=status
        return (gr.Dropdown(choices=choices(),value=customer_id), *switched,
                remove_confirmation_label(customer_id),
                gr.Dropdown(choices=removed_choices(),value=None))
    if not confirmed:
        return result(cid if STORE.is_current_day_customer(cid) else fallback_id,
                      "Removal cancelled. No customer records were changed.")
    if not STORE.active_business_day():
        return result(None,"The business day is closed. Start a new day before changing customer sessions.")
    if not cid or not STORE.is_current_day_customer(cid):
        return result(fallback_id if STORE.is_current_day_customer(fallback_id) else None,
                      "Removal stopped: the selected customer is no longer active. Refresh the list and try again.")
    customer=STORE.customer(cid)
    label=customer["display_name"] if customer else cid
    if not STORE.remove_customer(cid,day()):
        return result(fallback_id if STORE.is_current_day_customer(fallback_id) else None,
                      "Removal did not complete: the customer was already archived or is no longer active.")
    remaining=STORE.list_customers()
    next_id=remaining[0]["customer_id"] if remaining else None
    if next_id is None:
        # Keep the operator on a usable, independent session if the archived
        # customer was the last active one. Historical records remain attached
        # to the archived ID; this new session starts empty.
        try:
            next_id = STORE.create_customer()
        except RuntimeError:
            next_id = None
    switched=switch_customer_cart(next_id)
    switched=list(switched)
    switched[-1]=f"{label} ({cid}) was archived from the active list. Conversation, preferences, selected products, purchases and reports were retained."
    return (gr.Dropdown(choices=choices(),value=next_id), *switched,
            remove_confirmation_label(next_id),
            gr.Dropdown(choices=removed_choices(),value=None))

def restore_removed_customer(customer_id):
    if not customer_id or not STORE.restore_customer(customer_id):
        return gr.Dropdown(choices=choices(),value=STORE.active_customer_id()), gr.Dropdown(choices=removed_choices()), [], {"customer_id":None}, "", None, *outfit_payload(None)[:4], "Choose an archived customer to restore."
    STORE.set_active_customer(customer_id)
    return gr.Dropdown(choices=choices(),value=customer_id), gr.Dropdown(choices=removed_choices()), *switch_customer_cart(customer_id)

def remove_confirmation_label(customer_id):
    customer=STORE.customer(customer_id)
    if not customer: return gr.Markdown("Select an active customer to archive.")
    return gr.Markdown(f"**Archive confirmation target:** {customer['display_name']} (`{customer_id}`). This hides the customer from the active list; history and purchase/report records remain recoverable.")

def selected_item_quantity(outfit_id, slot):
    record=STORE.outfit_record(outfit_id)
    item=next((row for row in record["items"] if row["slot_key"]==slot),None)
    return gr.Number(value=int(item.get("quantity",1)) if item else 1)

def chat(message, session, outfit_id):
    cid=(session or {}).get("customer_id")
    html, table, remove, total, _ = outfit_payload(outfit_id)
    if not cid or not STORE.is_current_day_customer(cid): return [],"",session,html,table,remove,total,"Select a customer in the active business day."
    if not STORE.active_business_day(): return STORE.conversation(cid)[0],"",session,html,table,remove,total,"Business day is closed. Start a new day to continue."
    if not message or not message.strip(): return STORE.conversation(cid)[0],"",session,html,table,remove,total,""
    # SQLite is the source of truth for active chat history. The browser's
    # Chatbot value may be stale (for example just after Clear Chat), so never
    # rehydrate persistence from the client-provided history.
    stored,state=STORE.conversation(cid); active_history=stored
    if not DEMO_CONVERSATION:
        return render_history(active_history), "", session, html, table, remove, total, f"Demo catalog cannot load: {DEMO_ERROR}"
    prior_state=state
    reply, updated, action = DEMO_CONVERSATION.respond(message, state)
    reply_arabic = updated.get("demo_conversation", {}).get("language") == "ar"
    # The planner may propose an exact variant only after it has resolved an
    # identity.  Never add a product by name, or silently pick a different SKU.
    variant_to_add = action.get("add_variant_id")
    if variant_to_add and DEMO_CATALOG and DEMO_CATALOG.get(variant_to_add) and DEMO_CATALOG.get(variant_to_add).in_stock:
        variant = DEMO_CATALOG.get(variant_to_add)
        try:
            STORE.add_or_replace_outfit_item(outfit_id, variant.variant_id, variant.category, variant.product_id)
            persisted=any(item.get("variant_id")==variant.variant_id for item in STORE.outfit_record(outfit_id).get("items",[]))
        except Exception:
            persisted=False
        if persisted:
            html, table, remove, total, _ = outfit_payload(outfit_id)
            reply = (f"تمت إضافة {DEMO_CONVERSATION.product_name(variant,True)} — {DEMO_CONVERSATION.color_name(variant.color,True)}، مقاس {variant.size} — للسلة." if reply_arabic
                     else f"Added {variant.name} — {variant.color}/{variant.size} to Selected Products.")
        else:
            reply = "ماقدرتش أتأكد إن المنتج اتضاف للسلة، فمش هأكد الإضافة. جرّب تاني." if reply_arabic else "I could not verify that the item was added, so I cannot confirm it. Please try again."
            action["add_variant_id"] = None
    # Persist exact SKU-to-image associations in conversation history while
    # keeping repetitive image provenance text out of the customer-facing UI.
    presentation = action.get("presentation")
    card_record = None
    if isinstance(presentation, dict):
        allowed_fields = {"name", "image", "price", "sizes", "color", "colors", "category"}
        exact_ids = [variant_id for variant_id in presentation.get("variant_ids", [])
                     if DEMO_CATALOG and DEMO_CATALOG.get(str(variant_id))]
        if exact_ids:
            card_record = {"variant_ids": exact_ids,
                           "fields": [field for field in presentation.get("fields", []) if field in allowed_fields],
                           "language": presentation.get("language") if presentation.get("language") in {"ar", "en"} else "en"}
    image_records=[]
    image_variant_ids = list(action.get("image_variant_ids", []))
    if card_record and "image" in card_record.get("fields", []):
        image_variant_ids.extend(card_record.get("variant_ids", []))
    for variant_id in dict.fromkeys(image_variant_ids):
        variant=DEMO_CATALOG.get(variant_id) if DEMO_CATALOG else None
        if variant and variant.image_path and variant.image_path.is_file():
            image_records.append({"path":str(variant.image_path),"product_id":variant.product_id,
                "variant_id":variant.variant_id,"display_name":DEMO_CONVERSATION.product_name(variant,reply_arabic),
                "display_color":DEMO_CONVERSATION.color_name(variant.color,reply_arabic),
                "image_provenance":variant.image_provenance,"illustrative":True})
        elif variant:
            image_records.append({"path":"","product_id":variant.product_id,"variant_id":variant.variant_id,
                "display_name":DEMO_CONVERSATION.product_name(variant,reply_arabic),
                "display_color":DEMO_CONVERSATION.color_name(variant.color,reply_arabic),"illustrative":True})
    assistant_turn={"role":"assistant","content":reply,"_images":image_records}
    if card_record:
        assistant_turn["_cards"] = card_record
    updated_history=active_history+[{"role":"user","content":message},assistant_turn]
    STORE.save_conversation(cid,updated_history,updated); STORE.record_activity(cid,day(),"customer_message",{})
    _record_marketing_signals(cid,day(),message,prior_state,updated,action,reply)
    displayed = updated.get("demo_conversation", {}).get("displayed_variant_ids", [])
    if displayed:
        exact=[DEMO_CATALOG.get(variant_id) for variant_id in displayed] if DEMO_CATALOG else []
        exact=[variant for variant in exact if variant]
        STORE.record_activity(cid,day(),"ranked_catalog_inquiry",{
            "product_ids": list(dict.fromkeys(v.product_id for v in exact)),
            "variant_ids": list(dict.fromkeys(v.variant_id for v in exact)),
            "categories": list(dict.fromkeys(v.category for v in exact)),
            "colors": list(dict.fromkeys(v.color for v in exact)),
            "catalog": "workbook_demo"})
    return render_history(updated_history),"",{"customer_id":cid},html,table,remove,total,""

def clear_chat(session):
    customer_id=(session or {}).get("customer_id")
    if not customer_id or not STORE.is_current_day_customer(customer_id):
        return [], session or {}, "Select an active customer before clearing chat."
    archived, message_count = STORE.clear_conversation(customer_id)
    if archived:
        status=f"Chat cleared. Archived {message_count} messages; customer preferences and selected products were kept."
    else:
        status="Chat is already clear. Customer preferences and selected products were kept."
    return [], {"customer_id":customer_id}, status


def copy_safe_transcript(session):
    """Return a plain-text copy path independent of Gradio rich-card serialization."""
    customer_id = (session or {}).get("customer_id")
    if not customer_id:
        return gr.update(value="Select a customer to load the transcript.", visible=True)
    history, _state = STORE.conversation(customer_id)
    if not history:
        return gr.update(value="No messages in this conversation yet.", visible=True)

    lines = []
    for saved in history:
        role = saved.get("role", "assistant")
        content = str(saved.get("content") or "").strip()
        card_record = saved.get("_cards")
        # Older saved rich turns can lack readable text or contain the literal
        # frontend serialization artifact. Rebuild from exact saved variant IDs.
        if role == "assistant" and card_record and (not content or "undefined" in content.casefold()):
            is_arabic_reply = card_record.get("language") == "ar"
            fields = set(card_record.get("fields") or [])
            rebuilt = []
            for index, variant_id in enumerate(card_record.get("variant_ids", []), 1):
                variant = DEMO_CATALOG.get(str(variant_id)) if DEMO_CATALOG else None
                if not variant:
                    continue
                name = DEMO_CONVERSATION.product_name(variant, is_arabic_reply)
                facts = [name]
                if "price" in fields:
                    facts.append(format_egp(variant.price_egp, is_arabic_reply))
                if "color" in fields:
                    facts.append(DEMO_CONVERSATION.color_name(variant.color, is_arabic_reply))
                if "colors" in fields:
                    colors = sorted({v.color for v in DEMO_CATALOG.variants
                                     if v.product_id == variant.product_id and v.in_stock})
                    facts.append(("الألوان: " if is_arabic_reply else "Colors: ") + ", ".join(
                        DEMO_CONVERSATION.color_name(color, is_arabic_reply) for color in colors))
                if "sizes" in fields:
                    sizes = sorted({v.size for v in DEMO_CATALOG.variants
                                    if v.product_id == variant.product_id and v.color == variant.color and v.in_stock},
                                   key=lambda value: (len(value), value))
                    facts.append(("المقاسات: " if is_arabic_reply else "Sizes: ") + ", ".join(sizes))
                rebuilt.append(f"{index}. " + " — ".join(facts))
            if rebuilt:
                content = "\n".join(rebuilt)
        if not content:
            continue
        label = "العميل" if role == "user" and is_arabic(content) else "Customer" if role == "user" else "KAYAN"
        lines.append(f"{label}:\n{content}")
    return gr.update(value="\n\n".join(lines) or "No readable messages in this conversation.", visible=True)

def add_variant(outfit_id, variant_id, quantity=1):
    if not DEMO_CATALOG or not outfit_id: html,table,remove,total,_=outfit_payload(outfit_id); return html,table,remove,total,"Select a valid demo variant."
    owner=STORE.outfit_record(outfit_id).get("customer_id")
    if not STORE.active_business_day() or not STORE.is_current_day_customer(owner): html,table,remove,total,_=outfit_payload(outfit_id); return html,table,remove,total,"Start a new customer session in the active business day before changing selected products."
    variant=DEMO_CATALOG.get(variant_id)
    quantity=max(1,min(99,int(quantity or 1)))
    if not variant or not variant.in_stock or quantity>variant.stock_qty: html,table,remove,total,_=outfit_payload(outfit_id); return html,table,remove,total,"Requested quantity exceeds the exact variant's simulated stock."
    STORE.add_or_replace_outfit_item(outfit_id,variant.variant_id,variant.category,variant.product_id,quantity)
    html,table,remove,total,_=outfit_payload(outfit_id); return html,table,remove,total,f"Added exact demo variant {variant.variant_id} × {quantity}."

def update_item_quantity(outfit_id, slot, quantity):
    if not outfit_id or not slot: html,table,remove,total,_=outfit_payload(outfit_id); return html,table,remove,total,"Choose a selected product first."
    record=STORE.outfit_record(outfit_id)
    if not STORE.active_business_day() or not STORE.is_current_day_customer(record.get("customer_id")): html,table,remove,total,_=outfit_payload(outfit_id); return html,table,remove,total,"Selected products are read-only outside their active business day."
    item=next((row for row in record["items"] if row["slot_key"]==slot),None)
    variant=DEMO_CATALOG.get(item["variant_id"]) if item and DEMO_CATALOG else None
    quantity=max(1,min(99,int(quantity or 1)))
    if not variant or not variant.in_stock or quantity>variant.stock_qty: html,table,remove,total,_=outfit_payload(outfit_id); return html,table,remove,total,"Quantity is above the exact variant's simulated stock."
    STORE.set_item_quantity(outfit_id,slot,quantity)
    html,table,remove,total,_=outfit_payload(outfit_id); return html,table,remove,total,"Selected quantity updated from the demo stock field."

def remove_item(outfit_id, slot):
    record=STORE.outfit_record(outfit_id) if outfit_id else {}
    if outfit_id and (not STORE.active_business_day() or not STORE.is_current_day_customer(record.get("customer_id"))):
        html,table,remove,total,_=outfit_payload(outfit_id); return html,table,remove,total,"Historical selected products are read-only."
    if outfit_id and slot: STORE.remove_outfit_item(outfit_id,slot)
    html,table,remove,total,_=outfit_payload(outfit_id); return html,table,remove,total,"Item removed." if slot else "Choose an item to remove."

def prepare_buy(outfit_id, session):
    html,table,remove,total,items=outfit_payload(outfit_id)
    arabic=outfit_language(outfit_id)
    cid=(session or {}).get("customer_id")
    if not cid or not STORE.is_current_day_customer(cid) or not outfit_id or not STORE.active_business_day():
        return "<div class='error'>Select an active customer and shopping cart first.</div>",None
    if not items: return "<div class='error'>Add in-stock demo variants before preparing a mock purchase.</div>",None
    if any(not v.in_stock or qty > v.stock_qty for _,v,qty in items): return "<div class='error'>One or more selected demo variants do not have enough simulated stock.</div>",None
    token=uuid4().hex
    if arabic:
        summary=f"<div class='mock'><b>مراجعة طلب تجريبي MOCK</b><br>{format_integer(sum(qty for _,_,qty in items),True)} قطعة عبر {format_integer(len(items),True)} اختيار. الإجمالي: {total}. مفيش دفع حقيقي أو بيع فعلي أو خصم مخزون.</div>"
    else:
        summary=f"<div class='mock'><b>MOCK purchase review</b><br>{sum(qty for _,_,qty in items)} units across {len(items)} selected variants. Total {total}. No payment, real sale, or stock deduction will occur.</div>"
    return summary,token

def confirm_buy(session,outfit_id,confirmed,confirmation_key):
    cid=(session or {}).get("customer_id")
    saved_history,_=STORE.conversation(cid) if cid else ([],{})
    if not confirmed: return "<div class='error'>Explicit mock-purchase confirmation is required.</div>",False,confirmation_key,render_history(saved_history)
    if not cid or not STORE.is_current_day_customer(cid) or not STORE.active_business_day() or not confirmation_key:
        return "<div class='error'>No active customer or pending confirmation. Review the cart and click Buy Now (MOCK) again.</div>",False,None,[]
    existing=STORE.mock_purchase_for_confirmation(confirmation_key)
    history,state=STORE.conversation(cid)
    arabic=state.get("demo_conversation",{}).get("language")=="ar"
    if existing:
        return f"<div class='mock'>Mock purchase request {escape(existing['request_id'])} was already recorded; no duplicate was created.</div>",False,None,render_history(history)
    _,_,_,_,items=outfit_payload(outfit_id)
    if not items or any(not v.in_stock or qty > v.stock_qty for _,v,qty in items):
        return "<div class='error'>Request was not recorded because one or more exact selected variants are unavailable or exceed demo stock.</div>",False,None,render_history(history)
    persisted_items=[{"variant_id":v.variant_id,"product_id":v.product_id,"category":v.category,
                      "color":v.color,"size":v.size,"currency":v.currency,
                      "quantity":qty,"unit_price_piasters":v.price_piasters,
                      "line_total_piasters":v.price_piasters*qty} for _,v,qty in items]
    total_piasters=sum(v.price_piasters*qty for _,v,qty in items)
    try:
        request=STORE.record_mock_purchase(cid,outfit_id,total_piasters,persisted_items,confirmation_key,day())
        record=STORE.mock_purchase(request)
        if not record or record["status"]!="confirmed_mock_purchase":
            raise RuntimeError("persisted mock purchase record could not be verified")
    except Exception as exc:
        return f"<div class='error'>Purchase request was not confirmed: {escape(type(exc).__name__)}. Please retry after reviewing the cart.</div>",False,None,render_history(history)
    lines=[]
    for item in record["items"]:
        variant=DEMO_CATALOG.get(item["variant_id"]) if DEMO_CATALOG else None
        if not variant or variant.product_id != item.get("product_id"):
            return "<div class='error'>The saved request could not be matched to the catalog; no success message was shown.</div>",False,None,render_history(history)
        unit=int(item["unit_price_piasters"])
        line_total=unit*int(item["quantity"])
        name=localize_product_name(variant.name,arabic); color=localize_color(variant.color,arabic)
        quantity=format_integer(int(item["quantity"]),arabic)
        lines.append(f"- {name} — {color}/{variant.size} × {quantity}: {money(line_total,arabic)}")
    if arabic:
        message_text=("تم تسجيل طلب الشراء التجريبي بنجاح ✅\n"+"\n".join(lines)+f"\nالإجمالي: {money(int(record['total_piasters']),True)}\nدي عملية DEMO، ومفيش دفع حقيقي أو خصم مخزون حصل.")
    else:
        message_text=("Your MOCK purchase request was recorded successfully ✅\n"+"\n".join(lines)+f"\nTotal: {money(int(record['total_piasters']),False)}\nThis is a DEMO request; no real payment or stock deduction occurred.")
    updated=history+[{"role":"assistant","content":message_text}]
    STORE.save_conversation(cid,updated,state)
    return f"<div class='mock'>MOCK purchase request {escape(request)} was saved and verified. No payment or stock deduction occurred.</div>",False,None,render_history(updated)

def day_history_table():
    rows=STORE.list_business_days()
    return pd.DataFrame([{"Day":f"Day {r['day_number']}","Business date":r["business_date"],"Status":r["status"],
                          "Started":r["started_at"],"Ended":r["ended_at"] or "—","Day ID":r["day_id"]} for r in rows])

def day_history_choices():
    return [(f"Day {r['day_number']} — {r['business_date']} ({r['status']})",r["day_id"]) for r in STORE.list_business_days()]

def view_historical_day(day_id):
    if not day_id:return {"message":"Choose a business day to view its saved report."}
    return STORE.business_day_snapshot(day_id) or {"message":"No saved report exists for this day."}


def communication_day_choices():
    return [(f"Day {row['day_number']} — {row['business_date']} ({row['status']})",row["day_id"]) for row in STORE.list_business_days()]


def _task_frame(rows):
    return pd.DataFrame([{"Task ID":r["task_id"],"Task":r["title"],"Assigned date":r["assigned_date"],
        "Assigned by":r["assigned_by"],"Assigned to":r["assigned_to"],
        "Status":"Done ✓" if r["is_done"] else "Not Done ○","Completed at":r["completed_at"] or "—"} for r in rows],
        columns=["Task ID","Task","Assigned date","Assigned by","Assigned to","Status","Completed at"])


def refresh_ceo_tasks(assigned_date):
    try:date_value=datetime.fromisoformat(str(assigned_date)).date().isoformat()
    except Exception:date_value=business_date()
    rows=STORE.list_ceo_tasks(date_value)
    done=sum(bool(row["is_done"]) for row in rows)
    status=f"**{done} / {len(rows)} Done** · {len(rows)-done} Not Done · assigned for {date_value}. End Day does not auto-complete tasks."
    return _task_frame(rows),gr.Dropdown(choices=[(f"{r['title']} — {r['task_id'][-8:]}",r["task_id"]) for r in rows],value=None),status


def create_ceo_task_ui(title,assigned_date):
    try:
        task_id=STORE.create_ceo_task(title,assigned_date or business_date())
        return (*refresh_ceo_tasks(assigned_date),f"Task {task_id} assigned to Sales. Local demo event saved.")
    except Exception as exc:
        return (*refresh_ceo_tasks(assigned_date),f"Task not saved: {type(exc).__name__}: {escape(str(exc))}")


def load_ceo_task_ui(task_id):
    if not task_id:return "","", "Choose a task to edit."
    row=next((r for r in STORE.list_ceo_tasks() if r["task_id"]==task_id),None)
    if not row:return "","", "That task no longer exists."
    return row["title"],row["assigned_date"],f"Editing {row['task_id']} · assigned by CEO to Sales."


def update_ceo_task_ui(task_id,title,assigned_date):
    try:
        if not STORE.update_ceo_task(task_id,title,assigned_date):raise ValueError("Select an existing task")
        return (*refresh_ceo_tasks(assigned_date),"Task description/date updated. A versioned CEO task_updated event was saved.")
    except Exception as exc:
        return (*refresh_ceo_tasks(assigned_date),f"Task not updated: {type(exc).__name__}: {escape(str(exc))}")


def sales_task_status_ui(task_id,is_done,assigned_date):
    if not task_id:return (*refresh_ceo_tasks(assigned_date),"Choose a task first.")
    try:
        if not STORE.set_task_done_by_sales(task_id,bool(is_done),source_agent="sales"):
            raise ValueError("Task does not exist or is not assigned to Sales")
        return (*refresh_ceo_tasks(assigned_date),"Sales completion status saved. Mark Done only after the assigned work is verified.")
    except Exception as exc:
        return (*refresh_ceo_tasks(assigned_date),f"Status not saved: {type(exc).__name__}: {escape(str(exc))}")


def ceo_saved_day_report(day_id):
    report=STORE.communication_report(day_id,"ceo") if day_id else None
    if not report:return "No saved CEO report for this day.",_task_frame([])
    data=report["report"]
    return (f"**Day {data.get('business_day_id')} · {data.get('business_date')}** — "
            f"{data.get('completed_tasks',0)} / {data.get('total_tasks',0)} Done; "
            f"{data.get('not_done_tasks',0)} Not Done."),_task_frame(data.get("tasks",[]))


def _marketing_rows(rows):
    return pd.DataFrame([{"Signal":r.get("event_type",""),"Customer":r.get("customer_label",""),
        "Customer reference":r.get("customer_id",""),"Product ID":r.get("product_id",""),
        "Variant ID":r.get("variant_id",""),"Feedback":r.get("feedback_type",""),
        "Reason":r.get("reason") or "—","Category":r.get("category",""),"Color":r.get("color",""),
        "Size":r.get("size",""),"Unmet criteria":json.dumps(r.get("criteria",{}),ensure_ascii=False),
        "Timestamp":r.get("timestamp","")} for r in rows],columns=["Signal","Customer","Customer reference","Product ID","Variant ID","Feedback","Reason","Category","Color","Size","Unmet criteria","Timestamp"])


def marketing_refresh(from_date,to_date):
    try:
        data=STORE.marketing_contract_data(from_date or None,to_date or None)
        summary=(f"**Customers in conversations:** {data['conversation_customer_count']} · **Product interest:** {data['interest_count']} · "
                 f"**Explicit positive:** {data['explicit_positive_count']} · **Accepted recommendation:** {data['accepted_recommendation_count']} · "
                 f"**Explicit rejection:** {data['rejection_count']} · **Unmet requests:** {data['unmet_demand_count']} · "
                 f"**Confirmed MOCK requests:** {data['confirmed_mock_purchase_count']}\n\nLocal simulated Marketing adapter. Interest is not approval; a cart is not a purchase.")
        product_rows=[{"Product ID":pid,"Distinct customer interest signals":count}
                      for pid,count in sorted(data["product_interest_counts"].items(),key=lambda row:(-row[1],row[0]))]
        products=pd.DataFrame(product_rows,columns=["Product ID","Distinct customer interest signals"])
        return summary,products,_marketing_rows(data["customer_details"])
    except Exception as exc:return f"Marketing data could not be loaded: {type(exc).__name__}",pd.DataFrame(),pd.DataFrame()


def marketing_saved_report(day_id):
    report=STORE.communication_report(day_id,"marketing") if day_id else None
    if not report:return "No saved Marketing report for this day.",pd.DataFrame(),pd.DataFrame()
    data=report["report"]
    summary=(f"**{data.get('business_date')}** · interest {sum(data.get('product_interest_counts',{}).values())} · "
             f"positive {sum(data.get('explicit_positive_feedback_counts',{}).values())} · rejected {sum(data.get('explicit_rejection_counts',{}).values())} · "
             f"unmet {data.get('unmet_demand_count',0)} · confirmed MOCK {len(data.get('confirmed_mock_purchases',[]))}")
    products=pd.DataFrame([{"Product ID":k,"Interest":v,"Positive":data.get("explicit_positive_feedback_counts",{}).get(k,0),
                            "Rejected":data.get("explicit_rejection_counts",{}).get(k,0),"Rejection reasons":json.dumps(data.get("rejection_reason_counts",{}).get(k,{}),ensure_ascii=False)}
                           for k,v in data.get("product_interest_counts",{}).items()],columns=["Product ID","Interest","Positive","Rejected","Rejection reasons"])
    return summary,products,_marketing_rows(data.get("customer_specific_feedback",[]))


def inventory_refresh(from_date,to_date):
    data=STORE.inventory_contract_data(from_date or None,to_date or None)
    summary=(f"**Confirmed MOCK orders:** {data['confirmed_mock_orders']} · **MOCK revenue:** {format_piasters(data['mock_revenue_piasters'], False)} · "
             "**Cost / profit:** unavailable (no verified cost data).\n\n"
             "Inventory adapter is local demo. The workbook stock is simulated and no stock is deducted.")
    orders=pd.DataFrame([{"Order ID":r["request_id"],"Event ID":f"event-purchase-{r['request_id']}",
        "Customer":r.get("display_name") or r["customer_id"],"Customer reference":r["customer_id"],
        "Business day":r.get("business_date") or "—","Status":"Confirmed MOCK","Total (EGP)":format_piasters(r["total_piasters"], False),
        "Items":len(r["items"]),"Timestamp":r["created_at"]} for r in data["orders"]],
        columns=["Order ID","Event ID","Customer","Customer reference","Business day","Status","Total (EGP)","Items","Timestamp"])
    sales=pd.DataFrame([{"Product ID":r.get("product_id"),"Variant ID":r["variant_id"],"Color":r.get("color"),"Size":r.get("size"),
        "Quantity sold":r["quantity_sold"],"Unit price (minor units)":r["unit_price_piasters"],
        "MOCK revenue (minor units)":r["revenue_piasters"]} for r in data["variant_sales"]],
        columns=["Product ID","Variant ID","Color","Size","Quantity sold","Unit price (minor units)","MOCK revenue (minor units)"])
    stock=pd.DataFrame([{"Product ID":v.product_id,"Variant ID":v.variant_id,"Color":v.color,"Size":v.size,
        "Simulated demo stock":v.stock_qty,"Price (EGP)":v.price_egp,"Source status":v.stock_status}
        for v in DEMO_CATALOG.variants] if DEMO_CATALOG else [],
        columns=["Product ID","Variant ID","Color","Size","Simulated demo stock","Price (EGP)","Source status"])
    return summary,orders,sales,stock


def inventory_saved_report(day_id):
    report=STORE.communication_report(day_id,"inventory") if day_id else None
    if not report:return "No saved Inventory report for this day.",pd.DataFrame(),pd.DataFrame()
    data=report["report"]
    summary=(f"**{data.get('business_date')}** · confirmed MOCK orders {len(data.get('confirmed_mock_orders',[]))} · "
             f"MOCK revenue {format_piasters(data.get('mock_revenue_piasters',0), False)}. Cost, gross profit and net profit are unavailable. "
             "No stock deduction occurred.")
    orders=pd.DataFrame([{"Order ID":r.get("order_id"),"Event ID":r.get("event_id"),"Customer":r.get("customer_label"),
        "Customer reference":r.get("customer_id"),"Status":"Confirmed MOCK","Total (EGP)":format_piasters(r.get("total_piasters",0), False),
        "Timestamp":r.get("timestamp")} for r in data.get("confirmed_mock_orders",[])],columns=["Order ID","Event ID","Customer","Customer reference","Status","Total (EGP)","Timestamp"])
    sales=pd.DataFrame([{"Product ID":r.get("product_id"),"Variant ID":r.get("variant_id"),"Color":r.get("color"),"Size":r.get("size"),
        "Quantity sold":r.get("quantity_sold"),"Unit price (minor units)":r.get("unit_price_piasters"),"MOCK revenue (minor units)":r.get("revenue_piasters")}
        for r in data.get("variant_sales",[])],columns=["Product ID","Variant ID","Color","Size","Quantity sold","Unit price (minor units)","MOCK revenue (minor units)"])
    return summary,orders,sales

def _day_control_refresh(snapshot, status, day_confirm_value=False, start_confirm_value=False):
    cid=None
    current=STORE.active_customer_id()
    if current: cid=current
    if cid:
        history,_=STORE.conversation(cid); outfit_id=first_outfit(cid); customer=STORE.customer(cid)
        active_label=f"Active: {customer['display_name']} ({cid})"
        cart_html,table,remove,total,_=outfit_payload(outfit_id)
    else:
        history=[]; outfit_id=None; active_label="No active customer"; cart_html,table,remove,total,_=outfit_payload(None)
    return (snapshot,day_confirm_value,start_confirm_value,False,None,"",status,STORE.business_day_label(),gr.Dropdown(choices=choices(),value=cid),
            render_history(history),{"customer_id":cid},active_label,outfit_id,cart_html,table,remove,total,
            day_history_table(),gr.Dropdown(choices=day_history_choices(),value=None),
            gr.Dropdown(choices=communication_day_choices(),value=None),
            gr.Dropdown(choices=communication_day_choices(),value=None),
            gr.Dropdown(choices=communication_day_choices(),value=None))

def end_day(confirmed):
    if not confirmed:
        return _day_control_refresh(view_historical_day(None),"Check the confirmation box before ending the day.",False)
    if not STORE.active_business_day():
        snapshot=STORE.business_day_snapshot((STORE.current_business_day() or {}).get("day_id","")) or {"status":"closed"}
        return _day_control_refresh(snapshot,"This business day is already closed. Start a new day to continue.",False)
    snapshot,created=STORE.end_active_day()
    path=DAY_REPORT_DIR/f"KAYAN_Day_{snapshot.get('day_number','unknown')}_{snapshot.get('business_day_id','unknown')}_v{snapshot.get('schema_version',1)}.json"
    if created:path.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding="utf-8")
    status=(f"Day {snapshot.get('day_number')} closed and saved." if created else "This day was already closed; its original snapshot was returned.")
    return _day_control_refresh(snapshot,status,False)

def start_new_day(confirmed):
    if not confirmed:
        current=STORE.current_business_day()
        snapshot=STORE.business_day_snapshot(current["day_id"]) if current and current["status"]=="closed" else _active_day_preview(current)
        return _day_control_refresh(snapshot or {},"Check the confirmation box before starting a new day.",False)
    active_day=STORE.active_business_day()
    if active_day:
        snapshot=_active_day_preview(active_day)
        return _day_control_refresh(snapshot,"The current business day is still active. End it before starting the next day.",False)
    day_record,created=STORE.start_new_day(business_date())
    snapshot=_active_day_preview(day_record)
    status=(f"Day {day_record['day_number']} started. Create a new customer session to begin." if created else
            f"Day {day_record['day_number']} is already active; no duplicate day was created.")
    return _day_control_refresh(snapshot,status,False)

def historical_day_change(day_id):
    return view_historical_day(day_id)

def _active_day_preview(record):
    if not record:return {"status":"no_business_day"}
    return {"business_day_id":record["day_id"],"day_number":record["day_number"],
            "business_date":record["business_date"],"timezone":record["timezone"],
            "started_at":record["started_at"],"status":record["status"],
            "customer_sessions":{"customers_served":0},"demo_mock_purchases":{"confirmed_requests":0,"simulated_value_piasters":0},
            "sales":{"confirmed_orders":0,"revenue_piasters":0},"reports":{}}

def catalog_image_gallery():
    """One existing image per distinct product-color variant."""
    if not DEMO_CATALOG:
        return []
    result=[]; seen=set()
    for variant in DEMO_CATALOG.variants:
        path=variant.image_path
        if not path or not path.is_file() or str(path) in seen:
            continue
        seen.add(str(path))
        result.append((str(path),f"{variant.name} · {variant.color} · illustrative DEMO image"))
    return result

CSS=""".app{background:#091116}.cards{display:flex;gap:10px;overflow:auto}.product-card{min-width:220px;display:flex;gap:8px;border:1px solid #33414d;padding:8px;border-radius:8px}.product-card img{width:86px;height:110px;object-fit:contain}.product-card small{color:#f0a13d}.cart-panel{border:1px solid #39424a;padding:10px;border-radius:10px}.cart-panel h3{margin:0}.outfit-item{display:flex;gap:8px;margin:7px 0}.outfit-item img,.tiny-missing{width:44px;height:44px;object-fit:cover}.total{font-weight:bold;border-top:1px solid #555;padding-top:8px}.mock{padding:10px;border:1px solid #b98533}.error{padding:10px;border:1px solid #bd5454}.empty{color:#aaa;padding:10px}@media(max-width:800px){.cards{flex-direction:column}.product-card{min-width:0}}"""

def build_ui():
  with gr.Blocks(title="KAYAN Sales AI Agent",css=CSS) as demo:
    gr.Markdown("# KAYAN Sales AI Agent\nWorkbook catalog is DEMO data: prices and stock are simulated. Images are illustrative concepts, not verified SKU photographs.")
    session=gr.State({"customer_id":DEFAULT_CUSTOMER_ID})
    cart_id=gr.State(first_outfit(DEFAULT_CUSTOMER_ID) if DEFAULT_CUSTOMER_ID else None)
    # Generate a per-browser-session idempotency key. A module-time string is
    # shared by all sessions and can cause two operators' first clicks to
    # resolve to the same customer-creation request.
    new_customer_key=gr.State(value=lambda: uuid4().hex)
    purchase_token=gr.State(None)
    with gr.Tab("Product Search"):
      gr.Markdown("### Product Search — workbook-backed DEMO variants")
      product_query=gr.Textbox(label="What are you looking for?",placeholder="black cotton t-shirt size M / عايز تيشيرت أسود مقاس M")
      product_stock=gr.Checkbox(value=True,label="Show variants with positive demo stock only")
      product_top_k=gr.Radio(choices=[5,10],value=5,label="Show ranked products",interactive=True)
      product_search_button=gr.Button("Find products",variant="primary")
      product_search_status=gr.Markdown()
      product_search_gallery=gr.Gallery(label="Matching product images · illustrative DEMO assets",columns=4,height=300,object_fit="contain",show_label=True)
      product_search_table=gr.Dataframe(label="Ranked results · XLM-R ESCI output",interactive=False,max_height=460)
      with gr.Accordion("Classifier probabilities and logits",open=False):
        product_search_technical=gr.Dataframe(label="Actual XLM-R outputs",interactive=False,max_height=320)
    with gr.Tab("Sales Conversation"):
      day_label=gr.Markdown(f"**{STORE.business_day_label()}**")
      with gr.Row():
        with gr.Column(scale=2):
          gr.Markdown("### Customer Sessions")
          customer_picker=gr.Dropdown(choices=choices(),value=DEFAULT_CUSTOMER_ID,label="Switch customer")
          initial_customer=STORE.customer(DEFAULT_CUSTOMER_ID)
          active_label=gr.Markdown(value=(f"**Active:** {initial_customer['display_name']} (`{DEFAULT_CUSTOMER_ID}`)" if initial_customer else "**No active customer**"))
          new_name=gr.Textbox(label="New customer label",placeholder="Customer 2")
          with gr.Row():
            new_button=gr.Button("+ New Customer"); end_button=gr.Button("End Customer")
          remove_target=gr.Markdown("Select an active customer to archive.")
          remove_confirmation=gr.Checkbox(label="Archive confirmation",value=False,visible=False)
          customer_remove_button=gr.Button("Remove Customer",variant="stop")
          with gr.Accordion("Archived customers",open=False):
            removed_customer_picker=gr.Dropdown(choices=removed_choices(),label="Recover a removed customer")
            restore_customer_button=gr.Button("Restore Customer")
        with gr.Column(scale=5):
          chatbot=gr.Chatbot(label="Sales conversation",height=340); message=gr.Textbox(label="Message")
          with gr.Row():
            send=gr.Button("Send",variant="primary")
            clear_chat_button=gr.Button("Clear Chat")
          status=gr.Markdown()
          with gr.Accordion("Copy-safe text transcript", open=False):
            gr.Markdown("Readable conversation text for copying; rich product cards remain visible in the chat above.")
            transcript_refresh=gr.Button("Refresh transcript")
            transcript_box=gr.Textbox(label="Conversation text", lines=10, interactive=False, visible=False)
        with gr.Column(scale=3):
          gr.Markdown("### Selected Products — DEMO")
          gr.Markdown("Exact product variants selected by this customer. Prices and stock are simulated demo data.")
          cart_html=gr.HTML(); item_table=gr.Dataframe(label="Selected products",interactive=False,max_height=240); total=gr.Textbox(label="Total",interactive=False)
          remove_picker=gr.Dropdown(label="Selected product"); cart_quantity=gr.Number(value=1,minimum=1,maximum=99,step=1,label="Quantity")
          with gr.Row():
            quantity_button=gr.Button("Update quantity"); cart_remove_button=gr.Button("Remove selected product")
          buy_status=gr.HTML(); buy_button=gr.Button("Buy Now (MOCK)",variant="primary"); confirm=gr.Checkbox(label="I explicitly confirm a mock purchase request"); confirm_button=gr.Button("Confirm Buy Now (MOCK)")
      gr.Markdown("### Business day")
      gr.Markdown("Daily reports and history are saved privately in SQLite.")
      with gr.Row():
        day_button=gr.Button("End Day",variant="stop")
        start_day_button=gr.Button("Start New Day",variant="primary")
      day_status=gr.Markdown()
      # Keep report and historical views available to callbacks/API without
      # adding verbose administrative JSON to the customer-facing interface.
      day_confirm=gr.Checkbox(value=False,visible=False)
      start_day_confirm=gr.Checkbox(value=False,visible=False)
      day_snapshot=gr.JSON(visible=False)
      history_table=gr.Dataframe(value=day_history_table(),visible=False)
      history_picker=gr.Dropdown(choices=day_history_choices(),visible=False)
      history_snapshot=gr.JSON(visible=False)
    with gr.Tab("Catalog"):
      gr.Markdown("### Workbook-backed DEMO catalog\nPrices and stock quantities are simulated. Images are illustrative concept assets, not verified SKU photographs.")
      gr.DownloadButton("Download demo workbook",value=str(DEMO_WORKBOOK))
      if DEMO_CATALOG:
        gr.Gallery(value=catalog_image_gallery(),label="Catalog product images · illustrative DEMO assets",columns=4,height=520,object_fit="contain",show_label=True)
        gr.Dataframe(pd.DataFrame([{ "Product ID":v.product_id,"Variant ID":v.variant_id,"Name":v.name,"Category":v.category,"Color":v.color,"Size":v.size,"Price (EGP)":v.price_egp,"Demo stock qty":v.stock_qty,"Stock status":v.stock_status,"Material":v.material,"Fit":v.fit,"Pattern":v.pattern,"Occasion":v.occasion,"Record status":v.record_status} for v in DEMO_CATALOG.variants]),interactive=False,max_height=600)
      else: gr.Markdown("**Demo catalog validation failed:** "+DEMO_ERROR)
      with gr.Accordion("Legacy catalog fallback",open=False):
        gr.Markdown("The unchanged legacy XLM-R catalog remains available for rollback; it is not mixed with the demo workbook.")
        gr.DownloadButton("Download legacy catalog",value=str(LEGACY_XLSX)); gr.Dataframe(LEGACY_DF,interactive=False,max_height=500)
    with gr.Tab("Communication Contract"):
      gr.Markdown("### Communication Contract\nVersioned local demo adapters persist structured events for CEO, Marketing and Inventory. No external agent is connected; reports are derived from KAYAN's SQLite records.")
      with gr.Tabs():
        with gr.Tab("CEO"):
          gr.Markdown("#### Task management\nCEO owns task creation and editing. Sales records Done only after verifying the assigned work. Tasks have only **Done** and **Not Done** states.")
          with gr.Row():
            ceo_assigned_date=gr.Textbox(value=business_date(),label="Tasks assigned for date (YYYY-MM-DD)")
            ceo_task_title=gr.Textbox(label="New task",placeholder="Describe the work for Sales")
            ceo_create_task=gr.Button("Assign task to Sales",variant="primary")
          ceo_task_feedback=gr.Markdown()
          ceo_task_status=gr.Markdown()
          ceo_task_table=gr.Dataframe(value=_task_frame(STORE.list_ceo_tasks(business_date())),label="CEO task checklist",interactive=False)
          with gr.Accordion("Edit task / Sales completion update",open=False):
            ceo_task_picker=gr.Dropdown(choices=[(f"{r['title']} — {r['task_id'][-8:]}",r["task_id"]) for r in STORE.list_ceo_tasks(business_date())],label="Select task")
            with gr.Row():
              ceo_edit_title=gr.Textbox(label="Task title / description")
              ceo_edit_date=gr.Textbox(label="Assigned execution date (YYYY-MM-DD)")
            with gr.Row():
              ceo_update_task=gr.Button("Save CEO task edit")
              ceo_task_done=gr.Checkbox(label="Sales verified this task as Done",value=False)
              ceo_save_task_status=gr.Button("Save Sales completion status")
            ceo_edit_status=gr.Markdown()
          gr.Markdown("#### Saved daily CEO reports")
          ceo_history_picker=gr.Dropdown(choices=communication_day_choices(),label="Business day")
          ceo_history_status=gr.Markdown("Choose a day to view its saved report.")
          ceo_history_tasks=gr.Dataframe(label="Historical task checklist",interactive=False)
        with gr.Tab("Marketing"):
          gr.Markdown("#### Customer interests and feedback\nCustomer labels and internal references are retained for attribution. Full transcripts are not sent into reports. Interest, likes, acceptance, rejection, unmet demand and MOCK purchase are separate.")
          with gr.Row():
            marketing_from=gr.Textbox(value=business_date(),label="From date (YYYY-MM-DD)")
            marketing_to=gr.Textbox(value=business_date(),label="To date (YYYY-MM-DD)")
            marketing_refresh_button=gr.Button("Refresh Marketing")
          marketing_summary=gr.Markdown()
          with gr.Row():
            marketing_products=gr.Dataframe(label="Product interest and demand",interactive=False)
            marketing_feedback=gr.Dataframe(label="Customer-specific signals",interactive=False)
          gr.Markdown("#### Saved daily Marketing reports")
          marketing_history_picker=gr.Dropdown(choices=communication_day_choices(),label="Business day")
          marketing_history_summary=gr.Markdown("Choose a day to view its saved report.")
          marketing_history_products=gr.Dataframe(label="Historical product summary",interactive=False)
          marketing_history_feedback=gr.Dataframe(label="Historical customer-specific feedback",interactive=False)
        with gr.Tab("Inventory"):
          gr.Markdown("#### Confirmed MOCK purchases and variant movement\nOnly a successfully persisted, explicitly confirmed MOCK request is counted. Adding to Selected Products is not a sale. Workbook stock is simulated DEMO data; there is no authoritative inventory feed or stock deduction.")
          with gr.Row():
            inventory_from=gr.Textbox(value=business_date(),label="From date (YYYY-MM-DD)")
            inventory_to=gr.Textbox(value=business_date(),label="To date (YYYY-MM-DD)")
            inventory_refresh_button=gr.Button("Refresh Inventory")
          inventory_summary=gr.Markdown()
          inventory_orders=gr.Dataframe(label="Confirmed MOCK order history",interactive=False)
          inventory_sales=gr.Dataframe(label="Purchased product variants",interactive=False)
          inventory_stock=gr.Dataframe(value=pd.DataFrame([{"Product ID":v.product_id,"Variant ID":v.variant_id,"Color":v.color,"Size":v.size,
              "Simulated demo stock":v.stock_qty,"Price (EGP)":v.price_egp,"Source status":v.stock_status} for v in DEMO_CATALOG.variants] if DEMO_CATALOG else [],
              columns=["Product ID","Variant ID","Color","Size","Simulated demo stock","Price (EGP)","Source status"]),
              label="Workbook variant stock (simulated demo stock; no low-stock claim)",interactive=False,max_height=320)
          gr.Markdown("#### Saved daily Inventory reports")
          inventory_history_picker=gr.Dropdown(choices=communication_day_choices(),label="Business day")
          inventory_history_summary=gr.Markdown("Choose a day to view its saved report.")
          inventory_history_orders=gr.Dataframe(label="Historical confirmed MOCK orders",interactive=False)
          inventory_history_sales=gr.Dataframe(label="Historical variant sales summary",interactive=False)
    product_search_button.click(demo_search,[product_query,product_stock,product_top_k],[product_search_status,product_search_gallery,product_search_table,product_search_technical])
    # Chat history is output-only. SQLite is authoritative, so feeding the
    # Chatbot back into Send would serialize image Components as event inputs
    # and can break the next customer turn after an image was displayed.
    send.click(chat,[message,session,cart_id],[chatbot,message,session,cart_html,item_table,remove_picker,total,status]); message.submit(chat,[message,session,cart_id],[chatbot,message,session,cart_html,item_table,remove_picker,total,status])
    clear_chat_button.click(clear_chat,[session],[chatbot,session,status])
    transcript_refresh.click(copy_safe_transcript,[session],[transcript_box])
    # Synchronize Gradio's built-in Chatbot clear control with SQLite too. The
    # explicit button above provides a reliable application-level operation.
    chatbot.clear(clear_chat,[session],[chatbot,session,status])
    customer_picker.change(switch_customer_cart,[customer_picker],[chatbot,session,active_label,cart_id,cart_html,item_table,remove_picker,total,status])
    customer_picker.change(remove_confirmation_label,[customer_picker],[remove_target])
    new_button.click(new_customer_cart,[new_name,new_customer_key],[customer_picker,new_name,chatbot,session,active_label,cart_id,cart_html,item_table,remove_picker,total,status,new_customer_key])
    end_button.click(end_customer_cart,[session],[customer_picker,chatbot,session,active_label,cart_id,cart_html,item_table,remove_picker,total,status])
    customer_remove_button.click(remove_customer_selected,[session,customer_picker,active_label,remove_confirmation],[customer_picker,chatbot,session,active_label,cart_id,cart_html,item_table,remove_picker,total,status,remove_target,removed_customer_picker],js="(session, customerId, label, confirmed) => [session, customerId, label, confirmed && window.confirm('Archive the selected customer [' + customerId + ']? This reversible action hides the customer from the active list and preserves history, purchases, and reports.')]")
    restore_customer_button.click(restore_removed_customer,[removed_customer_picker],[customer_picker,removed_customer_picker,chatbot,session,active_label,cart_id,cart_html,item_table,remove_picker,total,status])
    remove_picker.change(selected_item_quantity,[cart_id,remove_picker],[cart_quantity])
    quantity_button.click(update_item_quantity,[cart_id,remove_picker,cart_quantity],[cart_html,item_table,remove_picker,total,status])
    cart_remove_button.click(remove_item,[cart_id,remove_picker],[cart_html,item_table,remove_picker,total,status])
    buy_button.click(prepare_buy,[cart_id,session],[buy_status,purchase_token]); confirm_button.click(confirm_buy,[session,cart_id,confirm,purchase_token],[buy_status,confirm,purchase_token,chatbot])
    day_outputs=[day_snapshot,day_confirm,start_day_confirm,confirm,purchase_token,buy_status,day_status,day_label,customer_picker,chatbot,session,active_label,cart_id,cart_html,item_table,remove_picker,total,history_table,history_picker,ceo_history_picker,marketing_history_picker,inventory_history_picker]
    day_button.click(end_day,[day_confirm],day_outputs,js="(confirmed) => [window.confirm('End and save this business day? Historical reports will be preserved.')]" )
    start_day_button.click(start_new_day,[start_day_confirm],day_outputs,js="(confirmed) => [window.confirm('Start the next business day? The current day must already be closed.')]" )
    history_picker.change(historical_day_change,[history_picker],[history_snapshot])
    ceo_create_task.click(create_ceo_task_ui,[ceo_task_title,ceo_assigned_date],[ceo_task_table,ceo_task_picker,ceo_task_status,ceo_task_feedback])
    ceo_assigned_date.change(refresh_ceo_tasks,[ceo_assigned_date],[ceo_task_table,ceo_task_picker,ceo_task_status])
    ceo_task_picker.change(load_ceo_task_ui,[ceo_task_picker],[ceo_edit_title,ceo_edit_date,ceo_edit_status])
    ceo_update_task.click(update_ceo_task_ui,[ceo_task_picker,ceo_edit_title,ceo_edit_date],[ceo_task_table,ceo_task_picker,ceo_task_status,ceo_task_feedback])
    ceo_save_task_status.click(sales_task_status_ui,[ceo_task_picker,ceo_task_done,ceo_assigned_date],[ceo_task_table,ceo_task_picker,ceo_task_status,ceo_edit_status])
    ceo_history_picker.change(ceo_saved_day_report,[ceo_history_picker],[ceo_history_status,ceo_history_tasks])
    marketing_refresh_button.click(marketing_refresh,[marketing_from,marketing_to],[marketing_summary,marketing_products,marketing_feedback])
    marketing_from.change(marketing_refresh,[marketing_from,marketing_to],[marketing_summary,marketing_products,marketing_feedback])
    marketing_to.change(marketing_refresh,[marketing_from,marketing_to],[marketing_summary,marketing_products,marketing_feedback])
    marketing_history_picker.change(marketing_saved_report,[marketing_history_picker],[marketing_history_summary,marketing_history_products,marketing_history_feedback])
    inventory_refresh_button.click(inventory_refresh,[inventory_from,inventory_to],[inventory_summary,inventory_orders,inventory_sales,inventory_stock])
    inventory_from.change(inventory_refresh,[inventory_from,inventory_to],[inventory_summary,inventory_orders,inventory_sales,inventory_stock])
    inventory_to.change(inventory_refresh,[inventory_from,inventory_to],[inventory_summary,inventory_orders,inventory_sales,inventory_stock])
    inventory_history_picker.change(inventory_saved_report,[inventory_history_picker],[inventory_history_summary,inventory_history_orders,inventory_history_sales])
    demo.load(restore_active_customer,outputs=[customer_picker,chatbot,session,active_label,cart_id,cart_html,item_table,remove_picker,total,status])
  return demo

if __name__=="__main__":
    build_ui().launch(server_name="127.0.0.1",server_port=int(os.getenv("KAYAN_PORT","7860")),inbrowser=False,allowed_paths=[str(DEMO_ASSETS)])
