"""Regression for fact-free Qwen phrasing and exact image/product association."""
from pathlib import Path

from src.demo_catalog import load_demo_catalog
from src.demo_conversation import DemoConversationEngine


ROOT=Path(__file__).resolve().parent
CATALOG=load_demo_catalog(ROOT/"catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",ROOT/"catalog_preview_review/preview_assets")


def ranker(_query,candidates,limit):
    return sorted(candidates,key=lambda v:(v.product_id,v.variant_id))[:limit]


def main():
    calls=[]
    def qwen(message,approved,arabic):
        calls.append((message,approved,arabic))
        return ("أكيد، أقدر أساعدك تختار ستايل مناسب.",True,"accepted")
    engine=DemoConversationEngine(CATALOG,ranker,qwen=qwen)
    reply,state,_=engine.respond("أنا رايح خروجة ومحتاج رأيك في الستايل",{})
    assert reply=="أكيد، أقدر أساعدك تختار ستايل مناسب." and calls and calls[-1][2]

    reply,state,action=engine.respond("هاتلي خمس حجات بيضا بصور وأسعار ومقاسات",state)
    ids=state["demo_conversation"]["displayed_variant_ids"]
    assert len(ids)==len(set(ids))==5
    assert len(action["image_variant_ids"])==5
    assert all(CATALOG.get(variant_id).image_path.is_file() for variant_id in action["image_variant_ids"])
    assert "جنيه" in reply and "المقاسات" in reply
    first=ids[0]
    reply,state,action=engine.respond("وريني صورته",state)
    assert action["image_variant_ids"]==[first]
    assert state["demo_conversation"]["focused_variant_id"]==first
    print("PASS: Qwen phrase callback, five distinct priced/size-listed products, exact image IDs, and single-product image reference")


if __name__=="__main__":
    main()
