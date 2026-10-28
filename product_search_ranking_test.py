"""Regression tests for workbook-backed, product-level XLM-R search."""
from pathlib import Path
import json
import sys

import pandas as pd

from src.demo_catalog import load_demo_catalog
from src.product_search import product_candidates, product_search_score, rank_product_candidates
from src.agent import XLMRRelevanceModel

ROOT = Path(__file__).resolve().parent
WORKBOOK = ROOT / "catalog_preview_review" / "outputs" / "KAYAN_catalog_color_variants_NEW.xlsx"
CATALOG = load_demo_catalog(WORKBOOK, ROOT / "catalog_preview_review" / "preview_assets")
METADATA = json.loads((ROOT / "models" / "checkpoint_metadata.json").read_text(encoding="utf-8"))


class DeterministicRankingStub:
    """Test-only stub for stable ranking mechanics; never used by the UI."""
    status = type("Status", (), {"loaded": True})()
    id2label = {value: key for key, value in METADATA["label2id"].items()}

    def predict_detailed(self, query, products):
        rows = []
        for index, _product in enumerate(products):
            logits = {"E": 4.0-index*.1, "S": 2.0, "C": 1.0, "I": 0.0}
            values = list(logits.values())
            exp = [__import__("math").exp(x) for x in values]
            denom = sum(exp)
            probabilities = {label: value/denom for label, value in zip(logits, exp)}
            rows.append({"logits": logits, "probabilities": probabilities,
                         "predicted_label": max(logits, key=logits.get), "confidence": max(probabilities.values())})
        return rows


def main():
    # 1. Active catalog/sheet and source rows are the workbook, not legacy data.
    assert CATALOG.validation["sheet"] == "Preview variants"
    assert CATALOG.validation["rows"] == 133 and CATALOG.validation["unique_product_ids"] == 24
    candidates = product_candidates(CATALOG.variants, only_in_stock=True, query="blue t-shirt")

    # 2. Arabic and 3. English text never narrows the academic candidate pool.
    arabic_ids = [item["product_id"] for item in product_candidates(CATALOG.variants, True, "هل عندكم تيشيرتات؟")]
    english_ids = [item["product_id"] for item in product_candidates(CATALOG.variants, True, "blue t-shirt")]
    assert arabic_ids == english_ids
    assert any(item["representative"].category == "t-shirt" for item in candidates)

    # 4. Exact product candidate is present; 5. unrelated queries still receive
    # candidates to classify rather than a false lexical-empty result.
    assert any(item["representative"].name == "Essential Crew T-Shirt" for item in candidates)
    assert next(item for item in candidates if item["product_id"] == "PVP-020")["in_stock_facet_match"] is True
    unrelated = product_candidates(CATALOG.variants, True, "unrelated dental floss")
    assert len(unrelated) == 24

    # 6. ESCI index mapping is read from the saved trained checkpoint metadata.
    assert METADATA["label2id"] == {"E": 0, "S": 1, "C": 2, "I": 3}

    # 7. The established Product Search score matches its documented code path.
    probabilities = {"E": .5, "S": .3, "C": .1, "I": .1}
    assert abs(product_search_score(probabilities) - .56) < 1e-12

    # 8. Product-level mode is one row per distinct Product ID (not per variant).
    assert len(candidates) == len({item["product_id"] for item in candidates})

    # 9. Multi-variant rows retain only actual source colors/sizes and variants.
    frame = pd.read_excel(WORKBOOK, sheet_name="Preview variants")
    first = candidates[0]
    source = frame[(frame["Product ID"] == first["product_id"]) & (pd.to_numeric(frame["Stock qty"]) > 0)]
    assert set(first["colors"]) == set(source["Color"].astype(str).str.strip())
    assert set(first["sizes"]) == set(source["Size"].astype(str).str.strip())
    assert {row.variant_id for row in first["variants"]} == set(source["Variant ID"].astype(str).str.strip())
    assert first["stock_qty"] == int(pd.to_numeric(source["Stock qty"]).sum())

    # 10. Top 5 is exactly the prefix of Top 10 from one stable ranked set.
    ranked = rank_product_candidates("Arabic t-shirt query", candidates, DeterministicRankingStub())
    top5, top10 = ranked[:5], ranked[:10]
    assert [r["product_id"] for r in top5] == [r["product_id"] for r in top10[:5]]
    assert [r["product_id"] for r in ranked] == [r["product_id"] for r in rank_product_candidates("same", candidates, DeterministicRankingStub())]

    # 11. Returned labels, probabilities and relevance values derive from logits.
    for item in ranked:
        probs = item["prediction"]["probabilities"]
        assert item["prediction"]["predicted_label"] == max(item["prediction"]["logits"], key=item["prediction"]["logits"].get)
        assert abs(item["ranking_score"] - probs["E"] - .2*probs["S"]) < 1e-12

    # 12. Missing weights cannot produce synthetic detailed predictions.
    unavailable = XLMRRelevanceModel(None)
    assert unavailable.status.loaded is False
    assert unavailable.predict_detailed("test", [candidates[0]["product"]]) == []
    print("PASS: 12 product-search regression assertions (source data, candidate coverage, product identity, ESCI mapping, score, Top-K stability, and missing-checkpoint behavior).")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
