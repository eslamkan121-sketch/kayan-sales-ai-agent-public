"""Non-training smoke tests for English and Egyptian Arabic catalog queries."""
from pathlib import Path
import sys

from src.agent import SalesAgent
from src.catalog import build_catalog


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

agent = SalesAgent(build_catalog(), Path("models/best.pt"))
queries = [
    "black cotton t-shirt size M under 500 EGP",
    "navy blazer for office meetings",
    "عايز تي شيرت اسود قطن مقاس M تحت ٥٠٠ جنيه",
    "عاوز جاكيت شتوي مبطن كحلي",
]
for query in queries:
    results = agent.search(query)
    assert results, f"No results for {query}"
    assert all(item["product"].stock > 0 for item in results)
    if "t-shirt" in query:
        assert "T-Shirt" in results[0]["product"].name, results[0]["product"].name
    print(query)
    print([(item["relation"], item["product"].sku, item["product"].name) for item in results[:3]])
print(agent.status_text())
