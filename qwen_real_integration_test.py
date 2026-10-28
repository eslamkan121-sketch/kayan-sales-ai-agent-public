"""Exercise the real local Qwen3 phrasing callback on a fact-free chat turn."""
import json
from pathlib import Path
import sys

from src.agent import SalesAgent
from src.catalog import build_catalog
from src.demo_catalog import load_demo_catalog
from src.demo_conversation import DemoConversationEngine


ROOT = Path(__file__).resolve().parent


def main():
    agent = SalesAgent(
        build_catalog(), ROOT / "models" / "best.pt", ROOT / "models" / "Qwen3-0.6B"
    )
    assert agent.conversation.available, agent.conversation.status
    catalog = load_demo_catalog(
        ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
        ROOT / "catalog_preview_review/preview_assets",
    )
    log_path = ROOT / "outputs/tests/qwen_real_integration.jsonl"
    engine = DemoConversationEngine(catalog, lambda _q, items, limit: items[:limit], log_path, qwen=agent.conversation.respond)
    message = "إيه الفرق بين القصة الواسعة والقصة العادية؟"
    reply, _state, action = engine.respond(message, {})
    raw_line = log_path.read_text(encoding="utf-8").splitlines()[-1]
    event = json.loads(raw_line[raw_line.index("{"):])
    assert action.get("add_variant_id") is None
    assert event["intent"] == "fit_advice"
    assert event["qwen"] == "used", f"Qwen was called but response was not accepted: {event['qwen']}"
    assert reply and "سعر" not in reply
    print("intent=fit_advice")
    print(f"Qwen available={agent.conversation.available}; UI route Qwen status={event['qwen']}")
    print(f"raw Qwen candidate={agent.conversation.last_generated_candidate}")
    print(f"customer-facing response={reply}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
