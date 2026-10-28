"""End-to-end regression for pending offers, SKU continuity, and beach intent."""
from src.agent import SalesAgent
from src.catalog import build_catalog


agent = SalesAgent(build_catalog(), checkpoint_path=None, qwen_path="models/not-installed")


def ask(message, state=None):
    return agent.customer_reply(message, state)


# Known observed sequence. Every transition asserts its stateful meaning rather
# than relying on a fixed response template.
reply, products, state = ask("أنا بحب لون الأزرق هل عندك حاجه لونها ازرق")
assert not products and state.pending_action == "alternative_offer" and state.color == "blue"
reply, products, state = ask("ايوه ياريت", state)
assert not products and state.pending_action == "alternative_scope" and "اللون" in reply
reply, products, state = ask("اللون", state)
assert not products and state.pending_action == "color_choice" and "الألوان المتاحة" in reply
reply, products, state = ask("طب عندكم تيشرت ابيض", state)
assert products and state.primary_sku == state.last_recommended_sku
primary = next(item["product"] for item in products if item["product"].sku == state.primary_sku)
reply, products, state = ask("لا عاوز اسأل عن سعره و مقاساته المتاحة", state)
assert products and products[0]["product"].sku == primary.sku
arabic_sizes = {"XS": "إكس سمول", "S": "سمول", "M": "ميديم", "L": "لارج", "XL": "إكس لارج", "XXL": "إكس إكس لارج"}
assert str(primary.price_egp) in reply and all(arabic_sizes.get(size, size) in reply for size in primary.sizes)
reply, products, state = ask("طب عندك حاجه لخروجة البحر", state)
assert products and state.occasion == "beach"
assert all(item["product"].category in {"shirt", "t-shirt"} for item in products)
reply, products, state = ask("اقترح عليا حاجه لخروجة البحر", state)
assert products and state.occasion == "beach"

# Arabic/English/mixed sea-outing forms reach the same fresh-task mapping.
for message in ("خروجه بحر", "فسحة على البحر", "beach outing", "عاوز outfit للبحر"):
    reply, products, fresh = ask(message)
    assert fresh.occasion == "beach"
    if products:
        assert all(item["product"].category in {"shirt", "t-shirt"} for item in products)

print("Pending-action end-to-end regression passed.")
