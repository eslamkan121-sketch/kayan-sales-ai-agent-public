"""Representative coverage checks for state boundaries and catalog grounding.

This uses the deterministic no-model configuration to isolate orchestration.
It never accesses training or official-test data.
"""
from src.agent import SalesAgent
from src.catalog import build_catalog


agent = SalesAgent(build_catalog(), checkpoint_path=None, qwen_path="models/not-installed")


def ask(message, state=None):
    return agent.customer_reply(message, state)


# Casual, preference, bilingual category and hard budget/size filters.
reply, products, state = ask("Hello, I need help")
assert not products and "help" in reply.lower()
reply, products, state = ask("بحب الكحلي", state)
assert state.favorite_color == "navy" and not products
reply, products, state = ask("navy blazer size M under 3000", state)
assert products and all(p["product"].category == "blazer" and p["product"].color == "Navy" for p in products)
assert all("M" in p["product"].sizes and p["product"].price_egp <= 3000 for p in products)

# A new occasion clears incompatible active category/color constraints.
reply, products, state = ask("I need something for university", state)
assert products and state.occasion == "university" and state.color is None and state.category is None
assert all(item["product"].category in {"t-shirt", "hoodie", "jeans"} for item in products)
reply, products, state = ask("لبس للرياضة", state)
assert not products and state.occasion == "sports"

# Explicit alternative permission allows a broadened result; ordinal references
# remain a clarification rather than a guessed, unshown product.
reply, products, state = ask("عاوز تيشرت احمر", state)
assert not products and state.color == "red"
reply, products, state = ask("وريني بديل", state)
assert products and products[0]["product"].color != "Red"
reply, products, state = ask("هات التاني", state)
assert not products and "التاني" in reply

# Unsupported and TrustGate workflows neither execute nor mutate catalog state.
reply, products, state = ask("عاوز شحن للعنوان", state)
assert not products and "شحن" in reply
reply, products, state = ask("عاوز خصم على المنتج", state)
assert not products and "موافقة" in reply

# The no-local-Qwen path is explicit and returns a grounded fallback.
assert not agent.conversation.available
print("Conversation coverage suite passed.")
