"""End-to-end regression for price anchors, budgets, and multi-color preferences."""
from src.agent import ConversationState, SalesAgent
from src.catalog import build_catalog


catalog = build_catalog()
agent = SalesAgent(catalog, checkpoint_path=None, qwen_path="models/not-installed")


def ask(message, state=None):
    return agent.customer_reply(message, state)


# Reported journey: every fact follows the primary SKU rather than a new search.
reply, products, state = ask("أنا بحب لون الأزرق هل عندك حاجه لونها ازرق")
assert not products and state.favorite_colors == ["blue"]
reply, products, state = ask("طب عندكم تيشرت ابيض", state)
assert products and state.primary_sku
white_tee = next(item["product"] for item in products if item["product"].sku == state.primary_sku)
assert white_tee.category == "t-shirt" and white_tee.color == "White"

reply, products, state = ask("حلو ده سعره كام", state)
assert products[0]["product"].sku == white_tee.sku
assert str(white_tee.price_egp) in reply

reply, products, state = ask("طب مفيش حاجه ارخص يكون نفس لون ابيض", state)
assert all(item["product"].price_egp < white_tee.price_egp for item in products)
if not products:
    assert str(white_tee.price_egp) in reply and "أرخص" in reply
current_product = products[0]["product"] if products else white_tee

# A failed cheaper search must not overwrite the prior factual reference.
reply, products, state = ask("حلوه ده ايه المقاسات المتاحه", state)
assert products[0]["product"].sku == current_product.sku
assert all(size in reply for size in ("سمول", "ميديم", "لارج"))
reply, products, state = ask("طب كام سعره", state)
assert products[0]["product"].sku == current_product.sku
assert str(current_product.price_egp) in reply

reply, products, state = ask("طب مفيش حاجه ارخص", state)
assert all(item["product"].price_egp < current_product.price_egp for item in products)

# A standalone strict budget creates a new active request and never returns an
# old product merely because it was the last reference.
reply, products, state = ask("أنا عاوز حاجه سعرها اقل من ٣٠٠ جنيه", state)
assert state.max_price_egp == 299 and not products and "300" in reply
assert white_tee.sku not in [item["product"].sku for item in products]

# Long-term preferences remain after the active request changes.
reply, products, state = ask("طب عارف أنا بحب لون ايه", state)
assert "الأزرق" in reply
reply, products, state = ask("أنا بحب لون الأصفر و الأزرق هل في حاجه مكس", state)
assert not products and set(state.favorite_colors) == {"blue", "yellow"}
assert "مفيهوش منتج متسجل بأكتر من لون" in reply
reply, products, state = ask("طب عارف أنا بحب لون ايه", state)
assert "الأصفر" in reply and "الأزرق" in reply

# Prove the strict comparison rule when a cheaper catalog result does exist.
reference = max((product for product in catalog if product.stock > 0), key=lambda product: product.price_egp)
comparison_state = ConversationState(
    active_context="shopping", primary_sku=reference.sku,
    last_recommended_sku=reference.sku, reference_price_sku=reference.sku,
)
reply, products, comparison_state = ask("مفيش حاجه ارخص", comparison_state)
assert products
assert all(item["product"].price_egp < reference.price_egp for item in products)

# The supported English forms have exactly the same strict budget semantics.
for message in ("under EGP 300", "less than 300", "تحت 300"):
    reply, products, budget_state = ask(message)
    assert budget_state.max_price_egp == 299
    assert all(item["product"].price_egp < 300 for item in products)

print("Price, preference, and product-reference end-to-end regression passed.")
