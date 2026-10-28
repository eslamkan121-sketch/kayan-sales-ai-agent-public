"""Regression for requested lists, catalog scope, selection, and budget resets."""
from src.agent import SalesAgent
from src.catalog import build_catalog


agent = SalesAgent(build_catalog(), checkpoint_path=None, qwen_path="models/not-installed")


def ask(message, state=None):
    return agent.customer_reply(message, state)


reply, products, state = ask("أنا بحب لون الأبيض اديني ليست خمس حجات عندكم لونها ابيض")
assert 1 <= len(products) <= 5
assert len(products) == len({item["product"].sku for item in products})
assert state.displayed_skus == [item["product"].sku for item in products]
assert state.primary_sku == products[0]["product"].sku
assert all(item["product"].color == "White" for item in products)
assert "لقيت" in reply and "منتج" in reply

reply, products, state = ask("طب عارف أنا بحب لون ايه", state)
assert not products and "الأبيض" in reply

reply, products, state = ask("أنا بحب لون الأصفر كمان عندك حاجه لونها اصفر", state)
assert not products and set(state.favorite_colors) == {"white", "yellow"}
assert state.color == "yellow"

# Unsupported catalog properties must not reuse the prior yellow-unavailable answer.
reply, products, state = ask("طب عندكم تيشرت مخطط", state)
assert not products and "مخطط" in reply and "أصفر" not in reply
assert state.color is None
reply, products, state = ask("طب هل عندكم بدله", state)
assert not products and "بدلة" in reply and "أصفر" not in reply

# Beach opens a fresh, supported task and provides a catalog-grounded item.
reply, products, state = ask("طب اقترح عليا حاجه لخروجه البحر", state)
assert products and state.occasion == "beach" and state.color is None
assert products[0]["product"].category in {"shirt", "t-shirt"}
reply, products, state = ask("في الوان ايه", state)
assert not products and "الألوان المتاحة" in reply

# Choosing black uses the active beach category, establishes a primary SKU,
# and factual follow-ups cannot drift to another color or product.
reply, products, state = ask("تمام هختار اسود ايه المقاسات المتاحه", state)
assert len(products) == 1
black_product = products[0]["product"]
assert black_product.color == "Black"
assert black_product.category in {"shirt", "t-shirt"}
assert state.primary_sku == black_product.sku
assert "بديل" not in reply and all(size in reply for size in ("سمول", "ميديم", "لارج"))
reply, products, state = ask("سعره كام", state)
assert len(products) == 1 and products[0]["product"].sku == black_product.sku
assert str(black_product.price_egp) in reply

reply, products, state = ask("تعرف الألوان إلى بحبها", state)
assert not products and "الأبيض" in reply and "الأصفر" in reply

# “Cheaper than 350” is a strict budget, not a request to report the preceding
# black product’s price. The synthetic catalog has no item below this ceiling.
reply, products, state = ask("هل في حاجه ارخص من ٣٥٠ جنيه", state)
assert state.max_price_egp == 349 and not products and "350" in reply
reply, products, state = ask("تمام عاوز اي لون او اي نوع ارخص من ٣٥٠ جنيه", state)
assert state.max_price_egp == 349 and state.color is None and not state.categories and not products
assert "350" in reply

print("Catalog scope, list, selection, and reset end-to-end regression passed.")
