"""Reproducible Egyptian-Arabic multi-turn regression suite (no training)."""
from src.agent import SalesAgent
from src.catalog import build_catalog

agent = SalesAgent(build_catalog(), checkpoint_path=None, qwen_path="models/not-installed")


def ask(message, state=None):
    reply, products, next_state = agent.customer_reply(message, state)
    return reply, products, next_state


# Actual failed-conversation variants: spelling errors, remembered colour,
# topic switches, and direct styling questions must not restart product search.
reply, products, state = ask("انا عايز تيشرت لونه احمر مقاس L هل فيه")
assert not products and state.active_context == "shopping" and state.size == "L"
assert "أحمر" in reply and "تحب" in reply

reply, products, state = ask("لا بس أنا بحب اللون الاحمر", state)
assert not products and state.favorite_color == "red" and "هفتكر" in reply

reply, products, state = ask("طب هل نت عرف ن بحب لون يه", state)
assert not products and "الأحمر" in reply and "تيشيرت" not in reply

reply, products, state = ask("طب قترح علي حجه طولي 190 و وزني 100 نت شيف يه يلقي علي في لفرح", state)
assert products and state.occasion == "wedding" and "الأحمر مش متوفر" not in reply

reply, products, state = ask("طب قترح علي لبس بحر", state)
assert products and state.occasion == "beach" and all(item["product"].category in {"shirt", "t-shirt"} for item in products)

reply, products, state = ask("انا مش عاوز اشتري دلوقتي بس عاوز رايك ايه الفرق بين Oversized وRegular Fit و انهي واحد يناسبني", state)
assert not products and "Oversized" in reply and "Regular Fit" in reply and "تيشيرت" not in reply

# New customer: vague styling, light colours, color changes, and catalog facts.
reply, products, state = ask("بص انا عندي خروجة مهمة بكرة ومحتار البس ايه ممكن تساعدني")
assert not products and "كاجوال" in reply

reply, products, state = ask("اختيار ستايل انا بحب الالوان الفاتحه عندكم ايه فيها الوان فاتحه", state)
assert state.color_tone == "light" and ("الأبيض" in reply or "أبيض" in reply)

reply, products, state = ask("هل عندكم تيشرت احمر عشان انا بحب اللون الاحمر", state)
assert not products and state.favorite_color == "red" and state.category == "t-shirt"

reply, products, state = ask("هل في الوان تانيه", state)
assert not products and "الأحمر مش متوفر" not in reply and "الألوان المتاحة" in reply

reply, products, state = ask("انت عارف انا بحب لون ايه", state)
assert not products and "الأحمر" in reply

reply, products, state = ask("طب اللون الازرق هل في حاجه للون الازرق تيشرت او اي حاجه", state)
assert not products and "الأحمر مش متوفر" not in reply

reply, products, state = ask("طب عندكم تيشرت ابيض وكان الاسعار والمقاسات", state)
assert products and "جنيه" in reply and "المقاسات المتاحة" in reply

# Multi-category retrieval, work/classic context, and last-product references.
reply, products, state = ask("طب هل عندك اي حاجه لونها احمر تيشرت او قميص مثلا")
assert not products and set(state.categories) == {"t-shirt", "shirt"} and "أحمر" in reply
reply, products, state = ask("طب اقترح عليا حاجه عشان خارج في موعد عمل", state)
assert not products and state.occasion == "work" and "المناسبة شغل" in reply
reply, products, state = ask("للشغل عاوز ستايل كلاسيك", state)
assert products and state.style == "classic" and state.occasion == "work"
reply, products, state = ask("كلاسيك ايه هو المتاح", state)
assert products and all(item["product"].category in {"shirt", "blazer"} for item in products)
last_sku = state.last_recommended_sku
reply, products, state = ask("حلو ده بكام و هل متوفر مقاس لارج", state)
assert products and products[0]["product"].sku == last_sku
assert "جنيه" in reply and "مقاس لارج" in reply and "متوفر" in reply

# Unseen open-ended shopping journey: cinema → preference → recommendation →
# feedback → selection → factual follow-up. This guards against memorizing only
# the earlier failure prompts.
reply, products, journey = ask("انا رايح السينما وعاوز لوك مريح بس شيك")
assert not products and journey.occasion == "cinema" and "للسينما" in reply
reply, products, journey = ask("انا بحب الاسود بس وريني تيشيرت مناسب", journey)
assert products and journey.favorite_color == "black" and journey.last_recommended_sku
first_sku = journey.last_recommended_sku
reply, products, journey = ask("مش عاجبني ده عاوز حاجة مختلفة", journey)
assert products and journey.last_recommended_sku != first_sku and first_sku in journey.rejected_skus
reply, products, journey = ask("هختار ده", journey)
assert not products and journey.selected_sku == journey.last_recommended_sku
reply, products, journey = ask("ده بكام وهل مقاس لارج متاح", journey)
assert products and "جنيه" in reply and "مقاس لارج" in reply

# Critical identity and preference regression: a colour browsing turn must not
# overwrite the selected black-trousers SKU or turn a later factual question
# into a substitute recommendation.
reply, products, identity_state = ask("أنا بحب لون الأصفر هل عندك حاجه لونها اصفر")
assert identity_state.favorite_color == "yellow"
reply, products, identity_state = ask("عاوز بنطلون اسود", identity_state)
assert products and products[0]["product"].color == "Black"
black_trousers_sku = identity_state.last_recommended_sku
black_trousers = next(item["product"] for item in products if item["product"].sku == black_trousers_sku)
reply, products, identity_state = ask("عندك الوان ايه", identity_state)
assert not products and identity_state.last_recommended_sku == black_trousers_sku
assert "أسود" in reply and "كحلي" in reply and "زيتي" in reply
reply, products, identity_state = ask("تمام انت عارف بحب لونه ايه", identity_state)
assert not products and "الأصفر" in reply
reply, products, identity_state = ask("سعر البنطلون الاسود بكام", identity_state)
assert products and products[0]["product"].sku == black_trousers_sku
assert str(black_trousers.price_egp) in reply and "بديل" not in reply

# Explicit unavailable colors, new workwear task boundaries, and factual
# follow-ups must not be silently converted into a different catalog item.
reply, products, critical_state = ask("أنا بحب لون الأزرق هل عندك حاجه لونها ازرق")
assert critical_state.favorite_color == "blue" and not products and "أزرق" in reply
reply, products, critical_state = ask("طب عندكم حاجه لونها اخضر", critical_state)
assert critical_state.color == "green" and not products and "أخضر" in reply
reply, products, critical_state = ask("طب عندكم تيشرت ابيض", critical_state)
assert products and all(item["product"].category == "t-shirt" and item["product"].color == "White" for item in products)
reply, products, critical_state = ask("عاوز لبس في مقابله شغل", critical_state)
assert products and critical_state.occasion == "work" and critical_state.color is None
assert critical_state.categories == ["shirt", "blazer"]
assert all(item["product"].category in {"shirt", "blazer"} and item["product"].stock > 0 for item in products)
workwear_sku = critical_state.last_recommended_sku
workwear_product = next(item["product"] for item in products if item["product"].sku == workwear_sku)
reply, products, critical_state = ask("حلو ده متوفر مقاس كام و اللون", critical_state)
assert products and products[0]["product"].sku == workwear_sku
arabic_size_names = {"XS": "إكس سمول", "S": "سمول", "M": "ميديم", "L": "لارج", "XL": "إكس لارج", "XXL": "إكس إكس لارج"}
assert all(arabic_size_names[size] in reply for size in workwear_product.sizes)
assert {"Black": "أسود", "White": "أبيض", "Navy": "كحلي", "Beige": "بيج", "Olive": "زيتي"}[workwear_product.color] in reply

assert not agent.conversation.available
print("Conversation intelligence regression suite passed.")
