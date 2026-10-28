# KAYAN conversation coverage map

The agent is a catalog-grounded assistant, not a human or a commerce system.
The active request holds category, color, size, budget, occasion, fit and style;
durable preferences hold favorite colors and fit/style choices; displayed history
holds only SKUs actually named to the customer. Catalog fields are authoritative
for product facts, and TrustGate independently blocks commercial mutations.

| Coverage area | Recognize / state update | Catalog authority and response | Representative regression coverage |
|---|---|---|---|
| Greeting, thanks, help | greeting/casual; `greeted` only | no call; concise answer | Arabic and English greeting/thanks |
| Categories and attributes | category/color/size/budget/fit/style slots | hard-filter, then retrieve and rank; catalog is authoritative | Arabic/English T-shirt, shirt, trousers, jeans, hoodie, jacket, dress, shoes |
| Occasions | work/interview is a fresh bounded workwear task; wedding, beach, university, travel, sports clear incompatible request filters | advice first; search only on a product request | interview switch; wedding, beach, university, travel and sports advice |
| Preferences vs constraints | `favorite_color`/`preferred_*` persist; explicit color is only an active constraint | no catalog call for pure preference or recall | favorite-color recall and later explicit color change |
| Availability and unavailable constraints | explicit color/size/category/budget stays hard | catalog only; report unavailable and ask permission for alternatives | unavailable red/blue/green; alternate request |
| Product facts and references | `last_recommended_sku`, `selected_sku`, rejected SKU list | exact SKU price, stock, color, sizes, details | `ده`, explicit black trousers, sizes/color follow-up |
| Ambiguous references and comparisons | do not invent ordinal identity; require a shown product | clarify if not resolvable; compare two known shown SKUs only | second-one ambiguity and two-product comparison |
| Corrections and topic changes | new explicit slots replace old slots; work/interview clears stale filters | catalog call only for a shopping action | blue → green → white T-shirt → interview |
| Styling / fit limits | style and occasion only; never derive a precise size from height/weight | no product fact generation | oversized vs regular; wedding advice |
| Unsupported / sensitive work | shipping/delivery is unsupported; payment/order/discount/refund/stock mutation is TrustGate | no catalog mutation; hand off/refuse | shipping limitation, discount/order refusal |
| Fault tolerance and language | Arabic/English/mixed text and common alias variants; model fallback is logged | no inference may create facts; malformed/unavailable Qwen falls back | Qwen-unavailable and malformed-response tests |

Tests are behavioral, not a phrase lookup: every case asserts the relevant state
boundary, catalog-only factual source, or refusal/clarification outcome. The
coverage suite is intentionally separate from training data and does not read an
official test split.
