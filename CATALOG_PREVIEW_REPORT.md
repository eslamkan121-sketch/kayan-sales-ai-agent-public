# KAYAN proposed catalog preview — review only

## Status

This is a separate preview artifact. It is **not approved**, is not connected to Gradio, XLM-R retrieval, the Sales Agent, or the existing Excel export, and does not replace any current catalog record.

## Recommendation

Use a two-stage approach:

1. Use a manually curated, transparently labelled demo catalog for prototype review.
2. For production, ingest an owner-authorized supplier, PIM, or commerce-platform feed keyed by immutable variant SKU. Require a contractually permitted product-image source and a variant-level inventory feed.

This is preferable to adopting a public fashion-image dataset as inventory. Public datasets can help with taxonomy research, but do not provide a trustworthy live color-size stock feed or a safe basis for real prices/availability.

## Preview contents

- 18 varied garment concepts: shirts, blazer, jeans, dresses, jackets, cardigans, T-shirts, sneakers, activewear, skirts, hoodies, trousers, polos, bags, and scarves.
- 77 explicit product + color + size variants.
- Simulated per-variant stock quantities and simulated demo prices, labelled in every row.
- Optional field examples: material, fit, pattern, sleeve/length, season, occasion.
- Three original AI-generated, non-branded concept boards. They are visual placeholders only, not SKU-specific product photography and not production-cleared.

## Proposed schema

The required identity is `product_id` plus a unique `variant_id`. Each variant owns its color, size, `stock_quantity`, price, currency, provenance status, and image rights metadata. The agent should consider a color-size selection available only when that exact `variant_id` has `stock_quantity > 0`.

Every field must carry one of: `Verified source data`, `Curated demo`, or `Simulated demo`. Never derive an unprovided material, fit, pattern, stock quantity, or price.

## External-source assessment — not adopted

| Candidate | What it offers | Rights/access concern | Variant inventory | Decision |
|---|---|---|---|---|
| Kaggle Fashion Product Images (Small) | Category labels, product IDs, descriptions and images; Kaggle lists MIT | Image chain-of-title and commercial clearance need owner/legal review even if the listing says MIT | No verified live color-size stock feed stated | Do not download/adopt automatically |
| DeepFashion | Research images with category/attribute annotations | Official terms restrict it to non-commercial research; it says the source images are not owned by MMLAB | No retail stock feed | Reject for a commercial sales catalog |
| Unsplash | General editorial/lifestyle imagery | Broad image license, but not SKU-accurate catalog data; rights around depicted brands/people/property need review | None | Do not use for product facts |

Official references: [Unsplash License](https://unsplash.com/license), [Unsplash Terms](https://unsplash.com/terms), [DeepFashion agreement](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/DeepFashionAgreement.pdf), [Kaggle Fashion Product Images (Small)](https://www.kaggle.com/paramaggarwal/fashion-product-images-small/notebooks).

## Current-project impact

None. The existing 100-item illustrative demo catalog and all model/checkpoint/dataset files remain unchanged. No external catalog, images, prices, or inventory data were downloaded or incorporated.
