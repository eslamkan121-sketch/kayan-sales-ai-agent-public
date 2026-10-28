# Local model assets

This repo omits model weights. Do not upload Hugging Face tokens, private checkpoints, or customer data to GitHub.

## Trained XLM-R checkpoint

The app looks for `models/best.pt` by default. Obtain the exact trained checkpoint and its approved tokenizer/config assets from the project owner through an authorized channel. Alternatively set `KAYAN_CHECKPOINT` to the local checkpoint path. The private trained checkpoint is not available to a friend merely by cloning this public source repository.

## Optional Qwen3 conversation model

Qwen3 is only loaded from local files by this app. If you have accepted the publisher's model terms and have adequate disk/RAM, place Qwen3-0.6B in `models/Qwen3-0.6B` or set `KAYAN_QWEN_MODEL_PATH`. Download weights directly from the model publisher using your own account where required. If the directory is absent, the app uses the safe fallback and does not claim that Qwen generated the reply.

## Without model assets

The app can still be launched to review the demo catalog and interface. XLM-R-ranked Product Search and Qwen3-generated conversation are unavailable until their model files are supplied. The bundled workbook remains simulated demo data, not live inventory.
