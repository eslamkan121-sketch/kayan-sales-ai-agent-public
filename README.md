# KAYAN Sales AI Agent — Public Demo

A Gradio clothing-sales assistant demo with product search, Egyptian Arabic/English conversation, a workbook-backed catalog, customer sessions, selected products, mock purchase confirmation, and daily reports.

## Important demo-data note

The included workbook at `catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx` is a **demo catalog**. Prices and stock quantities are simulated; product pictures are illustrative concept images, not verified SKU photographs or live store inventory. A confirmed purchase in this app is only a MOCK request: it does not charge money or deduct stock.

## Run locally

Use Python 3.10 or newer. From the project directory:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The app creates local SQLite records and exports under `outputs/`; those files are intentionally excluded from Git.

## Models

Model weights are intentionally **not** included in this public repository. The trained KAYAN XLM-R checkpoint is private/large, and Qwen3 weights are several GB when installed with tokenizer assets. The app can start without them in fallback/demo mode, but classifier-based Product Search and Qwen-generated chat require their respective local assets.

- **XLM-R:** Request authorized access to the trained KAYAN checkpoint from the project owner. Place the approved checkpoint at `models/best.pt`, or set `KAYAN_CHECKPOINT` to its local path. The tokenizer/config assets must also be available as expected by `src/agent.py`. Never commit a private checkpoint or Hugging Face token.
- **Qwen3-0.6B (optional):** If you have permission and suitable hardware, install the publisher's Qwen3-0.6B files locally under `models/Qwen3-0.6B`, or set `KAYAN_QWEN_MODEL_PATH` to that directory. Without it the app uses its safe non-model fallback.

See [MODEL_SETUP.md](MODEL_SETUP.md) for details and limitations.

## Tests

Catalog and conversation unit tests:

```bash
python -m compileall -q app.py src
python demo_catalog_constraints_test.py
python demo_catalog_arabic_retrieval_test.py
python demo_price_integrity_test.py
python presentation_localization_test.py
```

Live Gradio/model integration tests require the local assets described above. Run `python run_full_test_suite.py` after installing them.

## Data and privacy

The repository contains demo source code, the simulated workbook and its illustrative images. It does not contain trained model weights, customer databases, customer conversations, logs, recovery backups, or real retailer inventory. Do not put credentials or private customer information in issues or commits.

No software license has been selected for this repository yet.
