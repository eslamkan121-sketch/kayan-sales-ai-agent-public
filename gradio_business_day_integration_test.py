"""Full live Gradio close/start/restart regression on a temporary SQLite database."""
from pathlib import Path
import json
import os
import sqlite3
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import pandas as pd
from gradio_client import Client
from src.session_store import CustomerSessionStore
from src.demo_catalog import load_demo_catalog
from src.presentation import localize_product_name

ROOT = Path(__file__).resolve().parent
DEMO_CATALOG = load_demo_catalog(ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx",
                                 ROOT / "catalog_preview_review/preview_assets")


def image_component_labels(value):
    """Collect only customer-visible Gradio image labels, not internal props."""
    labels = []
    if isinstance(value, dict):
        if value.get("component") == "image":
            label = (value.get("props") or {}).get("label")
            if label:
                labels.append(str(label))
        for child in value.values():
            labels.extend(image_component_labels(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            labels.extend(image_component_labels(child))
    return labels


def launch(db: Path, log_path: Path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    env = os.environ.copy()
    env["KAYAN_SESSION_DB"] = str(db)
    env["KAYAN_DAY_REPORT_DIR"] = str(db.parent / "reports")
    env["KAYAN_EXPORT_DIR"] = str(db.parent / "outputs")
    env["KAYAN_PORT"] = str(port)
    log = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            log.close()
            raise RuntimeError(f"Gradio exited early; log: {log_path.read_text(encoding='utf-8', errors='replace')[-5000:]}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return proc, log, url
        except Exception:
            time.sleep(1)
    proc.terminate(); log.close()
    raise TimeoutError(f"Gradio did not start; log: {log_path.read_text(encoding='utf-8', errors='replace')[-5000:]}")


def stop(proc, log):
    # Gradio may spawn a child Python server process on Windows. Stop the
    # entire tree before TemporaryDirectory removes the SQLite database.
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait(timeout=5)
    log.close()


def main():
    # Gradio can release SQLite file handles asynchronously after process
    # teardown. Ignore Windows cleanup races for the disposable test folder.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        temp = Path(directory)
        db, log_path = temp / "sessions.sqlite3", temp / "gradio.log"
        proc, log, url = launch(db, log_path)
        try:
            client = Client(url[:-1])
            config = client.config
            labels = [component.get("props", {}).get("label", "") for component in config["components"]]
            assert not any(label in labels for label in (
                "Exact matching demo variant", "Add selected product to active customer cart", "Cart activity"))
            catalog_frames = [component for component in config["components"]
                              if component.get("props", {}).get("label") is None
                              and component.get("type") == "dataframe"]
            assert catalog_frames
            assert all("Image" not in component.get("props", {}).get("headers", [])
                       and "Provenance" not in component.get("props", {}).get("headers", [])
                       for component in catalog_frames)
            first = client.predict("Day 1 customer A", api_name="/new_customer_cart")
            first_customer = first[0]["value"]
            assert first[2] == []
            # Reproduce the state-leak report in the live Gradio workflow:
            # focus and add a shoe variant before launching unrelated lists.
            client.predict("هل عندكم جزم", api_name="/chat")
            shoe_variant = next(v for v in DEMO_CATALOG.variants
                                if v.category.lower() == "shoes" and v.color.lower() == "white"
                                and v.size == "39" and v.in_stock)
            shoe_search = client.predict("عايز كوتشي أبيض مقاس 39", api_name="/chat")
            shoe_state = CustomerSessionStore(db).conversation(first_customer)[1]["demo_conversation"]
            assert shoe_variant.variant_id in shoe_state["displayed_variant_ids"], shoe_state
            assert "كوتشي" in repr(shoe_search[0]) or "سنيكرز" in repr(shoe_search[0])
            client.predict("ضيفه للسلة مقاس 39", api_name="/chat")
            shoe_outfit = CustomerSessionStore(db).list_outfits(first_customer)[0]
            assert any(item["variant_id"] == shoe_variant.variant_id for item in
                       CustomerSessionStore(db).outfit_record(shoe_outfit["outfit_id"])["items"])

            cheapest = client.predict(
                "اديني جدول لأرخص خمس حاجات عندكم بصورهم وأسعارهم ومقاساتهم المتاحة", api_name="/chat")
            cheapest_state = CustomerSessionStore(db).conversation(first_customer)[1]["demo_conversation"]
            cheapest_ids = cheapest_state["displayed_variant_ids"]
            cheapest_variants = [DEMO_CATALOG.get(variant_id) for variant_id in cheapest_ids]
            assert len(cheapest_state["displayed_product_ids"]) == 5
            assert len(set(cheapest_state["displayed_product_ids"])) == 5
            assert all(v and v.in_stock for v in cheapest_variants)
            assert [v.price_piasters for v in cheapest_variants] == sorted(v.price_piasters for v in cheapest_variants)
            assert cheapest_state["active"].get("sort_by") == "price_asc"
            assert not {"max_price_egp", "size", "categories"}.intersection(cheapest_state["active"])
            assert "جنيه" in repr(cheapest[0]) and "المقاسات" in repr(cheapest[0])

            chat = client.predict("هاتلي خمس تيشيرتات بيضا بصورهم وأسعارهم ومقاساتهم", api_name="/chat")
            assert len(chat[0]) == len(cheapest[0]) + 2, "the independent white-list request must append one turn pair"
            first_cards = repr(chat[0])
            assert "component': 'html'" in first_cards and "تيشيرت" in first_cards, chat[0]
            first_card_record = CustomerSessionStore(db).conversation(first_customer)[1]["demo_conversation"]
            first_history = CustomerSessionStore(db).conversation(first_customer)[0]
            displayed_reply = first_history[-1]["content"]
            # Gradio's serialized component config may contain legitimate
            # Python None props; validate the actual customer reply separately.
            assert "undefined" not in first_cards.lower()
            assert "undefined" not in displayed_reply.lower() and "None" not in displayed_reply
            assert len(first_card_record["displayed_product_ids"]) == 5, first_card_record
            white_variants = [DEMO_CATALOG.get(variant_id) for variant_id in first_card_record["displayed_variant_ids"]]
            assert all(v and v.in_stock and "white" in v.color.lower() for v in white_variants), white_variants
            assert all(localize_product_name(v.name, True) in displayed_reply for v in white_variants)
            assert shoe_variant.variant_id not in first_card_record["displayed_variant_ids"]
            assert first_card_record["display_fields"] == ["name", "image", "price", "sizes", "color"]
            # A plural detail follow-up updates the card fields while preserving
            # the exact ordered product list and omitting the unrequested image.
            price_chat = client.predict("طب أسعارهم؟", api_name="/chat")
            price_payload = repr(price_chat[0])
            assert "component': 'html'" in price_payload and "جنيه" in price_payload
            price_history, _ = CustomerSessionStore(db).conversation(first_customer)
            assert price_history[-1]["_cards"]["variant_ids"] == first_card_record["displayed_variant_ids"]
            assert price_history[-1]["_cards"]["fields"] == ["name", "price"]
            assert "<img" not in repr(price_chat[0][-1]["content"][-1])
            image_chat = client.predict("وريني صورته", api_name="/chat")
            image_payload = repr(image_chat[0])
            # Image requests use the same compact, responsive card renderer as
            # ordinary recommendations, never a standalone Image component.
            assert "component': 'html'" in image_payload and "component': 'image'" not in image_payload, image_chat[0]
            focused_variant_id = first_card_record["displayed_variant_ids"][0]
            focused_variant = DEMO_CATALOG.get(focused_variant_id)
            assert focused_variant and focused_variant.image_path
            assert localize_product_name(focused_variant.name, True) in image_payload, image_chat[0]
            image_history, _ = CustomerSessionStore(db).conversation(first_customer)
            image_record = image_history[-1]["_images"][0]
            assert image_record["variant_id"] == focused_variant_id
            assert Path(image_record["path"]) == focused_variant.image_path
            assert "Illustrative DEMO image" in image_payload or "صورة توضيحية" in image_payload, image_payload
            assert focused_variant.product_id not in image_payload and focused_variant.variant_id not in image_payload
            assert str(focused_variant.image_path) not in image_payload
            assert localize_product_name(focused_variant.name, True) in image_history[-1]["content"]
            assert "undefined" not in image_payload.lower()
            # The asset served by the app is an existing workbook-linked image.
            image_response = client.config
            assert any("Catalog product images" in str(c.get("props", {}).get("label", "")) for c in image_response["components"])
            # Product Search now returns only classifier-backed results; cart
            # mutations are kept in Sales Conversation rather than this tab.
            search = client.predict("blue t-shirt", True, 10, api_name="/demo_search")
            assert "XLM-R checkpoint:" in search[0] and "models\\best.pt" in search[0], search[0]
            assert "Candidate pool: 24 distinct Product IDs" in search[0]
            assert "query/product text pair" in search[0]
            assert search[1] and search[2]["data"] and search[3]["data"], search
            columns = search[2]["headers"]
            assert "Predicted ESCI" in columns and "Relevance score" in columns and "Catalog facet match" in columns, columns
            assert "Product ID" in columns and "Available sizes" in columns and "Colors" in columns, columns
            rows = [dict(zip(columns, row)) for row in search[2]["data"]]
            assert len(rows) == 10 and len({row["Product ID"] for row in rows}) == 10, rows
            # The explicit blue T-shirt request is verified against exact
            # workbook variants independently of the model's predicted class.
            blue_tshirt = next(row for row in rows if row["Product ID"] == "PVP-020")
            assert blue_tshirt["Catalog facet match"] == "In-stock facet match"
            source = pd.read_excel(ROOT / "catalog_preview_review/outputs/KAYAN_catalog_color_variants_NEW.xlsx", sheet_name="Preview variants")
            for row in rows:
                source_rows = source[(source["Product ID"] == row["Product ID"]) & (pd.to_numeric(source["Stock qty"]) > 0)]
                assert row["Demo stock qty"] == int(pd.to_numeric(source_rows["Stock qty"]).sum())
                assert set(row["Available sizes"].split(", ")) == set(source_rows["Size"].astype(str).str.strip())
                assert set(row["Colors"].split(", ")) == set(source_rows["Color"].astype(str).str.strip())
                prices=sorted(set(pd.to_numeric(source_rows["Price"]).astype(float)))
                expected_price=(f"{prices[0]:,.0f} EGP" if len(prices)==1 else f"{prices[0]:,.0f} EGP – {prices[-1]:,.0f} EGP")
                assert row["Price (EGP)"] == expected_price, (row, expected_price)
            # The actual logits/probabilities come from the configured checkpoint;
            # its saved metadata establishes E/S/C/I indices, not UI guesses.
            detail_columns=search[3]["headers"]
            detail_rows=[dict(zip(detail_columns,row)) for row in search[3]["data"]]
            assert all(row["Predicted ESCI"] in set("ESCI") for row in rows), rows
            details_by_id={row["Product ID"]:row for row in detail_rows}
            for row in rows:
                detail=details_by_id[row["Product ID"]]
                assert row["Predicted ESCI"] == max("ESCI",key=lambda label:detail[f"Logit {label}"])
                assert abs(row["Relevance score"]-(detail["P(E)"]+.2*detail["P(S)"])) < 1e-5
            arabic=client.predict("هل عندكم تيشيرتات؟",True,10,api_name="/demo_search")
            assert "Candidate pool: 24 distinct Product IDs" in arabic[0] and len(arabic[2]["data"])==10
            assert arabic[2]["headers"] and arabic[2]["headers"][7] == "مخزون تجريبي"
            top5=client.predict("هل عندكم تيشيرتات؟",True,5,api_name="/demo_search")
            top5_rows=[dict(zip(top5[2]["headers"],row)) for row in top5[2]["data"]]
            arabic_rows=[dict(zip(arabic[2]["headers"],row)) for row in arabic[2]["data"]]
            assert [row["Product ID"] for row in top5_rows] == [row["Product ID"] for row in arabic_rows[:5]]
            assert all(row["Predicted ESCI"] in set("ESCI") for row in rows)
            # Even a query with no lexical overlap still reaches XLM-R; it is a
            # relevance demonstration, not a strict catalog filter.
            weak=client.predict("unrelated dental floss",True,5,api_name="/demo_search")
            assert "Candidate pool: 24 distinct Product IDs" in weak[0] and weak[2]["data"], weak
            weak_rows=[dict(zip(weak[2]["headers"],row)) for row in weak[2]["data"]]
            assert all(row["Predicted ESCI"] in set("ESCI") for row in weak_rows)
            assert "I" in {row["Predicted ESCI"] for row in weak_rows}, weak_rows
            # Ask the Sales Conversation to add the exact selected catalog SKU.
            cart = client.predict("ضيفه للسلة مقاس M", api_name="/chat")
            assert cart[5] != "0 EGP" and cart[5] != "0 جنيه", cart
            assert cart[3]["data"], cart
            with sqlite3.connect(db) as conn:
                assert conn.execute("SELECT COUNT(*) FROM demo_mock_purchase_requests").fetchone()[0] == 0
            client.predict(api_name="/prepare_buy")
            with sqlite3.connect(db) as conn:
                assert conn.execute("SELECT COUNT(*) FROM demo_mock_purchase_requests").fetchone()[0] == 0
            purchase = client.predict(True, api_name="/confirm_buy")
            assert "MOCK" in str(purchase[0]) or "MOCK" in str(purchase[2])
            second = client.predict("Day 1 customer B", api_name="/new_customer_cart")
            assert second[2] == []
            second_customer = second[0]["value"]
            # Exercise the real Gradio remove callback. The active session and
            # dropdown target are separate callback values; removal is keyed by
            # the selected dropdown ID, not by a stale session snapshot.
            third = client.predict("Archive target C", api_name="/new_customer_cart")
            third_customer = third[0]["value"]
            store_before_remove = CustomerSessionStore(db)
            assert store_before_remove.is_current_day_customer(third_customer)
            cancelled = client.predict(
                third_customer, "stale label", False,
                api_name="/remove_customer_selected",
            )
            assert not store_before_remove.customer(third_customer)["removed"]
            removed = client.predict(
                third_customer, "stale label", True,
                api_name="/remove_customer_selected",
            )
            store_after_remove = CustomerSessionStore(db)
            assert store_after_remove.customer(third_customer)["removed"] == 1
            assert not store_after_remove.is_current_day_customer(third_customer)
            assert third_customer not in [value for _label, value in removed[0]["choices"]]
            assert "archived" in str(removed[9]).lower()
            assert store_after_remove.conversation(first_customer)[0], "unrelated customer's history must remain intact"
            assert store_after_remove.customer(second_customer)["removed"] == 0
            ended = client.predict(True, api_name="/end_day")
            summary = ended[0]
            assert summary["day_number"] == 1 and summary["status"] == "closed"
            assert summary["customer_sessions"]["customers_served"] == 3
            assert summary["demo_mock_purchases"]["confirmed_requests"] == 1
            assert summary["sales"]["revenue_piasters"] == 0
            started = client.predict(True, api_name="/start_new_day")
            assert started[0]["day_number"] == 2 and started[0]["status"] == "active"
            assert started[7].get("choices") == [] and started[8] == []
            repeated = client.predict(True, api_name="/start_new_day")
            assert "still active" in repeated[5]
            assert repeated[0]["day_number"] == 2
        finally:
            stop(proc, log)

        store = CustomerSessionStore(db)
        days = store.list_business_days()
        assert len(days) == 2 and days[0]["day_number"] == 2 and days[0]["status"] == "active"
        closed = store.business_day_snapshot(days[1]["day_id"])
        assert closed["demo_mock_purchases"]["confirmed_requests"] == 1
        first_history, _ = store.conversation(first_customer)
        assert len(first_history) == 17 and [m["role"] for m in first_history] == ["user", "assistant"] * 8 + ["assistant"], (
            f"first customer history length={len(first_history)}; report counts={closed.get('conversation_message_counts')}")
        assert store.list_customers() == []

        proc, log, url = launch(db, log_path)
        try:
            client = Client(url[:-1])
            restored = client.predict(api_name="/restore_active_customer")
            assert restored[0].get("choices") == []
            assert restored[1] == []
            assert "No customer in this business day" in restored[-1]
            historical = client.predict(days[1]["day_id"], api_name="/historical_day_change")
            assert historical["day_number"] == 1 and historical["status"] == "closed"
            assert CustomerSessionStore(db).active_business_day()["day_id"] == days[0]["day_id"]
            repeated = client.predict(True, api_name="/start_new_day")
            assert repeated[0]["day_number"] == 2 and len(CustomerSessionStore(db).list_business_days()) == 2
        finally:
            stop(proc, log)
    print("PASS: live Gradio Day 1 chat and MOCK purchase, confirmed close, fresh Day 2, no duplicate start, and persisted restart")


if __name__ == "__main__":
    main()
