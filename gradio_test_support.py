"""Start a disposable Gradio app instance for integration tests."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import hashlib
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from gradio_client import Client


ROOT = Path(__file__).resolve().parent


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in [ROOT / "app.py", *sorted((ROOT / "src").glob("*.py"))]:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@contextmanager
def isolated_gradio_server():
    """Yield (client, database path) without touching the operator's app DB."""
    with tempfile.TemporaryDirectory(prefix="kayan-gradio-test-", ignore_cleanup_errors=True) as directory:
        temp = Path(directory)
        database = temp / "sessions.sqlite3"
        log_path = temp / "gradio.log"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        env = os.environ.copy()
        env["KAYAN_SESSION_DB"] = str(database)
        env["KAYAN_DAY_REPORT_DIR"] = str(temp / "reports")
        env["KAYAN_EXPORT_DIR"] = str(temp / "outputs")
        env["KAYAN_PORT"] = str(port)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            url = f"http://127.0.0.1:{port}/"
            try:
                deadline = time.time() + 180
                while time.time() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Gradio exited early:\n" + log_path.read_text(encoding="utf-8", errors="replace")[-5000:])
                    try:
                        with urllib.request.urlopen(url, timeout=2) as response:
                            if response.status == 200:
                                break
                    except Exception:
                        time.sleep(1)
                else:
                    raise TimeoutError("Gradio startup timed out:\n" + log_path.read_text(encoding="utf-8", errors="replace")[-5000:])
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                match = re.search(r"^KAYAN_BUILD (\{.*\})$", log_text, re.M)
                if not match:
                    raise RuntimeError("Isolated server did not emit its build identity:\n" + log_text[:2000])
                identity = json.loads(match.group(1))
                if identity.get("pid") != process.pid:
                    raise RuntimeError(f"Build identity PID {identity.get('pid')} != launched PID {process.pid}")
                if Path(identity.get("project_path", "")).resolve() != ROOT.resolve():
                    raise RuntimeError(f"Server loaded an unexpected project path: {identity.get('project_path')}")
                if identity.get("source_fingerprint") != source_fingerprint():
                    raise RuntimeError(f"Server source fingerprint mismatch: {identity.get('source_fingerprint')} != {source_fingerprint()}")
                yield Client(url[:-1]), database
            finally:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

