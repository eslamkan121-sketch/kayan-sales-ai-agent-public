import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
results = []
for path in sorted(root.glob("*_test.py")):
    print(f"=== {path.name} ===", flush=True)
    try:
        completed = subprocess.run([sys.executable, str(path)], cwd=root, text=True,
                                   encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=300)
        output = completed.stdout
        code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\nTIMEOUT after 300 seconds"
        code = 124
    # Windows consoles may use cp1252; escape Arabic in logs rather than
    # terminating the suite when the child output contains Unicode.
    safe_tail = "\n".join(output.splitlines()[-4:]).encode("ascii", "backslashreplace").decode("ascii")
    print(safe_tail, flush=True)
    print(f"EXIT={code}", flush=True)
    results.append({"test": path.name, "exit_code": code, "output_tail": output.splitlines()[-20:]})
    (root / "outputs" / "full_test_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
failed = [result for result in results if result["exit_code"]]
print("=== SUMMARY ===", flush=True)
print(f"Ran {len(results)} test files; passed {len(results)-len(failed)}; failed {len(failed)}", flush=True)
for result in failed:
    print(f"FAIL {result['test']}", flush=True)
sys.exit(bool(failed))
