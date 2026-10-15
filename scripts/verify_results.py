"""Integrity check. Run before every commit / before the presentation.  python scripts/verify_results.py [--strict]
 1. required result files exist     2. no mock/hard-coded-data patterns in code     3. every figure has provenance
 4. baselines.json agrees with the clean rows of attacks_evade.json     5. generated docs have no unfilled {{placeholders}}"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.results import rpath, results_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED = ["baselines.json", "attacks_evade.json", "attacks_impersonate.json", "patch_results.json", "defense.json", "MANIFEST.json"]
FORBIDDEN = [r"\bmock", r"hard-?coded", r"synthetic", r"np\.random\.(randn|uniform|rand)\("]


def main():
    strict = "--strict" in sys.argv
    problems, warns = [], []
    for f in REQUIRED:
        if not os.path.exists(rpath(f)):
            (problems if strict else warns).append(f"missing result file: {rpath(f)}")
    for py in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
        rel = os.path.relpath(py, ROOT)
        if rel.startswith("tests") or rel.endswith("verify_results.py") or rel.startswith("."):
            continue
        for i, line in enumerate(open(py, encoding="utf-8"), 1):
            if any(re.search(p, line, re.I) for p in FORBIDDEN):
                problems.append(f"{rel}:{i}: suspicious pattern: {line.strip()[:90]}")
    man = json.load(open(rpath("MANIFEST.json"))) if os.path.exists(rpath("MANIFEST.json")) else {}
    for png in glob.glob(os.path.join(results_dir(), "*.png")):
        name = os.path.basename(png)
        if name in ("patch.png", "patch_examples.png"):
            if "run_patch" not in man:
                problems.append(f"{name} has no 'run_patch' entry in MANIFEST.json")
        elif f"figure:{name}" not in man:
            problems.append(f"figure {name} has no provenance entry in MANIFEST.json (was it made by a real script run?)")
    if os.path.exists(rpath("baselines.json")) and os.path.exists(rpath("attacks_evade.json")):
        b = {r["model"]: r for r in json.load(open(rpath("baselines.json")))}
        for r in json.load(open(rpath("attacks_evade.json"))):
            if r["attack"] == "clean" and r["target"] in b and abs(r["rank1"] - b[r["target"]]["rank1"]) > 0.05:
                problems.append(f"clean Rank-1 mismatch for {r['target']}: baselines.json {b[r['target']]['rank1']} vs attacks_evade.json {r['rank1']} (different query subset?)")
    for md in glob.glob(os.path.join(ROOT, "docs", "*.md")):
        if "{{" in open(md, encoding="utf-8").read() and not md.endswith(".template.md"):
            problems.append(f"{os.path.relpath(md, ROOT)} still contains unfilled {{{{placeholders}}}}")
    for w in warns:
        print("WARN ", w)
    for p in problems:
        print("FAIL ", p)
    print("\nRESULT:", "OK - everything traces back to real runs" if not problems else f"{len(problems)} problem(s) found")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
