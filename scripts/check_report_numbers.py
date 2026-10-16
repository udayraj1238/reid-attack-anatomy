"""Flag every decimal number in a hand-written document (report, slides, notes) that does not appear in results/*.json.
Usage:  python scripts/check_report_numbers.py PHASE1_REPORT.md [more.md ...]

Why: verify_results.py proves the JSON files are real, but it cannot know whether a hand-written report copied them correctly.
Flagged numbers are not automatically wrong (section numbers, differences like 86.97-74.52, constants), but each one needs a
human to confirm it is derived from a real value. Anything you cannot trace back should be deleted."""
import glob, json, os, re, sys


def collect(obj, acc):
    if isinstance(obj, dict):
        for v in obj.values():
            collect(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            collect(v, acc)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        acc.add(round(float(obj), 2))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    results = os.environ.get("REID_RESULTS", "results")
    known = set()
    for f in glob.glob(os.path.join(results, "*.json")):
        collect(json.load(open(f)), known)
    if not known:
        raise SystemExit(f"[ERROR] no numbers found in {results}/*.json")
    bad_total = 0
    paths = []
    for arg in sys.argv[1:]:
        matched = glob.glob(arg)
        paths.extend(matched if matched else [arg])
    for path in paths:
        if not os.path.isfile(path):
            continue
        text = open(path, encoding="utf8").read()
        flagged = []
        for ln, line in enumerate(text.splitlines(), 1):
            for m in re.finditer(r"(?<![\w.])(\d{1,3}\.\d{1,2})(?![\d])", line):
                if round(float(m.group(1)), 2) not in known:
                    flagged.append((ln, m.group(1), line.strip()[:110]))
        print(f"\n{path}: {len(flagged)} number(s) not found in {results}/*.json")
        for ln, n, ctx in flagged:
            print(f"  line {ln}: {n}   | {ctx}")
        bad_total += len(flagged)
    print("\nReview each flagged number: derived (difference/mean) = OK if you can show the arithmetic; otherwise remove it.")
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
