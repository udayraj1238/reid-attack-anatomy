"""results/baselines.json -> results/baseline_table.md (real numbers only)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.results import load_json, rpath, stamp, order_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = a.input or rpath("baselines.json")
    rows = load_json(src, "Run scripts/eval_baselines.py first.")
    rows = sorted(rows, key=lambda r: (order_models([r["model"]]).index(r["model"]), r["model"]))
    md = ["# Clean baselines on Market-1501 (3,368 queries, 15,913 gallery images)\n",
          "| Model | Params (M) | Emb. dim | Rank-1 | Rank-5 | Rank-10 | mAP |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['params_M']} | {r['emb_dim']} | {r['rank1']:.2f} | {r['rank5']:.2f} | {r['rank10']:.2f} | {r['mAP']:.2f} |")
    out = a.out or rpath("baseline_table.md")
    open(out, "w").write("\n".join(md) + "\n")
    stamp(f"figure:{os.path.basename(out)}", sources=[src])
    print("\n".join(md))


if __name__ == "__main__":
    main()
