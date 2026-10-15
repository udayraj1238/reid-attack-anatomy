"""Standard vs adversarially-trained model under attack (real rows from results/defense.json, produced by run_attacks.py)."""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.results import load_json, rpath, stamp, order_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = a.input or rpath("defense.json")
    rows = load_json(src, "Run: run_attacks.py --ckpts checkpoints/resnet50.pth checkpoints/resnet50_adv_trained_eps8.pth --attacks pgd mifgsm --eps 8 --out results/defense.json")
    models = order_models({r["target"] for r in rows})
    def cond(r):
        if r["attack"] == "clean":
            return "clean"
        return f"white-box {r['attack'].upper()}" if r["source"] == r["target"] else f"transfer {r['attack'].upper()} from {r['source']}"
    conds = []
    for r in rows:
        c = cond(r)
        if c not in conds:
            conds.append(c)
    conds.sort(key=lambda c: (c != "clean", not c.startswith("white-box"), c))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    w = 0.8 / len(conds)
    for ax, metric in zip(axes, ("rank1", "mAP")):
        for k, c in enumerate(conds):
            vals = []
            for m in models:
                v = [r[metric] for r in rows if r["target"] == m and cond(r) == c]
                vals.append(v[0] if v else np.nan)
            bars = ax.bar(np.arange(len(models)) + k * w, vals, w, label=c)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_xticks(np.arange(len(models)) + 0.4 - w / 2); ax.set_xticklabels(models, fontsize=9)
        ax.set_ylabel(f"{metric} (%)"); ax.set_ylim(0, 105); ax.grid(axis="y", ls="--", alpha=0.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Defence trade-off: clean accuracy vs robustness (eps=8/255, real measurements)")
    plt.tight_layout(rect=(0, 0.1, 1, 0.95))
    out = a.out or rpath("defense_tradeoff.png")
    plt.savefig(out, dpi=200); plt.close()
    stamp(f"figure:{os.path.basename(out)}", sources=[src])
    print("saved", out)


if __name__ == "__main__":
    main()
