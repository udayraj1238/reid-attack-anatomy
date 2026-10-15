"""Bar chart of patch_results.json: clean vs random patch vs trained patch (fixed and EOT-randomised), per model."""
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
    src = a.input or rpath("patch_results.json")
    rows = load_json(src, "Run scripts/run_patch.py first.")
    models = order_models({r["model"] for r in rows})
    conds = ["clean", "random_patch", "trained_patch", "trained_patch_eot"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    w = 0.2
    for ax, metric in zip(axes, ("rank1", "mAP")):
        for k, c in enumerate(conds):
            vals = [next((r[metric] for r in rows if r["model"] == m and r["condition"] == c), np.nan) for m in models]
            bars = ax.bar(np.arange(len(models)) + k * w, vals, w, label=c)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=7)
        labels = [m + (" (source)" if any(r["model"] == m and r["is_source"] for r in rows) else " (transfer)") for m in models]
        ax.set_xticks(np.arange(len(models)) + 1.5 * w); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(f"{metric} (%)"); ax.set_ylim(0, 105); ax.grid(axis="y", ls="--", alpha=0.4)
    axes[0].legend(fontsize=8)
    fig.suptitle("Universal torso patch (EOT): effect on retrieval, with random-patch control (real measurements)")
    plt.tight_layout()
    out = a.out or rpath("patch_effect.png")
    plt.savefig(out, dpi=200); plt.close()
    stamp(f"figure:{os.path.basename(out)}", sources=[src])
    print("saved", out)


if __name__ == "__main__":
    main()
