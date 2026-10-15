"""Transfer-matrix heat-map from results/attacks_*.json (real rows only).
python scripts/plot_transfer_matrix.py --attack pgd --eps 8 [--metric rank1|mAP|impersonation_hit@1] [--input results/attacks_impersonate.json]"""
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
    ap.add_argument("--attack", default="pgd")
    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--metric", default="rank1", choices=["rank1", "mAP", "impersonation_hit@1"])
    a = ap.parse_args()
    src = a.input or rpath("attacks_evade.json")
    rows = load_json(src, "Run scripts/run_attacks.py first.")
    sel = [r for r in rows if r["attack"] == a.attack and r["eps"] == a.eps and a.metric in r]
    if not sel:
        have = sorted({(r["attack"], r["eps"]) for r in rows if r["attack"] != "clean"})
        raise SystemExit(f"[ERROR] no rows for attack={a.attack} eps={a.eps} metric={a.metric} in {src}. Available: {have}")
    srcs = order_models({r["source"] for r in sel}); tgts = order_models({r["target"] for r in sel})
    M = np.full((len(srcs), len(tgts)), np.nan)
    for r in sel:
        M[srcs.index(r["source"]), tgts.index(r["target"])] = r[a.metric]
    clean = {r["target"]: r[a.metric] for r in rows if r["attack"] == "clean" and a.metric in r}
    lower_is_stronger = a.metric != "impersonation_hit@1"
    fig, ax = plt.subplots(figsize=(1.9 * len(tgts) + 3, 1.3 * len(srcs) + 2.8))
    im = ax.imshow(M, cmap="YlOrRd_r" if lower_is_stronger else "YlOrRd", vmin=0, vmax=max(100 if a.metric != "impersonation_hit@1" else 50, np.nanmax(M)))
    plt.colorbar(im, label=f"{a.metric} (%)")
    ax.set_xticks(range(len(tgts))); ax.set_yticks(range(len(srcs)))
    ax.set_xticklabels([f"{t}\n(clean {clean[t]:.1f})" if t in clean else t for t in tgts], fontsize=9)
    ax.set_yticklabels(srcs, fontsize=9)
    ax.set_xlabel("Target (victim) model"); ax.set_ylabel("Source (attacker's surrogate) model")
    for i in range(len(srcs)):
        for j in range(len(tgts)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", color="grey")
            else:
                dark = (M[i, j] < 35) if lower_is_stronger else (M[i, j] > 25)
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontweight="bold", color="white" if dark else "black")
    ax.set_title(f"{a.attack.upper()} eps={a.eps}/255 - {a.metric}\n(diagonal = white-box, off-diagonal = black-box transfer; real measurements)", fontsize=10)
    plt.tight_layout()
    out = a.out or rpath(f"transfer_matrix_{a.attack}_eps{a.eps}_{a.metric.replace('@', '')}.png")
    plt.savefig(out, dpi=200); plt.close()
    stamp(f"figure:{os.path.basename(out)}", sources=[src], extra={"attack": a.attack, "eps": a.eps, "metric": a.metric})
    print("saved", out)


if __name__ == "__main__":
    main()
