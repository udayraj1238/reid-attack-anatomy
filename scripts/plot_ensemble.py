"""Held-out transfer: single surrogates vs leave-one-out ensemble (Rank-1, lower = stronger attack). Real rows only.
python scripts/plot_ensemble.py --attack pgd --eps 8"""
import argparse, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.results import load_json, rpath, stamp, order_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="pgd")
    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--single", default=None, help="default results/attacks_evade.json")
    ap.add_argument("--ens", default=None, help="default results/ensemble.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sp, ep = a.single or rpath("attacks_evade.json"), a.ens or rpath("ensemble.json")
    single = load_json(sp, "Run scripts/run_attacks.py first.")
    ens = load_json(ep, "Run scripts/run_ensemble_attack.py first.")
    held = [r for r in ens if r["attack"] == a.attack and r["eps"] == a.eps and r["role"] == "held_out"]
    if not held:
        raise SystemExit(f"[ERROR] no held-out ensemble rows for {a.attack} eps={a.eps} in {ep}")
    clean = {r["target"]: r["rank1"] for r in ens if r["attack"] == "clean"}
    groups = []
    for h in order_models({r["target"] for r in held}):
        er = next(r for r in held if r["target"] == h)
        bars = []
        for s in er["members"]:
            v = [r["rank1"] for r in single if r["attack"] == a.attack and r["eps"] == a.eps and r["source"] == s and r["target"] == h]
            if v:
                bars.append((f"single: {s}", v[0], "#8da0cb"))
        bars.append((f"ensemble: {'+'.join(er['members'])}", er["rank1"], "#e34a33"))
        groups.append((h, bars))
    fig, ax = plt.subplots(figsize=(4.2 * len(groups) + 1, 5))
    x0, ticks, labels = 0, [], []
    for h, bars in groups:
        for i, (lab, v, col) in enumerate(bars):
            ax.bar(x0 + i, v, 0.8, color=col)
            ax.text(x0 + i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
            ax.text(x0 + i, 2 if v >= 25 else v + 7, lab.replace(": ", ":\n").replace("+", "+\n"), ha="center", va="bottom", fontsize=6.5, rotation=90)
        if h in clean:
            ax.hlines(clean[h], x0 - 0.5, x0 + len(bars) - 0.5, colors="black", linestyles="dashed", linewidth=1)
        ticks.append(x0 + (len(bars) - 1) / 2)
        labels.append(f"held-out victim:\n{h}" + (f"\n(clean {clean[h]:.1f})" if h in clean else ""))
        x0 += len(bars) + 1
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("victim Rank-1 (%) - lower = stronger transfer"); ax.set_ylim(0, 105); ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_title(f"{a.attack.upper()} eps={a.eps}/255: single surrogate vs ensemble of the other models\n(dashed = clean Rank-1; real measurements, same budget and steps)", fontsize=10)
    plt.tight_layout()
    out = a.out or rpath(f"ensemble_transfer_{a.attack}_eps{a.eps}.png")
    plt.savefig(out, dpi=200); plt.close()
    stamp(f"figure:{os.path.basename(out)}", sources=[sp, ep], extra={"attack": a.attack, "eps": a.eps})
    print("saved", out)
    for h, bars in groups:
        print(h, {l: v for l, v, _ in bars})


if __name__ == "__main__":
    main()
