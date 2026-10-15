"""White-box degradation vs epsilon for one model (real rows only). Needs the eps sweep to have been run.
python scripts/plot_eps_curve.py --model resnet50"""
import argparse, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.results import load_json, rpath, stamp

LABEL = {"fgsm": "FGSM", "pgd": "PGD-10", "mifgsm": "MI-FGSM", "logit_pgd": "Logit-PGD (classification-style baseline)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = a.input or rpath("attacks_evade.json")
    rows = load_json(src, "Run scripts/run_attacks.py (with --eps 2 4 8 16) first.")
    clean = [r for r in rows if r["attack"] == "clean" and r["target"] == a.model]
    if not clean:
        raise SystemExit(f"[ERROR] no clean row for model {a.model} in {src}")
    attacks = sorted({r["attack"] for r in rows if r["attack"] != "clean" and r["source"] == a.model and r["target"] == a.model})
    if not attacks:
        raise SystemExit(f"[ERROR] no white-box rows for {a.model} in {src}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for atk in attacks:
        pts = sorted((r["eps"], r["rank1"], r["mAP"]) for r in rows if r["attack"] == atk and r["source"] == a.model and r["target"] == a.model)
        e = [0] + [p[0] for p in pts]
        for ax, k, name in ((axes[0], 1, "Rank-1 (%)"), (axes[1], 2, "mAP (%)")):
            base = clean[0]["rank1" if k == 1 else "mAP"]
            ax.plot(e, [base] + [p[k] for p in pts], marker="o", linewidth=2, label=LABEL.get(atk, atk))
            ax.set_xlabel("epsilon (/255)"); ax.set_ylabel(name); ax.grid(True, ls="--", alpha=0.5); ax.set_ylim(0, 100)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"White-box evasion on {a.model}: accuracy vs perturbation budget (real measurements, {len(attacks)} attacks)")
    plt.tight_layout()
    out = a.out or rpath("eps_degradation_curve.png")
    plt.savefig(out, dpi=200); plt.close()
    stamp(f"figure:{os.path.basename(out)}", sources=[src], extra={"model": a.model})
    print("saved", out)


if __name__ == "__main__":
    main()
