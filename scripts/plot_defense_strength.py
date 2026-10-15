"""Is the defended model's robustness real? White-box Rank-1 of the adversarially-trained ResNet-50 vs attack strength
(PGD-10 vs PGD-50, MI-FGSM-10 vs 50) and vs epsilon. Steps are read from MANIFEST.json (never typed in). Real rows only.
python scripts/plot_defense_strength.py"""
import argparse, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.results import load_json, rpath, stamp

ADV = "resnet50_adv_trained_eps8"


def steps_of(manifest, fname):
    e = manifest.get(f"run_attacks:{fname}")
    if not e or "steps" not in e:
        raise SystemExit(f"[ERROR] MANIFEST.json has no 'run_attacks:{fname}' entry with 'steps' - cannot verify how {fname} was produced.")
    return e["steps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    man = load_json(rpath("MANIFEST.json"), "Run the experiments first.")
    files = ["defense.json", "defense_pgd50.json"]
    bars = []
    for f in files:
        rows = load_json(rpath(f), "Run the Day-4 runner (PGD-50 defence attack) first." if "pgd50" in f else "")
        st = steps_of(man, f)
        for atk in ("pgd", "mifgsm"):
            r = [x for x in rows if x["attack"] == atk and x["eps"] == 8 and x["source"] == ADV and x["target"] == ADV]
            if r:
                bars.append((f"{atk.upper()}-{st}", r[0]["rank1"]))
    if len(bars) < 3:
        raise SystemExit(f"[ERROR] expected white-box rows on {ADV}; found {bars}")
    base = load_json(rpath("defense.json"))
    clean = next(r["rank1"] for r in base if r["attack"] == "clean" and r["target"] == ADV)
    sweep_p = rpath("defense_eps_sweep.json")
    has_sweep = os.path.exists(sweep_p)
    fig, axes = plt.subplots(1, 2 if has_sweep else 1, figsize=(12 if has_sweep else 6.5, 4.6), squeeze=False)
    ax = axes[0, 0]
    b = ax.bar([x[0] for x in bars], [x[1] for x in bars], color="#4daf4a")
    for rect, x in zip(b, bars):
        ax.text(rect.get_x() + rect.get_width() / 2, x[1] + 1, f"{x[1]:.1f}", ha="center")
    ax.axhline(clean, color="black", ls="--", lw=1); ax.text(-0.4, clean + 1, f"clean {clean:.1f}", fontsize=8)
    ax.set_ylim(0, 100); ax.set_ylabel("Rank-1 (%)"); ax.grid(axis="y", ls="--", alpha=0.4)
    ax.set_title("Adversarially-trained ResNet-50, white-box, eps=8/255\n(trained with PGD-5; attack steps read from MANIFEST)", fontsize=10)
    if has_sweep:
        sw = json.load(open(sweep_p))
        pts = sorted({(r["eps"], r["rank1"]) for r in sw + base if r["attack"] == "pgd" and r["source"] == ADV and r["target"] == ADV})
        std = sorted({(r["eps"], r["rank1"]) for r in sw + base if r["attack"] == "pgd" and r["source"] == "resnet50" and r["target"] == "resnet50"})
        ax2 = axes[0, 1]
        ax2.plot([0] + [p[0] for p in pts], [clean] + [p[1] for p in pts], "o-", label="adversarially trained", color="#4daf4a")
        if std:
            c0 = next(r["rank1"] for r in base if r["attack"] == "clean" and r["target"] == "resnet50")
            ax2.plot([0] + [p[0] for p in std], [c0] + [p[1] for p in std], "s--", label="standard", color="#e34a33")
        ax2.set_xlabel("epsilon (/255)"); ax2.set_ylabel("white-box PGD-10 Rank-1 (%)"); ax2.set_ylim(0, 100)
        ax2.grid(True, ls="--", alpha=0.4); ax2.legend()
        ax2.set_title("Robust Rank-1 should fall as epsilon grows\n(a flat or rising curve would hint at gradient masking)", fontsize=10)
    plt.tight_layout()
    out = a.out or rpath("defense_strength.png")
    plt.savefig(out, dpi=200); plt.close()
    stamp(f"figure:{os.path.basename(out)}", sources=[rpath(f) for f in files])
    print("saved", out, bars)


if __name__ == "__main__":
    main()
