"""Build docs/RESULTS_SUMMARY.md + results/facts.json from REAL result files, and fill the doc templates.

  python scripts/build_docs.py            # summary + facts.json (sections whose experiment hasn't run say 'NOT RUN YET')
  python scripts/build_docs.py --fill     # also turn docs/*.template.md into docs/*.md; FAILS if any {{number}} has no real source

Nothing here contains a hand-typed result. Template placeholders look like {{base_resnet50_r1}} (see facts.json for all keys).
"""
import argparse, glob, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.results import rpath, order_models, results_dir

ATKS = ["fgsm", "pgd", "mifgsm", "logit_pgd"]
BASE = ["resnet50", "densenet121", "vit_small"]


def opt(name):
    p = rpath(name)
    return json.load(open(p)) if os.path.exists(p) else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true")
    a = ap.parse_args()
    facts, md = {}, [f"# Results summary (auto-generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})\n",
                     "Every number below is computed from `results/*.json` written by real runs. See `results/MANIFEST.json` for provenance.\n"]

    # ---- baselines
    base = opt("baselines.json")
    md.append("## 1. Clean baselines")
    if base:
        for r in base:
            for k in ("rank1", "rank5", "rank10", "mAP"):
                facts[f"base_{r['model']}_{'r1' if k == 'rank1' else 'r5' if k == 'rank5' else 'r10' if k == 'rank10' else 'map'}"] = r[k]
            facts[f"params_{r['model']}"] = r["params_M"]
        md.append(table(["Model", "Params (M)", "Rank-1", "Rank-5", "Rank-10", "mAP"],
                        [[r["model"], r["params_M"], r["rank1"], r["rank5"], r["rank10"], r["mAP"]] for r in base]))
    else:
        md.append("**NOT RUN YET** (scripts/eval_baselines.py)")

    # ---- evasion
    ev = opt("attacks_evade.json")
    md.append("\n## 2. Evasion attacks")
    if ev:
        bm = [m for m in BASE if any(r["target"] == m for r in ev)]
        for atk in ATKS:
            sel = [r for r in ev if r["attack"] == atk and r["eps"] == 8 and r["source"] in bm and r["target"] in bm]
            if not sel:
                continue
            md.append(f"\n### {atk.upper()} eps=8/255 - Rank-1 (%) (rows = source, cols = target; diagonal = white-box)")
            md.append(table(["source \\ target"] + bm, [[s] + [next((x["rank1"] for x in sel if x["source"] == s and x["target"] == t), "-") for t in bm] for s in bm if any(x["source"] == s for x in sel)]))
            wb = [r for r in sel if r["source"] == r["target"]]
            tr = [r for r in sel if r["source"] != r["target"]]
            for r in wb:
                facts[f"wb_{atk}_{r['source']}_r1"] = r["rank1"]; facts[f"wb_{atk}_{r['source']}_map"] = r["mAP"]
            for r in tr:
                facts[f"xfer_{atk}_{r['source']}_to_{r['target']}_r1"] = r["rank1"]
            if wb: facts[f"wb_{atk}_mean_r1"] = round(float(np.mean([r["rank1"] for r in wb])), 2)
            if tr: facts[f"xfer_{atk}_mean_r1"] = round(float(np.mean([r["rank1"] for r in tr])), 2)
        md.append("\n### White-box vs black-box summary (eps=8, Rank-1 %, lower = stronger attack)")
        md.append(table(["attack", "white-box mean", "black-box (transfer) mean"],
                        [[atk, facts.get(f"wb_{atk}_mean_r1", "-"), facts.get(f"xfer_{atk}_mean_r1", "-")] for atk in ATKS if f"wb_{atk}_mean_r1" in facts]))
        sweep = sorted({r["eps"] for r in ev if r["attack"] != "clean"})
        if len(sweep) > 1 and "resnet50" in bm:
            md.append("\n### Epsilon sweep, white-box on resnet50 (Rank-1 %)")
            md.append(table(["attack"] + [f"eps={e}" for e in sweep], [[atk] + [next((r["rank1"] for r in ev if r["attack"] == atk and r["eps"] == e and r["source"] == r["target"] == "resnet50"), "-") for e in sweep] for atk in ATKS]))
            for r in ev:
                if r["attack"] in ATKS and r["source"] == r["target"] == "resnet50":
                    facts[f"eps{r['eps']}_{r['attack']}_resnet50_r1"] = r["rank1"]
    else:
        md.append("**NOT RUN YET** (scripts/run_attacks.py --mode evade)")

    # ---- impersonation
    im = opt("attacks_impersonate.json")
    md.append("\n## 3. Impersonation attacks (higher hit@1 = stronger attack)")
    if im:
        sel = [r for r in im if r["attack"] != "clean" and r["eps"] == 8 and "impersonation_hit@1" in r]
        md.append(table(["attack", "source", "target", "hit@1 (%)", "Rank-1 of true id (%)"],
                        [[r["attack"], r["source"], r["target"], r["impersonation_hit@1"], r["rank1"]] for r in sel]))
        for atk in ("pgd", "mifgsm"):
            wb = [r["impersonation_hit@1"] for r in sel if r["attack"] == atk and r["source"] == r["target"]]
            tr = [r["impersonation_hit@1"] for r in sel if r["attack"] == atk and r["source"] != r["target"]]
            for r in sel:
                if r["attack"] == atk and r["source"] == r["target"]:
                    facts[f"imp_{atk}_wb_{r['source']}_hit"] = r["impersonation_hit@1"]
            if wb: facts[f"imp_{atk}_wb_mean_hit"] = round(float(np.mean(wb)), 2)
            if tr: facts[f"imp_{atk}_xfer_mean_hit"] = round(float(np.mean(tr)), 2)
    else:
        md.append("**NOT RUN YET** (scripts/run_attacks.py --mode impersonate)")

    # ---- patch
    pr = opt("patch_results.json")
    md.append("\n## 4. Universal torso patch (EOT)")
    if pr:
        md.append(table(["model", "role", "condition", "Rank-1", "mAP"],
                        [[r["model"], "source" if r["is_source"] else "transfer", r["condition"], r["rank1"], r["mAP"]] for r in pr]))
        for r in pr:
            facts[f"patch_{r['condition']}_{r['model']}_r1"] = r["rank1"]; facts[f"patch_{r['condition']}_{r['model']}_map"] = r["mAP"]
        pt = opt("patch_training.json")
        if pt:
            md.append(f"\nPatch training: {pt['epochs']} epochs, mean cosine-sim to clean per epoch: {pt['mean_cos_sim_per_epoch']}")
            facts["patch_final_cos_sim"] = pt["mean_cos_sim_per_epoch"][-1]
    else:
        md.append("**NOT RUN YET** (scripts/run_patch.py)")

    # ---- defence
    df = opt("defense.json")
    md.append("\n## 5. Adversarial-training defence (eps=8)")
    if df:
        md.append(table(["attack", "source", "target", "Rank-1", "mAP"],
                        [[r["attack"], r["source"], r["target"], r["rank1"], r["mAP"]] for r in df]))
        for r in df:
            if r["attack"] == "clean":
                facts[f"def_clean_{r['target']}_r1"] = r["rank1"]; facts[f"def_clean_{r['target']}_map"] = r["mAP"]
            elif r["source"] == r["target"]:
                facts[f"def_wb_{r['attack']}_{r['target']}_r1"] = r["rank1"]; facts[f"def_wb_{r['attack']}_{r['target']}_map"] = r["mAP"]
            else:
                facts[f"def_xfer_{r['attack']}_{r['source']}_to_{r['target']}_r1"] = r["rank1"]
    else:
        md.append("**NOT RUN YET** (run_attacks.py on resnet50 + resnet50_adv_trained_eps8 -> results/defense.json)")

    out_md = os.path.join("docs", "RESULTS_SUMMARY.md")
    os.makedirs("docs", exist_ok=True)
    open(out_md, "w").write("\n".join(md) + "\n")
    json.dump(facts, open(rpath("facts.json"), "w"), indent=1, sort_keys=True)
    print(f"wrote {out_md} and {rpath('facts.json')} ({len(facts)} facts)")

    if a.fill:
        missing, outputs = set(), []
        for tp in sorted(glob.glob("docs/*.template.md")):
            text = open(tp).read()
            def sub(m):
                k = m.group(1)
                if k not in facts:
                    missing.add(k); return m.group(0)
                v = facts[k]
                return f"{v:.1f}" if isinstance(v, float) else str(v)
            filled = re.sub(r"\{\{([A-Za-z0-9_@.]+)\}\}", sub, text)
            outputs.append((tp.replace(".template", ""), filled))
        if missing:
            raise SystemExit("\n[ERROR] These template numbers have no real source yet (the experiment has not run):\n  "
                             + "\n  ".join(sorted(missing)) + "\nNothing was written. Run the missing experiments and try again.\n")
        for path, filled in outputs:
            hdr = f"<!-- AUTO-GENERATED {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from {results_dir()}/*.json by scripts/build_docs.py - do not hand-edit numbers -->\n"
            open(path, "w").write(hdr + filled)
            print("filled", path)


if __name__ == "__main__":
    main()
