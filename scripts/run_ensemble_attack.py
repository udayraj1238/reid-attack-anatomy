"""Leave-one-out ENSEMBLE transfer attack (REAL runs only; resumable).

Question it answers: single-surrogate attacks from ResNet-50 / DenseNet-121 barely hurt ViT-Small (Rank-1 ~88%).
Is that a real transfer barrier, or just because one CNN is a weak surrogate? Craft the perturbation on an ensemble of
the OTHER models (gradient of the mean cosine-evasion loss over members) and test on the held-out model.

Default ensembles (leave-one-out over the checkpoints you pass):
   resnet50+densenet121 -> held-out vit_small      resnet50+vit_small -> held-out densenet121     densenet121+vit_small -> held-out resnet50

python scripts/run_ensemble_attack.py --data /kaggle/input --ckpts checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth \
    --attacks pgd mifgsm --eps 8 [--ensembles resnet50+densenet121] [--max_query 160]
Same eps budget, same number of steps as the single-surrogate runs: the ensemble gets no larger perturbation, only better gradients.
"""
import argparse, json, os, sys, time
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.data import get_market
from reidattack.io import load_model
from reidattack.evaluate import extract, cmc_map
from reidattack.attacks import ATTACKS
from reidattack.results import results_dir, stamp


def ensemble_objective(members, cleans):
    """Maximise the mean (over members) of -cos(f_m(x_adv), f_m(x_clean)); same sign convention as attacks.evade_objective."""
    def obj(xa):
        return sum(-(m.embed(xa) * c).sum(1) for m, c in zip(members, cleans)) / len(members)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--ensembles", nargs="*", default=None, help="e.g. resnet50+densenet121 (default: every leave-one-out ensemble)")
    ap.add_argument("--attacks", nargs="+", default=["pgd", "mifgsm"], choices=["fgsm", "pgd", "mifgsm"])
    ap.add_argument("--eps", nargs="+", type=int, default=[8])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=16, help="crafting sub-batch size (an ensemble keeps several graphs in memory)")
    ap.add_argument("--max_query", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)
    out = a.out or os.path.join(results_dir(), "ensemble.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    _, q_dl, g_dl, _ = get_market(a.data, workers=a.workers)
    models = {}
    for p in a.ckpts:
        m = load_model(p, dev).eval()
        for q in m.parameters():
            q.requires_grad_(False)
        models[os.path.splitext(os.path.basename(p))[0]] = m
    names = list(models)
    ensembles = a.ensembles or ["+".join(n for n in names if n != h) for h in names]
    for e in ensembles:
        for n in e.split("+"):
            assert n in models, f"ensemble member {n} not among {names}"

    print("Extracting clean gallery embeddings...", flush=True)
    gal = {n: extract(models[n], g_dl, dev) for n in names}
    if dev == "cuda":
        torch.cuda.empty_cache()
    g_pids, g_cams = gal[names[0]][1], gal[names[0]][2]
    batches, seen = [], 0
    for x, y, c in q_dl:
        batches.append((x, y.numpy(), c.numpy())); seen += len(x)
        if a.max_query and seen >= a.max_query:
            break
    q_pids = np.concatenate([b[1] for b in batches]); q_cams = np.concatenate([b[2] for b in batches])
    print(f"{len(q_pids)} queries | ensembles={ensembles} | attacks={a.attacks} eps={a.eps} steps={a.steps}", flush=True)

    def score(feats, tgt):
        cmc, mAP = cmc_map(torch.cat(feats), gal[tgt][0], q_pids, g_pids, q_cams, g_cams)
        return {"rank1": round(float(cmc[0]) * 100, 2), "rank5": round(float(cmc[4]) * 100, 2), "mAP": round(mAP * 100, 2)}

    rows, done = [], set()
    if os.path.exists(out):
        try:
            rows = json.load(open(out)); done = {(r["attack"], r["eps"], r["ensemble"], r["target"]) for r in rows}
            print(f"Resuming: {len(rows)} rows in {out}", flush=True)
        except Exception:
            rows = []

    def save():
        json.dump(rows, open(out, "w"), indent=1)

    for t in names:
        if ("clean", 0, "-", t) not in done:
            feats = [models[t].embed(x.to(dev)).detach().cpu() for x, _, _ in batches]
            rows.append({"attack": "clean", "eps": 0, "ensemble": "-", "members": [], "target": t, "role": "-", **score(feats, t)})
            done.add(("clean", 0, "-", t))
    save()

    configs = [(atk, eps, e) for e in ensembles for atk in a.attacks for eps in a.eps]
    for ci, (atk, eps, e) in enumerate(configs, 1):
        if all((atk, eps, e, t) in done for t in names):
            print(f"[{ci}/{len(configs)}] {atk} eps={eps} {e}: cached", flush=True); continue
        members = [models[n] for n in e.split("+")]
        print(f"\n[{ci}/{len(configs)}] crafting {atk} eps={eps}/255 on ensemble {e}", flush=True)
        t0 = time.time(); feats = {t: [] for t in names}
        for bi, (x, _, _) in enumerate(batches):
            for xc in x.to(dev).split(a.chunk):
                with torch.no_grad():
                    cleans = [m.embed(xc) for m in members]
                xa = ATTACKS[atk](ensemble_objective(members, cleans), xc, eps / 255, a.steps)
                assert (xa - xc).abs().max() <= eps / 255 + 1e-5, "epsilon budget violated"
                with torch.no_grad():
                    for t in names:
                        feats[t].append(models[t].embed(xa).cpu())
            if dev == "cuda":
                torch.cuda.empty_cache()
            if (bi + 1) % 5 == 0 or bi + 1 == len(batches):
                print(f"   batch {bi+1}/{len(batches)}  {time.time()-t0:.0f}s", flush=True)
        for t in names:
            role = "member" if t in e.split("+") else "held_out"
            r = {"attack": atk, "eps": eps, "ensemble": e, "members": e.split("+"), "target": t, "role": role, **score(feats[t], t)}
            rows.append(r); done.add((atk, eps, e, t))
            print(f"   -> {t:14s} [{role:8s}] R1 {r['rank1']:.2f}  mAP {r['mAP']:.2f}", flush=True)
        save()
    stamp(f"run_ensemble:{os.path.basename(out)}", ckpts=a.ckpts,
          extra={"n_queries": int(len(q_pids)), "attacks": a.attacks, "eps": a.eps, "steps": a.steps, "ensembles": ensembles})
    print(f"\nSaved {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
