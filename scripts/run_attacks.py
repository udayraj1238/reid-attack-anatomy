"""Attack zoo + cross-architecture transfer matrix (REAL runs only; resumable).

For every (attack, eps, SOURCE model): craft adversarial queries on the source, then evaluate them
on EVERY model (diagonal = white-box, off-diagonal = black-box transfer).
Models are named by checkpoint file stem, so a standard and an adversarially-trained ResNet-50 can coexist.

Examples
  python scripts/run_attacks.py --data /kaggle/input --ckpts checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth \
      --attacks fgsm pgd mifgsm logit_pgd --eps 8 --mode evade
  python scripts/run_attacks.py ... --eps 2 4 16 --sources resnet50          # eps sweep, appended to the same JSON
  python scripts/run_attacks.py ... --mode impersonate --attacks pgd mifgsm
"""
import argparse, json, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.data import get_market
from reidattack.io import load_model
from reidattack.evaluate import extract, cmc_map, top1_pid
from reidattack.attacks import ATTACKS, pgd, evade_objective, impersonate_objective, logit_objective
from reidattack.results import results_dir, stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--sources", nargs="*", default=None, help="model names to craft attacks on (default: all)")
    ap.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "mifgsm", "logit_pgd"])
    ap.add_argument("--eps", nargs="+", type=int, default=[8])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=32, help="crafting sub-batch size (default: 32)")
    ap.add_argument("--mode", default="evade", choices=["evade", "impersonate"])
    ap.add_argument("--max_query", type=int, default=0, help="0 = all 3368 queries (use a small number only for dry runs)")
    ap.add_argument("--out", default=None, help="default: <results>/attacks_<mode>.json")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    out = a.out or os.path.join(results_dir(), f"attacks_{a.mode}.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    attacks = list(a.attacks)
    if a.mode == "impersonate" and "logit_pgd" in attacks:
        print("[note] logit_pgd is untargeted, skipping it in impersonate mode", flush=True)
        attacks.remove("logit_pgd")

    _, q_dl, g_dl, _ = get_market(a.data, workers=a.workers)
    models = {}
    for p in a.ckpts:
        m = load_model(p, dev)
        for q in m.parameters():
            q.requires_grad_(False)
        models[os.path.splitext(os.path.basename(p))[0]] = m
    names = list(models)
    sources = a.sources if a.sources else names
    for s in sources:
        assert s in models, f"--sources {s} not among {names}"

    print("Extracting clean gallery embeddings...", flush=True)
    gal = {n: extract(models[n], g_dl, dev) for n in names}
    if dev == "cuda":
        torch.cuda.empty_cache()
    g_pids, g_cams = gal[names[0]][1], gal[names[0]][2]
    centroids = {}
    for n in names:                                   # identity prototypes = impersonation targets
        gf = gal[n][0]
        centroids[n] = {int(p): torch.nn.functional.normalize(gf[torch.from_numpy(g_pids == p)].mean(0), dim=0)
                        for p in np.unique(g_pids) if p > 0}
    cand_pids = sorted(centroids[names[0]])

    batches, seen = [], 0
    for x, y, c in q_dl:
        batches.append((x, y.numpy(), c.numpy())); seen += len(x)
        if a.max_query and seen >= a.max_query:
            break
    q_pids = np.concatenate([b[1] for b in batches]); q_cams = np.concatenate([b[2] for b in batches])
    rng = random.Random(0)
    t_pids = np.array([rng.choice([p for p in rng.sample(cand_pids, 5) if p != qp]) for qp in q_pids])
    print(f"{len(q_pids)} queries, {len(models)} models, attacks={attacks}, eps={a.eps}, sources={sources}", flush=True)

    def score(feats, tgt, tpids=None):
        qf = torch.cat(feats)
        cmc, mAP = cmc_map(qf, gal[tgt][0], q_pids, g_pids, q_cams, g_cams)
        row = {"rank1": round(float(cmc[0]) * 100, 2), "rank5": round(float(cmc[4]) * 100, 2), "mAP": round(mAP * 100, 2)}
        if tpids is not None:
            top = top1_pid(qf, gal[tgt][0], g_pids, q_pids, q_cams, g_cams)
            row["impersonation_hit@1"] = round(float((top == tpids).mean()) * 100, 2)
        return row

    rows, done = [], set()
    if os.path.exists(out):                           # resume
        try:
            rows = json.load(open(out))
            done = {(r["attack"], r["eps"], r["source"], r["target"]) for r in rows}
            print(f"Resuming: {len(rows)} rows already in {out}", flush=True)
        except Exception:
            rows = []

    def save():
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)

    for tgt in names:                                 # clean baselines (same query set as the attacks)
        if ("clean", 0, "-", tgt) not in done:
            feats = [models[tgt].embed(x.to(dev)).detach().cpu() for x, _, _ in batches]
            r = {"attack": "clean", "eps": 0, "source": "-", "target": tgt, **score(feats, tgt)}
            rows.append(r); done.add(("clean", 0, "-", tgt))
            print(f"[clean] {tgt:28s} R1 {r['rank1']:.2f}  mAP {r['mAP']:.2f}", flush=True)
    save()

    configs = [(atk, eps, src) for atk in attacks for eps in a.eps for src in sources]
    for ci, (atk, eps, src) in enumerate(configs, 1):
        if all((atk, eps, src, t) in done for t in names):
            print(f"[{ci}/{len(configs)}] {atk} eps={eps} src={src}: cached", flush=True)
            continue
        print(f"\n[{ci}/{len(configs)}] crafting {atk} eps={eps}/255 on {src}", flush=True)
        t0 = time.time(); feats = {t: [] for t in names}; off = 0
        sm = models[src]
        for bi, (x, _, _) in enumerate(batches):
            x = x.to(dev); B = len(x)
            xas = []
            for ci, xc in enumerate(x.split(a.chunk)):
                with torch.no_grad():
                    clean = sm.embed(xc)
                if atk == "logit_pgd":                    # classification-style baseline (pseudo-label = own prediction)
                    with torch.no_grad():
                        lab = sm.logits(xc).argmax(1)
                    xac = pgd(logit_objective(sm, lab), xc, eps / 255, a.steps)
                else:
                    if a.mode == "evade":
                        obj = evade_objective(sm, clean)
                    else:
                        c_off = off + ci * a.chunk
                        tf = torch.stack([centroids[src][int(p)] for p in t_pids[c_off:c_off + len(xc)]]).to(dev)
                        obj = impersonate_objective(sm, tf)
                    xac = ATTACKS[atk](obj, xc, eps / 255, a.steps)
                xas.append(xac)
            xa = torch.cat(xas)
            assert (xa - x).abs().max() <= eps / 255 + 1e-5, "epsilon budget violated"
            with torch.no_grad():
                for t in names:
                    feats[t].append(models[t].embed(xa).cpu())
            off += B
            if (bi + 1) % 10 == 0 or bi + 1 == len(batches):
                print(f"   batch {bi+1}/{len(batches)}  {time.time()-t0:.0f}s", flush=True)
        for t in names:
            tp = t_pids if (a.mode == "impersonate") else None
            r = {"attack": atk, "eps": eps, "source": src, "target": t, **score(feats[t], t, tp)}
            rows.append(r); done.add((atk, eps, src, t))
            extra = f"  hit@1 {r['impersonation_hit@1']:.1f}" if tp is not None else ""
            print(f"   -> {t:28s} R1 {r['rank1']:.2f}  mAP {r['mAP']:.2f}{extra}", flush=True)
        save()                                        # incremental save: a crash never loses finished configs
    stamp(f"run_attacks:{os.path.basename(out)}", ckpts=a.ckpts,
          extra={"n_queries": int(len(q_pids)), "attacks": attacks, "eps": a.eps, "steps": a.steps, "out": os.path.basename(out)})
    print(f"\nSaved {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
