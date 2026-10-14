"""Attack zoo + cross-architecture transfer matrix.
For every (attack, eps, SOURCE model): craft adversarial queries on the source, then evaluate
them on EVERY model (diagonal = white-box, off-diagonal = black-box transfer).

python scripts/run_attacks.py --data /kaggle/input/market1501 \
    --ckpts checkpoints/resnet50.pth checkpoints/densenet121.pth checkpoints/vit_small.pth \
    --attacks fgsm pgd mifgsm logit_pgd --eps 4 8 --mode evade
"""
import argparse, json, os, random, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.data import get_market
from reidattack.io import load_model
from reidattack.evaluate import extract, cmc_map, top1_pid
from reidattack.attacks import ATTACKS, pgd, evade_objective, impersonate_objective, logit_objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "mifgsm", "logit_pgd"])
    ap.add_argument("--eps", nargs="+", type=int, default=[8])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--mode", default="evade", choices=["evade", "impersonate"])
    ap.add_argument("--max_query", type=int, default=0, help="0 = all 3368 queries")
    ap.add_argument("--out", default="results/attacks_{mode}.json")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); random.seed(0); np.random.seed(0)

    _, q_dl, g_dl, _ = get_market(a.data, workers=a.workers)
    models = {}
    for p in a.ckpts:
        m = load_model(p, dev)
        for q in m.parameters():
            q.requires_grad_(False)
        models[m.arch] = m
    names = list(models)

    gal = {n: extract(models[n], g_dl, dev) for n in names}            # clean gallery per model
    g_pids, g_cams = gal[names[0]][1], gal[names[0]][2]
    centroids = {}
    for n in names:                                                      # identity prototypes (impersonation targets)
        gf = gal[n][0]
        centroids[n] = {int(p): torch.nn.functional.normalize(gf[g_pids == p].mean(0), dim=0)
                        for p in np.unique(g_pids) if p > 0}
    cand_pids = sorted(centroids[names[0]])

    # ---- collect query batches once
    batches, seen = [], 0
    for x, y, c in q_dl:
        batches.append((x, y.numpy(), c.numpy())); seen += len(x)
        if a.max_query and seen >= a.max_query: break
    q_pids = np.concatenate([b[1] for b in batches]); q_cams = np.concatenate([b[2] for b in batches])
    rng = random.Random(0)
    t_pids = np.array([rng.choice([p for p in rng.sample(cand_pids, 5) if p != qp]) for qp in q_pids])

    def score(feats, tgt, tpids=None):
        qf = torch.cat(feats)
        cmc, mAP = cmc_map(qf, gal[tgt][0], q_pids, g_pids, q_cams, g_cams)
        row = {"rank1": round(float(cmc[0]) * 100, 2), "mAP": round(mAP * 100, 2)}
        if tpids is not None:
            top = top1_pid(qf, gal[tgt][0], g_pids, q_pids, q_cams, g_cams)
            row["impersonation_hit@1"] = round(float((top == tpids).mean()) * 100, 2)
        return row

    out = a.out.format(mode=a.mode)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Check for existing results to resume if possible
    rows = []
    examples = {}
    done_keys = set()
    if os.path.exists(out):
        try:
            with open(out, "r") as f:
                rows = json.load(f)
            for r in rows:
                done_keys.add((r.get("attack"), r.get("eps"), r.get("source"), r.get("target")))
            print(f"Loaded {len(rows)} previously computed results from {out} for resuming.", flush=True)
        except Exception:
            rows = []

    # Clean baseline
    print(f"\n=======================================================", flush=True)
    print(f"Computing Clean Baseline Retrieval Metrics for {len(names)} Models", flush=True)
    print(f"=======================================================", flush=True)
    for tgt in names:
        if ("clean", 0, "-", tgt) not in done_keys:
            feats = [models[tgt].embed(x.to(dev)).detach().cpu() for x, _, _ in batches]
            r = {"attack": "clean", "eps": 0, "source": "-", "target": tgt, **score(feats, tgt)}
            rows.append(r)
            done_keys.add(("clean", 0, "-", tgt))
            print(f"  [Clean Baseline] Model: {tgt:12s} | Rank-1: {r['rank1']:5.2f}% | mAP: {r['mAP']:5.2f}%", flush=True)
        else:
            print(f"  [Clean Baseline] Model: {tgt:12s} | Already cached in {out}", flush=True)

    with open(out, "w") as f:
        json.dump(rows, f, indent=1)

    # Compute total configurations
    configs = []
    for atk in a.attacks:
        for eps in a.eps:
            for src in names:
                configs.append((atk, eps, src))

    total_cfgs = len(configs)
    print(f"\n=======================================================", flush=True)
    print(f"Running Attack Suite ({a.mode.upper()})", flush=True)
    print(f"Total Configurations: {total_cfgs} | Batches per config: {len(batches)}", flush=True)
    print(f"=======================================================\n", flush=True)

    for cfg_idx, (atk, eps, src) in enumerate(configs, 1):
        # Check if all targets for this config are already done
        all_done = all((atk, eps, src, t) in done_keys for t in names)
        if all_done:
            print(f"[{cfg_idx:2d}/{total_cfgs}] {atk.upper()} (eps={eps}) from {src} -> Already cached, skipping.", flush=True)
            continue

        print(f"\n---> [{cfg_idx:2d}/{total_cfgs}] Crafting {atk.upper()} (eps={eps}/255) on Source: {src}", flush=True)
        t_cfg_start = time.time()
        feats = {t: [] for t in names}
        off = 0

        for bi, (x, _, _) in enumerate(batches):
            x = x.to(dev); B = len(x); sm = models[src]
            with torch.no_grad():
                clean = sm.embed(x)
            if atk == "logit_pgd":          # classification-style baseline (pseudo-label = own prediction)
                with torch.no_grad():
                    lab = sm.logits(x).argmax(1)
                obj = logit_objective(sm, lab)
                xa = pgd(obj, x, eps / 255, a.steps)
            else:
                if a.mode == "evade":
                    obj = evade_objective(sm, clean)
                else:
                    tf = torch.stack([centroids[src][int(p)] for p in t_pids[off:off + B]]).to(dev)
                    obj = impersonate_objective(sm, tf)
                xa = ATTACKS[atk](obj, x, eps / 255, a.steps)
            assert (xa - x).abs().max() <= eps / 255 + 1e-5
            if bi == 0 and src == names[0]:
                examples[f"{atk}_eps{eps}"] = {"clean": x[:8].cpu(), "adv": xa[:8].cpu()}
            with torch.no_grad():
                for t in names:
                    feats[t].append(models[t].embed(xa).cpu())
            off += B

            if (bi + 1) % 10 == 0 or (bi + 1) == len(batches):
                pct = 100.0 * (bi + 1) / len(batches)
                elapsed = time.time() - t_cfg_start
                print(f"     [batch {bi+1:2d}/{len(batches)} ({pct:3.0f}%)] crafted & projected in {elapsed:.1f}s", flush=True)

        # Score across all victim models
        print(f"     Evaluating cross-architecture transfer across {len(names)} targets:", flush=True)
        for t in names:
            tp = t_pids if (a.mode == "impersonate" and atk != "logit_pgd") else None
            r = {"attack": atk, "eps": eps, "source": src, "target": t, **score(feats[t], t, tp)}
            rows.append(r)
            done_keys.add((atk, eps, src, t))
            imp_str = f" | Impersonate Hit@1: {r.get('impersonation_hit@1', 0):.1f}%" if tp is not None else ""
            print(f"       >> Target: {t:12s} | Rank-1: {r['rank1']:5.2f}% | mAP: {r['mAP']:5.2f}%{imp_str}", flush=True)

        # Save incrementally after each configuration completes
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)

    torch.save(examples, out.replace(".json", "_examples.pt"))
    print(f"\nSaved all results ({len(rows)} evaluations) to {out}", flush=True)


if __name__ == "__main__":
    main()
