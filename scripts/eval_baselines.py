"""Evaluate every checkpoint on the full Market-1501 protocol -> results/baselines.json (real numbers only).
python scripts/eval_baselines.py --data /kaggle/input --ckpts checkpoints/resnet50.pth checkpoints/densenet121.pth ..."""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.data import get_market
from reidattack.io import load_model
from reidattack.evaluate import extract, cmc_map
from reidattack.results import results_dir, stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(results_dir(), "baselines.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, q_dl, g_dl, _ = get_market(a.data, workers=a.workers)
    rows = []
    for p in a.ckpts:
        m = load_model(p, dev)
        qf, qp, qc = extract(m, q_dl, dev)
        gf, gp, gc = extract(m, g_dl, dev)
        cmc, mAP = cmc_map(qf, gf, qp, gp, qc, gc)
        r = {"model": os.path.splitext(os.path.basename(p))[0], "arch": m.arch, "emb_dim": m.dim,
             "params_M": round(sum(x.numel() for x in m.parameters()) / 1e6, 1),
             "rank1": round(float(cmc[0]) * 100, 2), "rank5": round(float(cmc[4]) * 100, 2),
             "rank10": round(float(cmc[9]) * 100, 2), "mAP": round(mAP * 100, 2),
             "n_query": int(len(qf)), "n_gallery": int(len(gf))}
        rows.append(r); print(r, flush=True)
    json.dump(rows, open(out, "w"), indent=1)
    stamp("eval_baselines", ckpts=a.ckpts)


if __name__ == "__main__":
    main()
