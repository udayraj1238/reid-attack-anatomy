"""Train the universal torso patch on a source model, evaluate on queries (white-box + transfer),
compare with a RANDOM patch of the same size (control).

python scripts/run_patch.py --data /kaggle/input/market1501 --src checkpoints/resnet50.pth \
    --others checkpoints/densenet121.pth checkpoints/vit_small.pth --epochs 5
"""
import argparse, json, os, sys
import torch
from torchvision.utils import save_image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.data import get_market
from reidattack.io import load_model
from reidattack.evaluate import extract, cmc_map
from reidattack.patch import UniversalPatch, train_patch


@torch.no_grad()
def eval_patched(model, patcher, q_dl, gal, dev):
    qf, qp, qc = [], None, None
    pids, cams = [], []
    for x, y, c in q_dl:
        xx = patcher(x.to(dev), eot=False) if patcher is not None else x.to(dev)
        qf.append(model.embed(xx).cpu()); pids.append(y); cams.append(c)
    qf = torch.cat(qf); pids = torch.cat(pids).numpy(); cams = torch.cat(cams).numpy()
    cmc, mAP = cmc_map(qf, gal[0], pids, gal[1], cams, gal[2])
    return {"rank1": round(float(cmc[0]) * 100, 2), "mAP": round(mAP * 100, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    train_dl, q_dl, g_dl, _ = get_market(a.data, workers=a.workers)
    models = {}
    for p in [a.src] + a.others:
        m = load_model(p, dev); models[m.arch] = m
    src = list(models)[0]
    patcher = train_patch(models[src], train_dl, epochs=a.epochs, device=dev)
    torch.manual_seed(0)
    rand = UniversalPatch().to(dev)                                       # control: untrained random patch
    rows = []
    for n, m in models.items():
        gal = extract(m, g_dl, dev)
        for tag, pt in [("clean", None), ("random_patch", rand), ("trained_patch", patcher)]:
            r = {"model": n, "condition": tag, "is_source": n == src, **eval_patched(m, pt, q_dl, gal, dev)}
            rows.append(r); print(r, flush=True)
    json.dump(rows, open(os.path.join(a.out, "patch_results.json"), "w"), indent=1)
    torch.save(patcher.state_dict(), os.path.join(a.out, "patch.pth"))
    save_image(patcher.patch.detach().cpu(), os.path.join(a.out, "patch.png"))
    x, _, _ = next(iter(q_dl))
    save_image(torch.cat([x[:8], patcher(x[:8].to(dev), eot=False).detach().cpu()]),
               os.path.join(a.out, "patch_examples.png"), nrow=8)


if __name__ == "__main__":
    main()
