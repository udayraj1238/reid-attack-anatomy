"""Train the universal torso patch (EOT) on a SOURCE model; evaluate white-box + transfer.
Conditions per model: clean | random_patch (control) | trained_patch (fixed position) | trained_patch_eot (random scale/rotation/shift/lighting at test time)

python scripts/run_patch.py --data /kaggle/input --src checkpoints/resnet50.pth \
    --others checkpoints/densenet121.pth checkpoints/vit_small.pth --epochs 30
"""
import argparse, json, os, sys
import torch
from torchvision.utils import save_image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.data import get_market
from reidattack.io import load_model
from reidattack.evaluate import extract, cmc_map
from reidattack.patch import UniversalPatch, train_patch
from reidattack.results import results_dir, stamp
from reidattack.logger import log, setup_logging


@torch.no_grad()
def eval_patched(model, patcher, q_dl, gal, dev, eot=False):
    qf, pids, cams = [], [], []
    for x, y, c in q_dl:
        x = x.to(dev)
        if patcher is not None:
            x = patcher(x, eot=eot)
        qf.append(model.embed(x).cpu()); pids.append(y); cams.append(c)
    qf = torch.cat(qf); pids = torch.cat(pids).numpy(); cams = torch.cat(cams).numpy()
    cmc, mAP = cmc_map(qf, gal[0], pids, gal[1], cams, gal[2])
    return {"rank1": round(float(cmc[0]) * 100, 2), "rank5": round(float(cmc[4]) * 100, 2), "mAP": round(mAP * 100, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--tv", type=float, default=0.1)
    ap.add_argument("--ph", type=int, default=64)
    ap.add_argument("--pw", type=int, default=48)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or results_dir()
    os.makedirs(out, exist_ok=True)
    setup_logging("patch")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    train_dl, q_dl, g_dl, _ = get_market(a.data, workers=a.workers)
    models = {}
    for p in [a.src] + a.others:
        models[os.path.splitext(os.path.basename(p))[0]] = load_model(p, dev)
    src = list(models)[0]
    patcher = train_patch(models[src], train_dl, epochs=a.epochs, lr=a.lr, tv=a.tv, device=dev,
                          ph=a.ph, pw=a.pw, log=log)
    rand = UniversalPatch(a.ph, a.pw).to(dev)         # control: untrained random-noise patch of the same size
    rows = []
    for n, m in models.items():
        gal = extract(m, g_dl, dev)
        for tag, pt, eot in [("clean", None, False), ("random_patch", rand, False),
                             ("trained_patch", patcher, False), ("trained_patch_eot", patcher, True)]:
            r = {"model": n, "condition": tag, "is_source": n == src, **eval_patched(m, pt, q_dl, gal, dev, eot)}
            rows.append(r); log(str(r))
    json.dump(rows, open(os.path.join(out, "patch_results.json"), "w"), indent=1)
    json.dump({"epochs": a.epochs, "lr": a.lr, "tv": a.tv, "size": [a.ph, a.pw],
               "mean_cos_sim_per_epoch": [round(h, 4) for h in patcher.history]},
              open(os.path.join(out, "patch_training.json"), "w"), indent=1)
    torch.save(patcher.state_dict(), os.path.join(out, "patch.pth"))
    save_image(patcher.patch.detach().cpu(), os.path.join(out, "patch.png"))
    x, _, _ = next(iter(q_dl))
    with torch.no_grad():
        save_image(torch.cat([x[:8], patcher(x[:8].to(dev), eot=False).cpu()]), os.path.join(out, "patch_examples.png"), nrow=8)
    stamp("run_patch", ckpts=[a.src] + a.others, extra={"epochs": a.epochs, "source_model": src})
    log(f"saved patch results to {out}")


if __name__ == "__main__":
    main()
