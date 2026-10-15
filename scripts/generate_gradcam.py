"""REAL Grad-CAM for retrieval: where does ResNet-50 look when matching a query to its true identity,
and what happens to that evidence under FGSM / PGD / MI-FGSM (eps=8) and the universal patch?

Queries are RANDOM (seeded) among those correctly matched at rank-1 on clean data - not hand-picked failures.
python scripts/generate_gradcam.py --data /kaggle/input --ckpt checkpoints/resnet50.pth [--patch results/patch.pth]
"""
import argparse, os, random, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.io import load_model
from reidattack.attacks import fgsm, pgd, mifgsm, evade_objective
from reidattack.patch import UniversalPatch
from reidattack.viz_utils import load_split, load_tensor, extract_items, rank_gallery, identity_centroid, gradcam_resnet
from reidattack.results import rpath, stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="checkpoints/resnet50.pth")
    ap.add_argument("--patch", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or rpath("gradcam_analysis.png")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, dev)
    q_items, g_items = load_split(a.data)
    gf, gp, gc = extract_items(model, g_items, dev)

    patch_path = a.patch or rpath("patch.pth")
    patcher = None
    if os.path.exists(patch_path):
        patcher = UniversalPatch().to(dev)
        patcher.load_state_dict(torch.load(patch_path, map_location=dev, weights_only=False))

    order = list(range(len(q_items))); random.Random(0).shuffle(order)
    chosen = []
    for i in order:
        path, pid, cam = q_items[i]
        with torch.no_grad():
            qf = model.embed(load_tensor(path)[None].to(dev))[0].cpu()
        if gp[rank_gallery(qf, gf, gp, gc, pid, cam, 1)[0]] == pid:
            chosen.append(i)
        if len(chosen) == a.n:
            break

    eps = a.eps / 255
    names = ["Clean", f"FGSM (eps={a.eps})", f"PGD-10 (eps={a.eps})", f"MI-FGSM (eps={a.eps})"] + (["Universal patch"] if patcher else [])
    fig, axes = plt.subplots(2 * len(chosen), len(names), figsize=(2.6 * len(names), 4.6 * len(chosen)))
    for r, i in enumerate(chosen):
        path, pid, cam = q_items[i]
        x0 = load_tensor(path)[None].to(dev)
        with torch.no_grad():
            clean = model.embed(x0)
        obj = evade_objective(model, clean)
        variants = [x0, fgsm(obj, x0, eps), pgd(obj, x0, eps, 10), mifgsm(obj, x0, eps, 10)]
        if patcher:
            with torch.no_grad():
                variants.append(patcher(x0, eot=False))
        cen = identity_centroid(gf, gp, gc, pid, cam).to(dev)
        for c, (name, xv) in enumerate(zip(names, variants)):
            cam_map, score = gradcam_resnet(model, xv, cen)
            with torch.no_grad():
                qf = model.embed(xv)[0].cpu()
            ok = gp[rank_gallery(qf, gf, gp, gc, pid, cam, 1)[0]] == pid
            img = xv[0].detach().cpu().permute(1, 2, 0).numpy()
            ax = axes[2 * r, c]; ax.imshow(img); ax.axis("off")
            ax.set_title(f"{name}\ntop-1: {'correct' if ok else 'WRONG'}", fontsize=9, color="green" if ok else "red")
            ax = axes[2 * r + 1, c]; ax.imshow(img); ax.imshow(cam_map[0].numpy(), cmap="jet", alpha=0.5); ax.axis("off")
            ax.set_title(f"sim to true identity = {score[0]:.2f}", fontsize=9)
    fig.suptitle("Grad-CAM for retrieval (ResNet-50): evidence for the TRUE identity, before vs after attack\n"
                 "(random queries that were correctly matched on clean data; heat-maps normalised per image)", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out, dpi=200); plt.close()
    stamp(f"figure:{os.path.basename(out)}", ckpts=[a.ckpt] + ([patch_path] if patcher else []), extra={"queries": [q_items[i][0] for i in chosen]})
    print("saved", out)


if __name__ == "__main__":
    main()
