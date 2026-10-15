"""The 'money slide': for random queries that clean ResNet-50 matches correctly, show the real top-5 gallery
results before and after a white-box PGD-10 evasion attack. Green border = same identity, red = different.

python scripts/retrieval_figure.py --data /kaggle/input --ckpt checkpoints/resnet50.pth
"""
import argparse, os, random, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.io import load_model
from reidattack.attacks import pgd, evade_objective
from reidattack.viz_utils import load_split, load_tensor, extract_items, rank_gallery
from reidattack.results import rpath, stamp


def show(ax, img, border=None, title=None):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(border if border else "none"); s.set_linewidth(4 if border else 0)
    if title:
        ax.set_title(title, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="checkpoints/resnet50.pth")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or rpath("retrieval_before_after.png")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, dev)
    q_items, g_items = load_split(a.data)
    gf, gp, gc = extract_items(model, g_items, dev)
    order = list(range(len(q_items))); random.Random(1).shuffle(order)

    rows, n_hit = [], 0
    for i in order:
        path, pid, cam = q_items[i]
        x = load_tensor(path)[None].to(dev)
        with torch.no_grad():
            clean = model.embed(x)
        top_c = rank_gallery(clean[0].cpu(), gf, gp, gc, pid, cam, a.k)
        if gp[top_c[0]] != pid:
            continue                                   # only queries the clean model gets right
        xa = pgd(evade_objective(model, clean), x, a.eps / 255, 10)
        with torch.no_grad():
            adv = model.embed(xa)
        top_a = rank_gallery(adv[0].cpu(), gf, gp, gc, pid, cam, a.k)
        n_hit += int(gp[top_a[0]] == pid)
        rows.append((x[0].cpu(), xa[0].detach().cpu(), pid, top_c, top_a))
        if len(rows) == a.n:
            break

    fig, axes = plt.subplots(2 * len(rows), a.k + 1, figsize=(1.7 * (a.k + 1), 3.6 * len(rows)))
    for r, (x, xa, pid, top_c, top_a) in enumerate(rows):
        show(axes[2 * r, 0], x.permute(1, 2, 0).numpy(), title="clean query")
        show(axes[2 * r + 1, 0], xa.permute(1, 2, 0).numpy(), title=f"PGD eps={a.eps}")
        for j in range(a.k):
            for rr, top in ((2 * r, top_c), (2 * r + 1, top_a)):
                gi = top[j]
                ok = gp[gi] == pid
                show(axes[rr, j + 1], np.asarray(Image.open(g_items[gi][0]).convert("RGB").resize((128, 256))),
                     border="limegreen" if ok else "red", title=f"rank {j+1}")
    fig.suptitle(f"Real top-{a.k} retrieval (ResNet-50), before vs after white-box PGD-10 (eps={a.eps}/255)\n"
                 f"{len(rows)} random queries that were correct at rank-1 on clean data; still correct after attack: {n_hit}/{len(rows)}.  "
                 "green = same identity, red = different", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(out, dpi=170); plt.close()
    stamp(f"figure:{os.path.basename(out)}", ckpts=[a.ckpt], extra={"still_correct_after_attack": n_hit, "n": len(rows)})
    print("saved", out)


if __name__ == "__main__":
    main()
