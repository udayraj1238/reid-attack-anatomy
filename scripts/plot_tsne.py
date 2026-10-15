"""REAL t-SNE of ResNet-50 embeddings: clean gallery images of several identities, their clean queries,
and the same queries after a white-box PGD evasion attack. Also prints the quantitative drift.

python scripts/plot_tsne.py --data /kaggle/input --ckpt checkpoints/resnet50.pth
"""
import argparse, os, random, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.io import load_model
from reidattack.attacks import pgd, evade_objective
from reidattack.viz_utils import load_split, load_tensor, extract_items, rank_gallery
from reidattack.results import rpath, stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="checkpoints/resnet50.pth")
    ap.add_argument("--n_ids", type=int, default=8)
    ap.add_argument("--q_per_id", type=int, default=4)
    ap.add_argument("--g_per_id", type=int, default=25)
    ap.add_argument("--eps", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or rpath("tsne_embedding_drift.png")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(0)
    model = load_model(a.ckpt, dev)
    q_items, g_items = load_split(a.data)
    gf, gp, gc = extract_items(model, g_items, dev)

    by_q = {}
    for it in q_items:
        by_q.setdefault(it[1], []).append(it)
    pids = [p for p, v in by_q.items() if len(v) >= a.q_per_id and (gp == p).sum() >= 8]
    pids = sorted(rng.sample(pids, a.n_ids))

    G, Gl, Qc, Qa, Ql, ok_adv, ok_clean = [], [], [], [], [], 0, 0
    for k, pid in enumerate(pids):
        idx = np.where(gp == pid)[0]
        idx = rng.sample(list(idx), min(a.g_per_id, len(idx)))
        G.append(gf[idx]); Gl += [k] * len(idx)
        qs = rng.sample(by_q[pid], a.q_per_id)
        x = torch.stack([load_tensor(p) for p, _, _ in qs]).to(dev)
        with torch.no_grad():
            clean = model.embed(x)
        xa = pgd(evade_objective(model, clean), x, a.eps / 255, 10)
        with torch.no_grad():
            adv = model.embed(xa)
        for j, (_, qp, qcam) in enumerate(qs):
            ok_clean += int(gp[rank_gallery(clean[j].cpu(), gf, gp, gc, qp, qcam, 1)[0]] == qp)
            ok_adv += int(gp[rank_gallery(adv[j].cpu(), gf, gp, gc, qp, qcam, 1)[0]] == qp)
        Qc.append(clean.cpu()); Qa.append(adv.cpu()); Ql += [k] * len(qs)
    G, Qc, Qa = torch.cat(G).numpy(), torch.cat(Qc).numpy(), torch.cat(Qa).numpy()
    Gl, Ql = np.array(Gl), np.array(Ql)
    allx = np.concatenate([G, Qc, Qa])
    emb2 = TSNE(n_components=2, perplexity=min(30, max(5, len(allx) // 8)), init="pca", random_state=0).fit_transform(allx)
    g2, c2, a2 = emb2[:len(G)], emb2[len(G):len(G) + len(Qc)], emb2[len(G) + len(Qc):]
    cos = float((Qc * Qa).sum(1).mean())
    nq = len(Ql)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for k in range(len(pids)):
        col = cmap(k % 10)
        ax.scatter(*g2[Gl == k].T, s=14, color=col, alpha=0.45)
        ax.scatter(*c2[Ql == k].T, s=95, color=col, edgecolor="black", linewidth=1.2, marker="o", zorder=3)
        ax.scatter(*a2[Ql == k].T, s=90, color=col, edgecolor="black", linewidth=1.0, marker="X", zorder=3)
    for i in range(nq):
        ax.plot([c2[i, 0], a2[i, 0]], [c2[i, 1], a2[i, 1]], color="grey", alpha=0.35, linewidth=0.8)
    ax.scatter([], [], s=14, color="grey", label="clean gallery images (colour = identity)")
    ax.scatter([], [], s=95, color="white", edgecolor="black", marker="o", label="clean query")
    ax.scatter([], [], s=90, color="white", edgecolor="black", marker="X", label=f"same query after PGD-10 (eps={a.eps})")
    ax.legend(loc="best", fontsize=9)
    ax.set_title(f"ResNet-50 embedding space (t-SNE): identity clusters vs attacked queries\n"
                 f"mean cosine(clean, adv) = {cos:.3f} | top-1 correct: clean {ok_clean}/{nq} -> adversarial {ok_adv}/{nq}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    stamp(f"figure:{os.path.basename(out)}", ckpts=[a.ckpt],
          extra={"ids": [int(p) for p in pids], "mean_cos_clean_adv": cos, "top1_clean": ok_clean, "top1_adv": ok_adv, "n_queries": nq})
    print(f"saved {out} | mean cos(clean,adv)={cos:.3f} | top-1 clean {ok_clean}/{nq} -> adv {ok_adv}/{nq}")


if __name__ == "__main__":
    main()
