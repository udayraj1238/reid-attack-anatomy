"""Live demo on the REAL ResNet-50 (not a simulation): pick a Market-1501 query or upload a photo, apply a real
attack / the trained universal patch, and watch the real retrieval ranking change.

python demo/app.py --data <market1501_dir> --ckpt checkpoints/resnet50.pth [--patch results/patch.pth]
python demo/app.py --data ... --check        # headless self-check, no UI
The demo gallery is a seeded SUBSET of Market-1501 (150 identities) so it starts fast."""
import argparse, os, random, sys
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.io import load_model
from reidattack.patch import UniversalPatch
from reidattack.viz_utils import load_split, load_tensor, extract_items
from reidattack.demo_core import craft, search, ATTACK_CHOICES


def build_state(a, dev):
    model = load_model(a.ckpt, dev)
    q_items, g_items = load_split(a.data)
    rng = random.Random(0)
    by = {}
    for it in g_items:
        if it[1] > 0:
            by.setdefault(it[1], []).append(it)
    pids = sorted(rng.sample([p for p, v in by.items() if len(v) >= 4], a.n_ids))
    sub_g = [it for p in pids for it in by[p]]
    gf, gp, gc = extract_items(model, sub_g, dev)
    sub_q = [it for it in q_items if it[1] in set(pids)]
    patcher = None
    if a.patch and os.path.exists(a.patch):
        patcher = UniversalPatch().to(dev)
        patcher.load_state_dict(torch.load(a.patch, map_location=dev, weights_only=False))
    return dict(model=model, g_items=sub_g, gf=gf, gp=gp, gc=gc, q_items=sub_q, patcher=patcher, dev=dev)


def run_query(S, q_choice, upload, attack, eps):
    dev, model = S["dev"], S["model"]
    q_pid = q_cam = None
    if upload is not None:
        x = load_tensor_from_pil(upload).to(dev)
    else:
        path, q_pid, q_cam = S["q_items"][int(q_choice)]
        x = load_tensor(path)[None].to(dev)
    xa = craft(model, x, attack, eps, S["patcher"])
    res = []
    for xv in (x, xa):
        idx, sims, q = search(model, xv, S["gf"], S["gp"], S["gc"], q_pid, q_cam, 5)
        res.append((idx, sims, q))
    imgs = []
    for tag, (idx, sims, _) in zip(("before", "after"), res):
        for j, gi in enumerate(idx):
            im = Image.open(S["g_items"][gi][0]).convert("RGB").resize((128, 256))
            if q_pid is not None:
                ok = S["gp"][gi] == q_pid
                ImageDraw.Draw(im).rectangle([0, 0, 127, 255], outline=(0, 200, 0) if ok else (230, 0, 0), width=6)
            imgs.append((im, f"{tag} - rank {j+1}  sim {sims[j]:.2f}"))
    cos = float((res[0][2] * res[1][2]).sum())
    msg = f"cosine(clean emb, attacked emb) = {cos:.3f}"
    if q_pid is not None:
        msg += f" | top-1 correct: before={S['gp'][res[0][0][0]] == q_pid}, after={S['gp'][res[1][0][0]] == q_pid}"
    else:
        msg += " | (uploaded photo has no ground truth, so no green/red borders)"
    to_pil = lambda t: Image.fromarray((t[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    return to_pil(x), to_pil(xa), imgs, msg


def load_tensor_from_pil(pil):
    from torchvision import transforms as T
    return T.Compose([T.Resize((256, 128)), T.ToTensor()])(pil.convert("RGB"))[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="checkpoints/resnet50.pth")
    ap.add_argument("--patch", default="results/patch.pth")
    ap.add_argument("--n_ids", type=int, default=150)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = build_state(a, dev)
    print(f"demo gallery: {len(S['g_items'])} images, {len(S['q_items'])} sample queries, patch loaded: {S['patcher'] is not None}")
    if a.check:
        for atk in ATTACK_CHOICES:
            if atk == "Universal patch" and S["patcher"] is None:
                print("skip patch (no patch.pth)"); continue
            _, _, _, msg = run_query(S, 0, None, atk, 8)
            print(f"[check] {atk:16s} {msg}")
        return
    import gradio as gr
    choices = [(f"{os.path.basename(p)}", i) for i, (p, _, _) in enumerate(S["q_items"])]
    with gr.Blocks(title="ReID attack demo (real model)") as ui:
        gr.Markdown("# Attacking person re-identification (real ResNet-50, real attacks)\n"
                    "Pick a query, choose an attack, see the retrieval result before/after. Green = same identity, red = wrong.")
        with gr.Row():
            q = gr.Dropdown(choices, value=0, label="Sample query (Market-1501)")
            up = gr.Image(type="pil", label="...or upload your own photo (optional)")
        with gr.Row():
            atk = gr.Radio(ATTACK_CHOICES, value="PGD-10", label="Attack")
            eps = gr.Slider(1, 16, value=8, step=1, label="epsilon (/255) - ignored for the patch")
        btn = gr.Button("Run")
        with gr.Row():
            o1 = gr.Image(label="query (clean)"); o2 = gr.Image(label="query (attacked)")
        gal = gr.Gallery(label="top-5 before (first 5) and after (last 5)", columns=5, height=620)
        txt = gr.Textbox(label="summary")
        btn.click(lambda *args: run_query(S, *args), [q, up, atk, eps], [o1, o2, gal, txt])
    ui.launch(server_port=a.port, share=False)


if __name__ == "__main__":
    main()
