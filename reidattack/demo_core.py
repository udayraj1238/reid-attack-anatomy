"""Core logic of the live demo (no Gradio here, so it can be tested without a UI). Uses the REAL model."""
import torch
from .attacks import fgsm, pgd, mifgsm, evade_objective
from .viz_utils import rank_gallery

ATTACK_CHOICES = ["none", "FGSM", "PGD-10", "MI-FGSM", "Universal patch"]


def craft(model, x, attack, eps_255=8, patcher=None):
    """x: (1,3,256,128) in [0,1]. Returns the attacked image (same shape)."""
    if attack == "none":
        return x
    if attack == "Universal patch":
        if patcher is None:
            raise ValueError("No trained patch loaded (run scripts/run_patch.py first).")
        with torch.no_grad():
            return patcher(x, eot=False)
    with torch.no_grad():
        clean = model.embed(x)
    obj, eps = evade_objective(model, clean), eps_255 / 255
    return {"FGSM": lambda: fgsm(obj, x, eps), "PGD-10": lambda: pgd(obj, x, eps, 10),
            "MI-FGSM": lambda: mifgsm(obj, x, eps, 10)}[attack]()


@torch.no_grad()
def search(model, x, gf, g_pids, g_cams, q_pid=None, q_cam=None, k=5):
    q = model.embed(x)[0].cpu()
    idx = rank_gallery(q, gf, g_pids, g_cams, q_pid, q_cam, k)
    sims = (gf[idx] @ q).numpy()
    return idx, sims, q
