"""Adversarial Training Defence for Person Re-Identification.
Trains victim models with on-the-fly PGD perturbations to defend against evasion and metric manipulation.

Usage:
    python -m reidattack.adv_train --data /kaggle/input/market1501 --arch resnet50 --eps 8 --epochs 25
"""
import argparse
import os
import time
import torch
import torch.nn as nn
from .data import get_market
from .models import ReIDNet
from .losses import ReIDLoss
from .attacks import pgd, evade_objective
from .evaluate import extract, cmc_map


def evaluate(model, q_dl, g_dl, device):
    qf, qp, qc = extract(model, q_dl, device)
    gf, gp, gc = extract(model, g_dl, device)
    cmc, mAP = cmc_map(qf, gf, qp, gp, qc, gc)
    return cmc[0], mAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "densenet121", "vit_small"])
    ap.add_argument("--eps", type=int, default=8, help="Perturbation budget in /255")
    ap.add_argument("--pgd_steps", type=int, default=5, help="Inner PGD iterations for speed")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)
    train_dl, q_dl, g_dl, ncls = get_market(a.data, workers=a.workers)

    model = ReIDNet(a.arch, ncls).to(dev)
    crit = ReIDLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=5e-4)
    milestones = [int(a.epochs * 0.5), int(a.epochs * 0.8)]
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    eps_val = a.eps / 255.0

    print(f"=== Starting Adversarial Training for {a.arch.upper()} (ε={a.eps}/255, PGD-{a.pgd_steps}) ===")

    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        tot_clean, tot_adv = 0.0, 0.0

        warm = min(1.0, (ep + 1) / 5)
        for g in opt.param_groups:
            g["lr"] = a.lr * warm * (0.1 ** sum(ep >= m for m in milestones))

        for x, y, _ in train_dl:
            x, y = x.to(dev), y.to(dev)

            # 1. Craft adversarial perturbation on current batch
            model.eval()
            with torch.no_grad():
                clean_feat = model.embed(x)
            obj = evade_objective(model, clean_feat)
            x_adv = pgd(obj, x, eps=eps_val, steps=a.pgd_steps)

            # 2. Train on combination of clean and adversarial examples
            model.train()
            with torch.autocast(dev, enabled=dev == "cuda"):
                f_clean, logits_clean = model(x)
                loss_clean = crit(f_clean.float(), logits_clean.float(), y)

                f_adv, logits_adv = model(x_adv)
                loss_adv = crit(f_adv.float(), logits_adv.float(), y)

                # Total defence loss: standard ReID + robustness against perturbed representation
                loss = 0.5 * loss_clean + 0.5 * loss_adv

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tot_clean += loss_clean.item()
            tot_adv += loss_adv.item()

        n_batches = len(train_dl)
        print(f"Epoch {ep+1}/{a.epochs} | Clean Loss: {tot_clean/n_batches:.3f} | Adv Loss: {tot_adv/n_batches:.3f} ({time.time()-t0:.0f}s)", flush=True)

        if (ep + 1) % 5 == 0 or ep + 1 == a.epochs:
            r1, mAP = evaluate(model, q_dl, g_dl, dev)
            print(f"  >> [Robust Check] Clean Rank-1: {r1*100:.1f}%, mAP: {mAP*100:.1f}%", flush=True)
            torch.save(
                {"arch": a.arch, "ncls": ncls, "state": model.state_dict(), "adv_trained": True, "eps": a.eps},
                os.path.join(a.out, f"{a.arch}_adv_trained_eps{a.eps}.pth")
            )

    print(f"Adversarial training complete! Model saved to checkpoints/{a.arch}_adv_trained_eps{a.eps}.pth")


if __name__ == "__main__":
    main()
