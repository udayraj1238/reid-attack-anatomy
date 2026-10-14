import argparse, os, sys, time, torch
from .data import get_market
from .models import ReIDNet
from .losses import ReIDLoss
from .evaluate import extract, cmc_map
from .logger import log, setup_logging


def evaluate(model, q_dl, g_dl, device):
    log(f"  [eval] Extracting query embeddings ({len(q_dl)} batches)...")
    qf, qp, qc = extract(model, q_dl, device)
    log(f"  [eval] Extracting gallery embeddings ({len(g_dl)} batches)...")
    gf, gp, gc = extract(model, g_dl, device)
    log(f"  [eval] Computing CMC curve and mAP for {len(qf)} queries against {len(gf)} gallery images...")
    cmc, mAP = cmc_map(qf, gf, qp, gp, qc, gc)
    return cmc, mAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch", default="resnet50", choices=["resnet50", "densenet121", "vit_small"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--log_every", type=int, default=25, help="Batch logging interval")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip_if_exists", action="store_true", help="Skip if final checkpoint already exists")
    a = ap.parse_args()

    setup_logging(f"train_{a.arch}")

    target_ckpt = os.path.join(a.out, f"{a.arch}.pth")
    if a.skip_if_exists and os.path.exists(target_ckpt):
        log(f"=== Checkpoint {target_ckpt} already exists! Skipping training. ===")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True)

    log(f"\n=======================================================")
    log(f"Starting Training: {a.arch.upper()} on {dev.upper()}")
    log(f"Dataset: {a.data} | Epochs: {a.epochs} | Base LR: {a.lr}")
    log(f"=======================================================")

    t_start = time.time()
    train_dl, q_dl, g_dl, ncls = get_market(a.data, workers=a.workers)
    n_batches = len(train_dl)
    log(f"Loaded Market-1501: {ncls} identities, {n_batches} train batches, {len(q_dl)} query batches, {len(g_dl)} gallery batches")

    model = ReIDNet(a.arch, ncls).to(dev)
    crit = ReIDLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=5e-4)
    milestones = [int(a.epochs * 0.5), int(a.epochs * 0.8)]
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    best_map = 0.0

    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        warm = min(1.0, (ep + 1) / 5)                      # 5-epoch linear warm-up
        current_lr = a.lr * warm * (0.1 ** sum(ep >= m for m in milestones))
        for g in opt.param_groups:
            g["lr"] = current_lr

        log(f"\n--- Epoch {ep+1}/{a.epochs} (lr={current_lr:.2e}) ---")

        for bi, (x, y, _) in enumerate(train_dl):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                f, logits = model(x)
                loss = crit(f.float(), logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()

            if (bi + 1) % a.log_every == 0 or (bi + 1) == n_batches:
                pct = 100.0 * (bi + 1) / n_batches
                avg_loss = tot / (bi + 1)
                elapsed = time.time() - t0
                eta = (elapsed / (bi + 1)) * (n_batches - (bi + 1))
                log(f"  [ep {ep+1:2d}/{a.epochs:2d} | batch {bi+1:3d}/{n_batches} ({pct:3.0f}%)] loss: {avg_loss:.4f} | elapsed: {elapsed:.1f}s | eta: {eta:.1f}s")

        epoch_time = time.time() - t0
        log(f"Epoch {ep+1}/{a.epochs} complete in {epoch_time:.1f}s | Avg Loss: {tot/n_batches:.4f}")

        # Save latest checkpoint after every epoch to protect compute progress
        torch.save({"arch": a.arch, "ncls": ncls, "epoch": ep + 1, "state": model.state_dict()},
                   os.path.join(a.out, f"{a.arch}_latest.pth"))

        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            cmc, mAP = evaluate(model, q_dl, g_dl, dev)
            r1 = cmc[0]
            log(f"  >> [EVAL] Clean Rank-1: {r1*100:.2f}% | Rank-5: {cmc[4]*100:.2f}% | mAP: {mAP*100:.2f}%")
            if mAP >= best_map:
                best_map = mAP
                torch.save({"arch": a.arch, "ncls": ncls, "epoch": ep + 1, "mAP": mAP, "rank1": r1, "state": model.state_dict()},
                           target_ckpt)
                log(f"  >> [BEST] New best model saved to {target_ckpt} (mAP: {best_map*100:.2f}%)")

    total_time = time.time() - t_start
    log(f"\n=======================================================")
    log(f"Training Complete for {a.arch.upper()} in {total_time/60:.1f} mins | Best mAP: {best_map*100:.2f}%")
    log(f"Checkpoint saved to {target_ckpt}")
    log(f"=======================================================\n")


if __name__ == "__main__":
    main()

