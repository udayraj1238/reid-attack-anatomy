import argparse, os, time, torch
from .data import get_market
from .models import ReIDNet
from .losses import ReIDLoss
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
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--eval_every", type=int, default=10)
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
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        warm = min(1.0, (ep + 1) / 5)                      # 5-epoch linear warm-up
        for g in opt.param_groups:
            g["lr"] = a.lr * warm * (0.1 ** sum(ep >= m for m in milestones))
        for x, y, _ in train_dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                f, logits = model(x)
                loss = crit(f.float(), logits.float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()
        print(f"epoch {ep+1}/{a.epochs} loss {tot/len(train_dl):.3f} ({time.time()-t0:.0f}s)", flush=True)
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            r1, mAP = evaluate(model, q_dl, g_dl, dev)
            print(f"  >> clean Rank-1 {r1*100:.1f}  mAP {mAP*100:.1f}", flush=True)
            torch.save({"arch": a.arch, "ncls": ncls, "state": model.state_dict()},
                       os.path.join(a.out, f"{a.arch}.pth"))


if __name__ == "__main__":
    main()
