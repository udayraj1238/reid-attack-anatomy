"""Universal adversarial torso patch optimised with Expectation over Transformation (EOT).

ONE patch is trained for the whole dataset (universal) and pasted on the torso.
During optimisation the patch is randomly scaled / rotated / shifted and the image is
randomly brightened + noised, so the patch keeps working under viewpoint and lighting change.
Loss needs no labels: make the patched embedding dissimilar from the clean embedding.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class UniversalPatch(nn.Module):
    def __init__(self, ph=64, pw=48, center=(0.0, -0.05)):
        super().__init__()
        self.raw = nn.Parameter(torch.rand(1, 3, ph, pw))
        self.ph, self.pw = ph, pw
        self.center = center        # (cx, cy) in [-1,1]; cy slightly above middle = torso

    @property
    def patch(self):
        return self.raw.clamp(0, 1)

    def forward(self, x, eot=True):
        B, _, H, W = x.shape
        dev = x.device
        cx = torch.full((B,), self.center[0], device=dev)
        cy = torch.full((B,), self.center[1], device=dev)
        s = torch.ones(B, device=dev)
        ang = torch.zeros(B, device=dev)
        if eot:
            s = torch.empty(B, device=dev).uniform_(0.8, 1.25)
            ang = torch.empty(B, device=dev).uniform_(-math.radians(20), math.radians(20))
            cx = cx + torch.empty(B, device=dev).uniform_(-0.15, 0.15)
            cy = cy + torch.empty(B, device=dev).uniform_(-0.15, 0.15)
        cos, sin = torch.cos(ang), torch.sin(ang)
        # output pixel (u,v in [-1,1]) -> patch coordinate, rotation done in pixel units
        kx = (W / 2) / (s * self.pw / 2)
        ky = (H / 2) / (s * self.pw / 2)
        kx2 = (W / 2) / (s * self.ph / 2)
        ky2 = (H / 2) / (s * self.ph / 2)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = cos * kx
        theta[:, 0, 1] = sin * ky
        theta[:, 0, 2] = -(cos * cx * (W / 2) + sin * cy * (H / 2)) / (s * self.pw / 2)
        theta[:, 1, 0] = -sin * kx2
        theta[:, 1, 1] = cos * ky2
        theta[:, 1, 2] = -(-sin * cx * (W / 2) + cos * cy * (H / 2)) / (s * self.ph / 2)
        grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
        patch = self.patch.expand(B, -1, -1, -1)
        warped = F.grid_sample(patch, grid, padding_mode="zeros", align_corners=False)
        mask = F.grid_sample(torch.ones(B, 1, self.ph, self.pw, device=dev), grid,
                             padding_mode="zeros", align_corners=False)
        out = x * (1 - mask) + warped * mask
        if eot:
            out = out * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
            out = out + 0.02 * torch.randn_like(out)
        return out.clamp(0, 1)


def total_variation(p):
    return (p[..., 1:, :] - p[..., :-1, :]).abs().mean() + (p[..., :, 1:] - p[..., :, :-1]).abs().mean()


def train_patch(model, loader, epochs=5, lr=0.05, tv=0.1, device="cuda", ph=64, pw=48, log=print):
    import time
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    patcher = UniversalPatch(ph, pw).to(device)
    opt = torch.optim.Adam(patcher.parameters(), lr=lr)
    n_batches = len(loader)
    log(f"Starting Universal Patch Training: {epochs} epochs, {n_batches} batches/epoch, TV weight={tv}")

    for ep in range(epochs):
        tot, n = 0.0, 0
        t0 = time.time()
        for bi, (x, _, _) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                clean = model.embed(x)
            sim = (model.embed(patcher(x, eot=True)) * clean).sum(1).mean()
            loss = sim + tv * total_variation(patcher.patch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += sim.item() * len(x)
            n += len(x)

            if (bi + 1) % 25 == 0 or (bi + 1) == n_batches:
                pct = 100.0 * (bi + 1) / n_batches
                elapsed = time.time() - t0
                log(f"  [patch ep {ep+1:2d}/{epochs} | batch {bi+1:3d}/{n_batches} ({pct:3.0f}%)] mean sim: {tot/n:.3f} | {elapsed:.1f}s")

        epoch_time = time.time() - t0
        log(f"[patch] Epoch {ep+1}/{epochs} complete in {epoch_time:.1f}s | Mean Cosine Sim: {tot/n:.3f} (lower = stronger attack)\n")
    return patcher
