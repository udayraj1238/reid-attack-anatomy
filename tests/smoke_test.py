"""Run this FIRST (CPU is fine, ~1-2 min, no dataset needed):  python tests/smoke_test.py
Checks maths + plumbing on random weights. It does NOT produce any result or number for the report.

Design note: a randomly-initialised ResNet-50 gives nearly identical embeddings for every input (cosine ~0.9999), so
absolute thresholds are meaningless there. We therefore (a) test the attack ENGINE on a tiny non-collapsed embedding model,
comparing each attack with random noise of the same budget, and (b) use the real ResNet-50 only for plumbing checks."""
import os, sys, tempfile
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.models import ReIDNet
from reidattack.losses import ReIDLoss
from reidattack.attacks import fgsm, pgd, mifgsm, evade_objective, impersonate_objective, logit_objective
from reidattack.patch import UniversalPatch
from reidattack.evaluate import cmc_map_np
from reidattack.viz_utils import gradcam_resnet
from reidattack.demo_core import craft, search
from reidattack import results

torch.manual_seed(0)


class TinyEmbed(torch.nn.Module):
    """Deliberately simple differentiable embedding: random linear map of a 16x8 average-pooled image."""
    def __init__(self, d=64, ncls=10):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(d, 3 * 16 * 8) / (3 * 16 * 8) ** 0.5)
        self.C = torch.nn.Parameter(torch.randn(ncls, d))

    def feat(self, x):
        return F.adaptive_avg_pool2d(x, (16, 8)).flatten(1) @ self.W.T

    def embed(self, x):
        return F.normalize(self.feat(x), dim=1)

    def logits(self, x):
        return self.feat(x) @ self.C.T


# ---------------- (a) attack engine on a non-collapsed model ----------------
t = TinyEmbed().eval()
x = F.interpolate(torch.rand(6, 3, 16, 8), size=(256, 128), mode="nearest")   # low-frequency images -> distinct embeddings
clean = t.embed(x).detach()
off_diag = (clean @ clean.T)[~torch.eye(6, dtype=torch.bool)].mean().item()
assert off_diag < 0.95, f"test model is collapsed (mean cross-image cosine {off_diag:.3f})"
eps = 16 / 255
noise = (x + eps * torch.sign(torch.randn_like(x))).clamp(0, 1)
sim_noise = (t.embed(noise) * clean).sum(1).mean().item()
for name, fn in [("fgsm", lambda o: fgsm(o, x, eps)), ("pgd", lambda o: pgd(o, x, eps, 5)), ("mifgsm", lambda o: mifgsm(o, x, eps, 5))]:
    xa = fn(evade_objective(t, clean))
    assert (xa - x).abs().max() <= eps + 1e-5 and xa.min() >= 0 and xa.max() <= 1, f"{name} broke the eps budget / pixel range"
    sim = (t.embed(xa) * clean).sum(1).mean().item()
    assert sim < 1 - 1e-4, f"{name} barely moved the embedding (cos={sim:.6f})"
    assert sim < sim_noise, f"{name} is not stronger than random noise (cos {sim:.6f} vs noise {sim_noise:.6f})"
    print(f"{name} ok: cos to clean {sim:.5f} < random-noise {sim_noise:.5f}")
tgt = clean.roll(1, 0)
before = (clean * tgt).sum(1).mean().item()
after = (t.embed(mifgsm(impersonate_objective(t, tgt), x, eps, 5)) * tgt).sum(1).mean().item()
assert after > before + 1e-3, f"impersonation did not move towards the target ({before:.4f} -> {after:.4f})"
print(f"impersonation ok: cos to target {before:.4f} -> {after:.4f}")
labels = t.logits(x).argmax(1)
ce0 = F.cross_entropy(t.logits(x), labels).item()
ce1 = F.cross_entropy(t.logits(pgd(logit_objective(t, labels), x, eps, 5)), labels).item()
assert ce1 > ce0, "logit attack did not increase the loss"; print(f"logit attack ok: CE {ce0:.3f} -> {ce1:.3f}")

# ---------------- (b) real ResNet-50 architecture: plumbing only ----------------
m = ReIDNet("resnet50", num_classes=10, pretrained=False).eval()
xr = torch.rand(4, 3, 256, 128)
e = m.embed(xr)
assert e.shape == (4, 2048) and torch.allclose(e.norm(dim=1), torch.ones(4), atol=1e-4); print("resnet50 embed shape/norm ok")
m.train(); f, lg = m(xr); loss = ReIDLoss()(f, lg, torch.tensor([0, 0, 1, 1])); loss.backward(); m.eval()
print("loss + backward ok", float(loss.detach()))
cr = m.embed(xr).detach()
for name, fn in [("fgsm", lambda o: fgsm(o, xr, eps)), ("pgd", lambda o: pgd(o, xr, eps, 2)), ("mifgsm", lambda o: mifgsm(o, xr, eps, 2))]:
    xa = fn(evade_objective(m, cr))
    assert (xa - xr).abs().max() <= eps + 1e-5 and xa.min() >= 0 and xa.max() <= 1
    ea = m.embed(xa); assert torch.isfinite(ea).all() and (ea - cr).abs().max() > 1e-6
print("attacks run on resnet50 (finite, budget respected, embedding changes)")

p = UniversalPatch().eval()
out = p(xr, eot=True); assert out.shape == xr.shape
(m.embed(out) * cr).sum().backward(); assert p.raw.grad is not None and p.raw.grad.abs().sum() > 0
assert not torch.equal(p(xr, eot=False), xr), "patch did not change the image"
print("patch + EOT gradient flow ok")

d = np.array([[0.0, 0.1, 0.5, 0.6, 0.7, 0.2], [0.9, 0.8, 0.3, 0.1, 0.5, 0.6]])
cmc, mAP = cmc_map_np(d, np.array([1, 2]), np.array([1, 1, 2, 2, 3, 1]), np.array([1, 1]), np.array([1, 2, 2, 3, 1, 3]), max_rank=3)
assert np.allclose(cmc, 1) and abs(mAP - 1) < 1e-9; print("CMC/mAP + junk rule ok")

gf = F.normalize(torch.randn(30, 2048), dim=1)
gp = np.repeat(np.arange(6), 5); gc = np.tile(np.arange(5), 6)
cam, sc = gradcam_resnet(m, xr[:2], gf[:2]); assert cam.shape == (2, 256, 128) and cam.min() >= 0 and cam.max() <= 1
idx, sims, q = search(m, xr[:1], gf, gp, gc, q_pid=0, q_cam=0, k=5); assert len(idx) == 5
for atk in ("none", "FGSM", "PGD-10", "MI-FGSM"):
    assert craft(m, xr[:1], atk, 8).shape == xr[:1].shape
assert (craft(m, xr[:1], "Universal patch", 8, p) != xr[:1]).any(); print("gradcam + demo core ok")

try:
    results.load_json(os.path.join(tempfile.gettempdir(), "definitely_missing.json")); raise AssertionError("should have stopped")
except SystemExit:
    print("missing-results guard ok")
print("ALL SMOKE TESTS PASSED")
