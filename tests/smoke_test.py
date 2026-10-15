"""Run this FIRST (CPU is fine, ~1-2 min, no dataset needed):  python tests/smoke_test.py
Checks the maths + plumbing on random weights. It does NOT produce any result or number for the report."""
import os, sys, tempfile, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.models import ReIDNet
from reidattack.losses import ReIDLoss
from reidattack.attacks import fgsm, pgd, mifgsm, evade_objective, impersonate_objective, logit_objective
from reidattack.patch import UniversalPatch
from reidattack.evaluate import cmc_map_np
from reidattack.viz_utils import gradcam_resnet, rank_gallery
from reidattack.demo_core import craft, search
from reidattack import results

torch.manual_seed(0)
m = ReIDNet("resnet50", num_classes=10, pretrained=False).eval()
x = torch.rand(4, 3, 256, 128)
e = m.embed(x)
assert e.shape == (4, 2048) and torch.allclose(e.norm(dim=1), torch.ones(4), atol=1e-4); print("embed ok")
m.train(); f, lg = m(x); loss = ReIDLoss()(f, lg, torch.tensor([0, 0, 1, 1])); loss.backward(); m.eval(); print("loss ok", float(loss))

clean = m.embed(x).detach(); eps = 8 / 255
for name, fn in [("fgsm", lambda o: fgsm(o, x, eps)), ("pgd", lambda o: pgd(o, x, eps, 5)), ("mifgsm", lambda o: mifgsm(o, x, eps, 5))]:
    xa = fn(evade_objective(m, clean))
    assert (xa - x).abs().max() <= eps + 1e-5 and xa.min() >= 0 and xa.max() <= 1
    sim = (m.embed(xa) * clean).sum(1).mean().item()
    assert sim < 0.999, f"{name} did not move the embedding"
    print(f"{name} ok (budget respected, embedding moved)")
tgt = clean.roll(1, 0)
before = (m.embed(x) * tgt).sum(1).mean().item()
xa = pgd(impersonate_objective(m, tgt), x, eps, 5)
assert (m.embed(xa) * tgt).sum(1).mean().item() > before; print("impersonation moves towards target ok")
pgd(logit_objective(m, torch.tensor([0, 1, 2, 3])), x, eps, 2); print("logit attack ok")

p = UniversalPatch().eval()
out = p(x, eot=True); assert out.shape == x.shape
(m.embed(out) * clean).sum().backward(); assert p.raw.grad is not None and p.raw.grad.abs().sum() > 0
assert not torch.equal(p(x, eot=False), x), "patch did not change the image"
print("patch + EOT gradient flow ok")

# metric: perfect ranking + junk rule
d = np.array([[0.0, 0.1, 0.5, 0.6, 0.7, 0.2], [0.9, 0.8, 0.3, 0.1, 0.5, 0.6]])
cmc, mAP = cmc_map_np(d, np.array([1, 2]), np.array([1, 1, 2, 2, 3, 1]), np.array([1, 1]), np.array([1, 2, 2, 3, 1, 3]), max_rank=3)
assert np.allclose(cmc, 1) and abs(mAP - 1) < 1e-9; print("CMC/mAP + junk rule ok")

# Grad-CAM + demo plumbing on random weights
gf = torch.nn.functional.normalize(torch.randn(30, 2048), dim=1)
gp = np.repeat(np.arange(6), 5); gc = np.tile(np.arange(5), 6)
cam, sc = gradcam_resnet(m, x[:2], gf[:2]); assert cam.shape == (2, 256, 128) and 0 <= cam.min() and cam.max() <= 1
idx, sims, q = search(m, x[:1], gf, gp, gc, q_pid=0, q_cam=0, k=5); assert len(idx) == 5
for atk in ("none", "FGSM", "PGD-10", "MI-FGSM"):
    xa = craft(m, x[:1], atk, 8); assert xa.shape == x[:1].shape
assert (craft(m, x[:1], "Universal patch", 8, p) != x[:1]).any(); print("gradcam + demo core ok")

# results module refuses to invent data
try:
    results.load_json(os.path.join(tempfile.gettempdir(), "definitely_missing.json")); raise AssertionError("should have stopped")
except SystemExit:
    print("missing-results guard ok")
print("ALL SMOKE TESTS PASSED")
