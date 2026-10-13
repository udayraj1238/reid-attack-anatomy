"""Run this FIRST (CPU is fine, ~1 min, no dataset needed). If it passes, the pipeline is sane.
python tests/smoke_test.py"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reidattack.models import ReIDNet
from reidattack.losses import ReIDLoss
from reidattack.attacks import fgsm, pgd, mifgsm, evade_objective, impersonate_objective, logit_objective
from reidattack.patch import UniversalPatch

torch.manual_seed(0)
m = ReIDNet("resnet50", num_classes=10, pretrained=False).eval()
x = torch.rand(4, 3, 256, 128)
e = m.embed(x)
assert e.shape == (4, 2048) and torch.allclose(e.norm(dim=1), torch.ones(4), atol=1e-4); print("embed ok")

m.train(); f, lg = m(x); loss = ReIDLoss()(f, lg, torch.tensor([0, 0, 1, 1])); loss.backward(); m.eval(); print("loss ok", float(loss))

clean = m.embed(x).detach()
eps = 8 / 255
for name, fn in [("fgsm", lambda o: fgsm(o, x, eps)), ("pgd", lambda o: pgd(o, x, eps, 5)), ("mifgsm", lambda o: mifgsm(o, x, eps, 5))]:
    obj = evade_objective(m, clean)
    xa = fn(obj)
    assert (xa - x).abs().max() <= eps + 1e-5 and xa.min() >= 0 and xa.max() <= 1
    sim = (m.embed(xa) * clean).sum(1).mean().item()
    print(f"{name}: cos-sim to clean {sim:.3f} (should be < 1.0)"); assert sim < 0.999
tgt = clean.roll(1, 0)
xa = pgd(impersonate_objective(m, tgt), x, eps, 5)
print("impersonate ok", (m.embed(xa) * tgt).sum(1).mean().item())
pgd(logit_objective(m, torch.tensor([0, 1, 2, 3])), x, eps, 2); print("logit attack ok")

p = UniversalPatch().eval()
out = p(x, eot=True); assert out.shape == x.shape
(m.embed(out) * clean).sum().backward(); assert p.raw.grad is not None and p.raw.grad.abs().sum() > 0
print("patch + EOT gradient flow ok")
print("ALL SMOKE TESTS PASSED")
