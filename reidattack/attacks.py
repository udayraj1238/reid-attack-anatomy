"""Gradient attacks written from scratch, all in [0,1] pixel space with an L-inf budget.

One generic engine, three classic attacks (they differ only in steps / momentum / start):
  FGSM     : 1 step of size eps
  PGD      : many small steps, random start, project back into the eps-ball
  MI-FGSM  : PGD + momentum on the (L1-normalised) gradient  -> more transferable

The part that is specific to ReID is the OBJECTIVE, not the optimiser:
  evade        : push the adversarial embedding away from the clean embedding
  impersonate  : pull the adversarial embedding towards a chosen identity's embedding
  logit        : the 'classification-style' attack on the ID-classifier head (baseline to compare)
"""
import torch, torch.nn.functional as F


def evade_objective(model, clean_feat):
    return lambda xa: -(model.embed(xa) * clean_feat).sum(1)           # minimise cosine sim to self


def impersonate_objective(model, target_feat):
    return lambda xa: (model.embed(xa) * target_feat).sum(1)           # maximise cosine sim to target


def logit_objective(model, labels):
    return lambda xa: F.cross_entropy(model.logits(xa), labels, reduction="none")


def attack(objective, x, eps=8 / 255, steps=10, alpha=None, momentum=0.0, random_start=False):
    """Maximise objective(x_adv) subject to ||x_adv - x||_inf <= eps and x_adv in [0,1]."""
    alpha = alpha if alpha is not None else eps / max(steps, 1) * (2.5 if steps > 1 else 1.0)
    x = x.detach()
    xa = x.clone()
    if random_start:
        xa = (xa + torch.empty_like(xa).uniform_(-eps, eps)).clamp(0, 1)
    g = torch.zeros_like(x)
    for _ in range(steps):
        xa.requires_grad_(True)
        obj = objective(xa).sum()
        grad, = torch.autograd.grad(obj, xa)
        if momentum > 0:                                               # MI-FGSM
            grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
            g = momentum * g + grad
            step = g.sign()
        else:
            step = grad.sign()
        xa = xa.detach() + alpha * step
        xa = torch.min(torch.max(xa, x - eps), x + eps).clamp(0, 1)
    return xa.detach()


def fgsm(objective, x, eps):
    return attack(objective, x, eps, steps=1, alpha=eps)


def pgd(objective, x, eps, steps=10):
    return attack(objective, x, eps, steps=steps, random_start=True)


def mifgsm(objective, x, eps, steps=10, mu=1.0):
    return attack(objective, x, eps, steps=steps, momentum=mu)


ATTACKS = {"fgsm": lambda o, x, e, s: fgsm(o, x, e),
           "pgd": lambda o, x, e, s: pgd(o, x, e, s),
           "mifgsm": lambda o, x, e, s: mifgsm(o, x, e, s)}
