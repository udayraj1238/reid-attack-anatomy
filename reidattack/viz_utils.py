"""Helpers shared by the figure scripts and the demo. Everything here runs on REAL checkpoints/data."""
import os
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms as T
from .data import find_market_root, parse_dir, ReIDDataset, H, W
from .evaluate import extract

_to_tensor = T.Compose([T.Resize((H, W)), T.ToTensor()])


def load_split(root):
    root = find_market_root(root)
    return parse_dir(os.path.join(root, "query")), parse_dir(os.path.join(root, "bounding_box_test"))


def load_tensor(path):
    return _to_tensor(Image.open(path).convert("RGB"))


def extract_items(model, items, device="cuda", bs=128, workers=2):
    dl = DataLoader(ReIDDataset(items), batch_size=bs, num_workers=workers)
    return extract(model, dl, device)


def rank_gallery(qf, gf, g_pids, g_cams, q_pid=None, q_cam=None, k=5):
    """Indices of the top-k gallery images for ONE query embedding.
    If the query identity/camera is known, same-id-same-camera images are removed (Market-1501 junk rule)."""
    sims = gf @ qf
    if q_pid is not None:
        sims = sims.clone()
        sims[torch.from_numpy((g_pids == q_pid) & (g_cams == q_cam))] = -2
    return torch.topk(sims, k).indices.numpy()


def identity_centroid(gf, g_pids, g_cams, pid, cam):
    """Mean gallery embedding of identity `pid` seen by cameras other than `cam`."""
    m = (g_pids == pid) & (g_cams != cam)
    if m.sum() == 0:
        m = g_pids == pid
    return F.normalize(gf[torch.from_numpy(m)].mean(0), dim=0)


def gradcam_resnet(model, x, centroid):
    """Grad-CAM for retrieval (ResNet-50 only).
    Score = cosine similarity between the query embedding and the TRUE identity's gallery centroid, so the
    heat-map shows which pixels support (or no longer support) a correct match. Returns (cam[B,H,W] in 0..1, score[B])."""
    assert model.arch == "resnet50", "Grad-CAM helper is written for the ResNet-50 body"
    model.eval()
    x = x.detach().requires_grad_(True)          # keeps the graph alive even if weights are frozen
    xn = (x - model.mean) / model.std
    act = model.body(xn)
    emb = F.normalize(model.bn(model.pool(act).flatten(1)), dim=1)
    cen = centroid if centroid.dim() == 2 else centroid[None].expand(len(x), -1)
    score = (emb * cen).sum(1)
    grads, = torch.autograd.grad(score.sum(), act)
    w = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
    mn, mx = cam.amin(dim=(2, 3), keepdim=True), cam.amax(dim=(2, 3), keepdim=True)
    cam = (cam - mn) / (mx - mn + 1e-8)
    return cam[:, 0].detach().cpu(), score.detach().cpu()
