"""Feature extraction + the standard Market-1501 protocol (CMC / mAP).
Protocol: for each query, gallery images with the SAME id AND SAME camera are ignored."""
import numpy as np, torch


@torch.no_grad()
def extract(model, loader, device="cuda"):
    model.eval()
    feats, pids, cams = [], [], []
    for x, y, c in loader:
        feats.append(model.embed(x.to(device)).cpu())
        pids.append(y), cams.append(c)
    return torch.cat(feats), torch.cat(pids).numpy(), torch.cat(cams).numpy()


def cmc_map(qf, gf, q_pids, g_pids, q_cams, g_cams, max_rank=10):
    """qf, gf: L2-normalised torch tensors. Returns (cmc[max_rank], mAP)."""
    return cmc_map_np((1 - qf @ gf.T).numpy(), q_pids, g_pids, q_cams, g_cams, max_rank)


def cmc_map_np(dist, q_pids, g_pids, q_cams, g_cams, max_rank=10):
    idx = np.argsort(dist, axis=1)
    cmcs, aps = [], []
    for i in range(len(dist)):
        order = idx[i]
        same_id = g_pids[order] == q_pids[i]
        junk = same_id & (g_cams[order] == q_cams[i])
        matches = same_id[~junk].astype(np.int32)
        if matches.sum() == 0:
            continue
        c = matches.cumsum()
        c[c > 1] = 1
        cmcs.append(c[:max_rank])
        prec = matches.cumsum() / (np.arange(len(matches)) + 1.0)
        aps.append((prec * matches).sum() / matches.sum())
    return np.mean(cmcs, axis=0), float(np.mean(aps))


def top1_pid(qf, gf, g_pids, q_pids, q_cams, g_cams):
    """Top-1 retrieved identity per query (same-id-same-cam junk removed), for impersonation success."""
    dist = (1 - qf @ gf.T).numpy()
    out = []
    for i in range(len(qf)):
        d = dist[i].copy()
        d[(g_pids == q_pids[i]) & (g_cams == q_cams[i])] = 1e9
        out.append(g_pids[d.argmin()])
    return np.array(out)
