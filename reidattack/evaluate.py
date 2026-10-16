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


def cmc_map(qf, gf, q_pids, g_pids, q_cams, g_cams, max_rank=10, chunk_size=500):
    """qf, gf: L2-normalised torch tensors. Returns (cmc[max_rank], mAP).
    Chunks queries in batches of 500 to keep memory footprint under 32 MB on Windows.
    """
    cmcs, aps = [], []
    for start in range(0, len(qf), chunk_size):
        end = min(start + chunk_size, len(qf))
        dist_chunk = (1 - qf[start:end] @ gf.T).numpy()
        for i in range(len(dist_chunk)):
            order = np.argsort(dist_chunk[i])
            same_id = g_pids[order] == q_pids[start + i]
            junk = same_id & (g_cams[order] == q_cams[start + i])
            matches = same_id[~junk].astype(np.int32)
            if matches.sum() == 0:
                continue
            c = matches.cumsum()
            c[c > 1] = 1
            cmcs.append(c[:max_rank])
            prec = matches.cumsum() / (np.arange(len(matches)) + 1.0)
            aps.append((prec * matches).sum() / matches.sum())
    return np.mean(cmcs, axis=0), float(np.mean(aps))


def cmc_map_np(dist, q_pids, g_pids, q_cams, g_cams, max_rank=10):
    cmcs, aps = [], []
    for i in range(len(dist)):
        order = np.argsort(dist[i])
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


def top1_pid(qf, gf, g_pids, q_pids, q_cams, g_cams, chunk_size=500):
    """Top-1 retrieved identity per query (same-id-same-cam junk removed), for impersonation success."""
    out = []
    for start in range(0, len(qf), chunk_size):
        end = min(start + chunk_size, len(qf))
        dist_chunk = (1 - qf[start:end] @ gf.T).numpy()
        for i in range(len(dist_chunk)):
            d = dist_chunk[i].copy()
            d[(g_pids == q_pids[start + i]) & (g_cams == q_cams[start + i])] = 1e9
            out.append(g_pids[d.argmin()])
    return np.array(out)

