import torch, torch.nn as nn, torch.nn.functional as F


def batch_hard_triplet(feat, labels, margin=0.3):
    """For each anchor: hardest positive and hardest negative in the batch."""
    d = torch.cdist(feat, feat)
    same = labels[:, None] == labels[None, :]
    hardest_pos = (d * same).max(1)[0]
    hardest_neg = (d + 1e5 * same).min(1)[0]
    return F.relu(hardest_pos - hardest_neg + margin).mean()


class ReIDLoss(nn.Module):
    """ID loss (label-smoothed CE) + batch-hard triplet loss."""
    def __init__(self, margin=0.3, smooth=0.1):
        super().__init__()
        self.margin, self.ce = margin, nn.CrossEntropyLoss(label_smoothing=smooth)

    def forward(self, feat, logits, labels):
        return self.ce(logits, labels) + batch_hard_triplet(feat, labels, self.margin)
