"""Market-1501 loading. Images are returned in [0,1] pixel space (no mean/std here);
normalisation lives inside the model so attacks can work directly on pixels."""
import os, re, glob, random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms as T
from PIL import Image

PATTERN = re.compile(r"([-\d]+)_c(\d)")
H, W = 256, 128


def find_market_root(root):
    """Kaggle nests the dataset in odd ways; find the folder containing bounding_box_train."""
    for dp, dn, _ in os.walk(root):
        if "bounding_box_train" in dn and "query" in dn:
            return dp
    raise FileNotFoundError(f"Market-1501 folders not found under {root}")


def parse_dir(d):
    items = []
    for p in sorted(glob.glob(os.path.join(d, "*.jpg"))):
        m = PATTERN.search(os.path.basename(p))
        pid, cam = int(m.group(1)), int(m.group(2))
        if pid == -1:  # junk images
            continue
        items.append((p, pid, cam))
    return items


class ReIDDataset(Dataset):
    def __init__(self, items, train=False, relabel=None):
        self.items = items
        self.train = train
        self.relabel = relabel
        if train:
            self.tf = T.Compose([
                T.Resize((H, W)), T.RandomHorizontalFlip(),
                T.Pad(10), T.RandomCrop((H, W)), T.ToTensor(),
                T.RandomErasing(p=0.5, value=0)])
        else:
            self.tf = T.Compose([T.Resize((H, W)), T.ToTensor()])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, pid, cam = self.items[i]
        img = self.tf(Image.open(p).convert("RGB"))
        label = self.relabel[pid] if self.relabel else pid
        return img, label, cam


class PKSampler(Sampler):
    """P identities x K images per batch (needed for batch-hard triplet loss)."""
    def __init__(self, items, P=16, K=4):
        self.P, self.K = P, K
        self.index = defaultdict(list)
        for i, (_, pid, _) in enumerate(items):
            self.index[pid].append(i)
        self.pids = list(self.index)
        self.length = (len(self.pids) // P) * P * K

    def __iter__(self):
        pids = self.pids[:]
        random.shuffle(pids)
        batch = []
        for pid in pids:
            idx = self.index[pid]
            idx = random.sample(idx, self.K) if len(idx) >= self.K else random.choices(idx, k=self.K)
            batch += idx
            if len(batch) == self.P * self.K:
                yield from batch
                batch = []

    def __len__(self):
        return self.length


def get_market(root, P=16, K=4, workers=2):
    root = find_market_root(root)
    train = parse_dir(os.path.join(root, "bounding_box_train"))
    query = parse_dir(os.path.join(root, "query"))
    gallery = parse_dir(os.path.join(root, "bounding_box_test"))
    pids = sorted({pid for _, pid, _ in train})
    relabel = {pid: i for i, pid in enumerate(pids)}
    train_ds = ReIDDataset(train, train=True, relabel=relabel)
    train_dl = DataLoader(train_ds, batch_size=P * K, sampler=PKSampler(train, P, K),
                          num_workers=workers, drop_last=True, pin_memory=True)
    q_dl = DataLoader(ReIDDataset(query), batch_size=128, num_workers=workers)
    g_dl = DataLoader(ReIDDataset(gallery), batch_size=128, num_workers=workers)
    return train_dl, q_dl, g_dl, len(pids)
