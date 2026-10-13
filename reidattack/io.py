import torch
from .models import ReIDNet


def load_model(path, device="cuda"):
    ck = torch.load(path, map_location=device)
    m = ReIDNet(ck["arch"], ck["ncls"], pretrained=False)
    m.load_state_dict(ck["state"])
    return m.to(device).eval()
