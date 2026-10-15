import torch
from .models import ReIDNet


def load_model(path, device="cuda"):
    """Load a trained ReIDNet model from a checkpoint.
    
    Supports PyTorch 2.6+ by explicitly disabling weights_only restrictions
    for checkpoint dictionaries containing numpy scalar evaluation metrics.
    """
    try:
        ck = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch versions without weights_only argument
        ck = torch.load(path, map_location=device)

    m = ReIDNet(ck["arch"], ck["ncls"], pretrained=False)
    m.load_state_dict(ck["state"])
    return m.to(device).eval()
