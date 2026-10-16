"""reidattack: adversarial attacks on person re-identification, built from scratch."""
try:
    import torch

    # Compatibility shim for PyTorch 2.4+ and Python 3.13 with torchvision
    _orig_register_fake = getattr(torch.library, "register_fake", None)
    if _orig_register_fake is not None:
        def _safe_register_fake(op_name, *args, **kwargs):
            dec = _orig_register_fake(op_name, *args, **kwargs)
            def wrapper(func):
                try:
                    return dec(func)
                except Exception:
                    return func
            return wrapper
        torch.library.register_fake = _safe_register_fake
except Exception:
    pass

