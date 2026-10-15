"""Single source of truth for result files + provenance.

RULES THIS PROJECT FOLLOWS
  * Every table / figure / slide number is computed from a JSON file produced by a real run.
  * If the JSON is missing, scripts STOP with an error. They never draw placeholder data.
  * Every real run writes an entry to results/MANIFEST.json (time, git commit, checkpoint hashes).

This module must NOT import torch at module level (plot scripts run on a laptop without a GPU).
"""
import hashlib, json, os, subprocess, time

MODEL_ORDER = ["resnet50", "densenet121", "vit_small"]


def results_dir():
    return os.environ.get("REID_RESULTS", "results")


def rpath(name):
    d = results_dir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


def require(path, hint=""):
    if not os.path.exists(path):
        raise SystemExit(
            f"\n[ERROR] Required result file not found: {path}\n"
            f"  Figures/tables here are built ONLY from real experiment output - no placeholders are ever drawn.\n"
            f"  {hint}\n")
    return path


def load_json(path, hint=""):
    require(path, hint)
    with open(path) as f:
        data = json.load(f)
    if not data:
        raise SystemExit(f"\n[ERROR] {path} is empty - the experiment that should fill it did not finish.\n")
    return data


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def stamp(tag, sources=(), ckpts=(), extra=None):
    """Record how a result/figure was produced in results/MANIFEST.json."""
    path = rpath("MANIFEST.json")
    try:
        manifest = json.load(open(path))
    except Exception:
        manifest = {}
    entry = {"time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "git_commit": _git_commit(),
             "sources": [os.path.basename(s) for s in sources],
             "checkpoints": {os.path.basename(c): sha12(c) for c in ckpts if os.path.exists(c)}}
    try:
        import torch
        entry["torch"] = torch.__version__
        entry["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    if extra:
        entry.update(extra)
    manifest[tag] = entry
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1)


def order_models(names):
    names = list(names)
    return sorted(names, key=lambda n: (MODEL_ORDER.index(n) if n in MODEL_ORDER else 99, n))
