"""Package the complete Phase 1 reviewer bundle into an ultra-compact zip archive.

Includes:
- All core engine code (reidattack/)
- All execution and plotting scripts (scripts/)
- Unit and smoke tests (tests/)
- Interactive demo (demo/)
- Kaggle pipeline notebooks and runners (notebooks/)
- Complete Phase 1 Report, Reviewer Guide, README, and licenses
- Auto-filled presentation docs and slides (docs/)
- All 100% verified empirical results, JSONs, facts, and figures (results/)
- Trained patch artifact (results/patch.pth)

Excludes:
- .git repository
- Large backbone weights (300+ MB), which are externally referenced
- __pycache__, scratch, temporary files
"""
from pathlib import Path
import zipfile
import sys

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = ROOT / "reid_attack_anatomy_phase1.zip"

INCLUDE_EXTS = {
    ".py", ".md", ".json", ".png", ".pth", ".txt", ".yml", ".yaml", ".pptx", ".gitignore", "LICENSE"
}

EXCLUDE_DIRS = {
    ".git", "__pycache__", "kaggle_outputs", "scratch", "checkpoints", ".gemini", ".idea", ".vscode"
}

EXCLUDE_FILES = {
    "densenet121.pth",
    "resnet50.pth",
    "resnet50_adv_trained_eps8.pth",
    "vit_small.pth",
    "reid-attack-anatomy.zip",
    "reid-attack-anatomy_v2_real.zip",
    "reid_attack_anatomy_phase1.zip"
}

def should_include(rel_path: Path) -> bool:
    parts = rel_path.parts
    for ex in EXCLUDE_DIRS:
        if ex in parts:
            return False
    if rel_path.name in EXCLUDE_FILES:
        return False
    if rel_path.is_file():
        if rel_path.name in ("LICENSE", ".gitignore"):
            return True
        if rel_path.suffix.lower() in INCLUDE_EXTS:
            return True
    return False

def package_zip():
    print(f"Building {OUTPUT_ZIP.name}...")
    files_to_pack = []
    total_uncompressed = 0

    for path in sorted(ROOT.rglob("*")):
        if path.is_file():
            rel = path.relative_to(ROOT)
            if should_include(rel):
                files_to_pack.append((path, rel))
                total_uncompressed += path.stat().st_size

    print(f"Found {len(files_to_pack)} files to pack ({total_uncompressed / (1024*1024):.2f} MB uncompressed)")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for abs_path, rel_path in files_to_pack:
            # Use forward slashes inside zip for cross-platform compatibility (Linux/Mac/Windows)
            arcname = str(rel_path).replace("\\", "/")
            zf.write(abs_path, arcname)

    compressed_size = OUTPUT_ZIP.stat().st_size
    print(f"Successfully packaged {OUTPUT_ZIP.name}:")
    print(f"  Compressed size:   {compressed_size / (1024*1024):.2f} MB ({compressed_size:,} bytes)")
    print(f"  Uncompressed size: {total_uncompressed / (1024*1024):.2f} MB")
    print(f"  Compression ratio: {compressed_size / total_uncompressed * 100:.1f}%")
    print(f"  Total items:       {len(files_to_pack)}")

if __name__ == "__main__":
    package_zip()
