"""Package the complete Phase 1 submission into the smallest possible zip file.

Excludes heavy binary weights, dataset directories, git history, and caches,
while retaining 100% of the code, scripts, configurations, publication figures,
reports, guides, and presentation documents.
"""
import os
import zipfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ZIP = os.path.join(ROOT_DIR, "reid_attack_anatomy_phase1.zip")

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "checkpoints",
    "data",
    "kaggle_outputs",
    "scratch",
    ".ipynb_checkpoints",
    ".idea",
    ".vscode"
}

EXCLUDE_EXTENSIONS = {
    ".pth",
    ".pt",
    ".pyc",
    ".zip",
    ".tar",
    ".gz"
}


def should_include(rel_path):
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts:
        if part in EXCLUDE_DIRS:
            return False
    _, ext = os.path.splitext(rel_path)
    if ext.lower() in EXCLUDE_EXTENSIONS:
        return False
    return True


def create_zip():
    print(f"Creating compact Phase 1 submission package: {OUTPUT_ZIP}")
    file_count = 0
    total_uncompressed = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, dirs, files in os.walk(ROOT_DIR):
            # Prune excluded directories in-place
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in sorted(files):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, ROOT_DIR)

                if should_include(rel_path):
                    zipf.write(full_path, arcname=rel_path)
                    file_count += 1
                    total_uncompressed += os.path.getsize(full_path)
                    print(f"  + Added: {rel_path} ({os.path.getsize(full_path):,} bytes)")

    zip_size = os.path.getsize(OUTPUT_ZIP)
    print("\n" + "=" * 65)
    print(f"Phase 1 Zip Created Successfully!")
    print(f"File Name: {os.path.basename(OUTPUT_ZIP)}")
    print(f"Total Files Included: {file_count}")
    print(f"Uncompressed Size: {total_uncompressed / (1024*1024):.2f} MB ({total_uncompressed:,} bytes)")
    print(f"Final Compressed Size: {zip_size / (1024*1024):.2f} MB ({zip_size:,} bytes)")
    print(f"Compression Ratio: {(1 - zip_size / total_uncompressed) * 100:.1f}% space saved")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    create_zip()
