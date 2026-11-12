"""Download the local field reader once, into this project's model cache."""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "tracking"))

from gridiron_tracking.registration.local import DEFAULT_MODEL, MODEL_FILES, cache_dir, existing_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face MLX model repository")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("this MLX reader requires an Apple Silicon Mac")
    os.chdir(ROOT)
    from gridiron_tracking.config import load_env
    load_env()
    model = args.model.removeprefix("mlx:")
    if existing := existing_model_path(model):
        print(f"Ready: reusing {existing}; no download needed", flush=True)
        return
    # Use the ordinary file download, avoiding a second Xet chunk cache on disk.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import HfApi, snapshot_download

    dest = cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(model, files_metadata=True)
    files = [f for f in info.siblings if any(fnmatch(f.rfilename, p) for p in MODEL_FILES)]
    try:
        existing = snapshot_download(model, cache_dir=dest, allow_patterns=MODEL_FILES,
                                     local_files_only=True)
        # A partially downloaded snapshot can already exist; check every file.
        remaining = sum(f.size or 0 for f in files if not (Path(existing) / f.rfilename).is_file())
    except FileNotFoundError:
        remaining = sum(f.size or 0 for f in files)
    free = shutil.disk_usage(dest).free
    if remaining and free < remaining + 3 * 1024**3:
        parser.error(f"need {remaining / 1024**3:.2f} GiB for the model plus 3 GiB free; "
                     f"only {free / 1024**3:.2f} GiB available")
    print(f"Downloading {model} ({remaining / 1024**3:.2f} GiB remaining) into {dest}", flush=True)
    path = snapshot_download(model, cache_dir=dest, allow_patterns=MODEL_FILES)
    print(f"Ready: {path}\nRun: .venv/bin/python -m gridiron_tracking render nfl_sf_sea", flush=True)


if __name__ == "__main__":
    main()
