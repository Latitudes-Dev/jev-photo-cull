"""Find image files under a folder. Directories are not followed through symlinks."""

from __future__ import annotations

import os
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"})


def discover_images(root: Path) -> list[Path]:
    """Return image paths relative to a walk of ``root``, sorted for stable group ids."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                found.append(path)
    return found
