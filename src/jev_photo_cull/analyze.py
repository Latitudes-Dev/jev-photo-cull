"""Pass 1 measurements: dHash, exposure, Laplacian sharpness. Reads files, never writes them."""

from __future__ import annotations

import hashlib
import io
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from jev_photo_cull.records import ImageRecord

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIF_ENABLED = True
except ImportError:  # pragma: no cover - depends on the optional extra
    HEIF_ENABLED = False

_DATETIME_FORMATS = ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S")


def file_sha256(path: Path) -> tuple[str, int, bytes]:
    digest = hashlib.sha256()
    size = 0
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
            chunks.append(block)
    return digest.hexdigest(), size, b"".join(chunks)


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """64-bit difference hash. Neighbor comparisons survive mild JPEG recompression."""
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    raw_pixels = gray.get_flattened_data() if hasattr(gray, "get_flattened_data") else gray.getdata()
    pixels = list(raw_pixels)
    width = hash_size + 1
    value = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * width + col]
            right = pixels[row * width + col + 1]
            value = (value << 1) | (1 if left > right else 0)
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _parse_exif_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def capture_time(image: Image.Image) -> datetime | None:
    exif = image.getexif()
    if not exif:
        return None
    values: list[object] = [exif.get(36867), exif.get(306)]
    try:
        ifd = exif.get_ifd(0x8769)
    except (KeyError, AttributeError, ValueError):
        ifd = {}
    if ifd:
        values.append(ifd.get(36867))
        values.append(ifd.get(306))
    for value in values:
        parsed = _parse_exif_datetime(value)
        if parsed is not None:
            return parsed
    return None


def orient(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation. Returns the original image if the tag is missing or unusable."""
    try:
        transposed = ImageOps.exif_transpose(image)
    except (ValueError, OSError):
        return image
    return transposed if transposed is not None else image


def _fit(image: Image.Image, max_edge: int) -> Image.Image:
    if max(image.size) <= max_edge:
        return image
    fitted = image.copy()
    fitted.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return fitted


def jpeg_bytes(image: Image.Image, max_edge: int, quality: int) -> bytes:
    fitted = _fit(image, max_edge)
    if fitted.mode != "RGB":
        fitted = fitted.convert("RGB")
    buffer = io.BytesIO()
    fitted.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _laplacian_variance(gray: np.ndarray) -> float:
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * center
    )
    return float(lap.var())


def normalized_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance after a percentile stretch, so a dark sharp frame is not called blurry."""
    low, high = np.percentile(gray, [1, 99])
    if float(high - low) < 8.0:
        return 0.0
    stretched = np.clip((gray - low) * (255.0 / float(high - low)), 0.0, 255.0)
    return _laplacian_variance(stretched)


def measure(image: Image.Image, analysis_max_edge: int = 1024) -> dict[str, object]:
    rgb_image = image.convert("RGB")
    analyzed = _fit(rgb_image, analysis_max_edge)
    gray = np.asarray(analyzed.convert("L"), dtype=np.float32)
    rgb = np.asarray(analyzed, dtype=np.float32)
    histogram, _edges = np.histogram(gray, bins=16, range=(0, 256))
    return {
        "width": image.width,
        "height": image.height,
        "mean_luminance": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "shadow_fraction": float((gray <= 8).mean()),
        "highlight_fraction": float((gray >= 247).mean()),
        "rgb_mean": tuple(float(channel) for channel in rgb.mean(axis=(0, 1))),
        "histogram": [int(count) for count in histogram.tolist()],
        "sharpness": normalized_sharpness(gray),
        "perceptual_hash": dhash(analyzed),
    }


def analyze_file(
    path: Path,
    root: Path,
    *,
    analysis_max_edge: int = 1024,
    model_max_edge: int = 512,
    thumb_max_edge: int = 280,
) -> ImageRecord:
    """Hash and measure one file. Decode failures become ``error`` records, not exceptions."""
    digest, size, raw = file_sha256(path)
    relative = path.relative_to(root).as_posix()
    record = ImageRecord(
        relative_path=relative,
        absolute_path=str(path.resolve()),
        sha256=digest,
        file_size=size,
    )
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"} and not HEIF_ENABLED:
        record.error = (
            "HEIC/HEIF support is not installed. Install the optional extra: "
            "pip install 'jev-photo-cull[heif]' (needs libheif)."
        )
        return record
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            upright = orient(image)
            taken = capture_time(image)
            stats = measure(upright, analysis_max_edge=analysis_max_edge)
            record.thumb_jpeg = jpeg_bytes(upright, thumb_max_edge, quality=70)
            record.model_jpeg = jpeg_bytes(upright, model_max_edge, quality=80)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        record.error = f"unreadable image: {exc.__class__.__name__}"
        return record

    record.captured_at = taken
    record.width = int(stats["width"])
    record.height = int(stats["height"])
    record.mean_luminance = float(stats["mean_luminance"])
    record.luminance_std = float(stats["luminance_std"])
    record.shadow_fraction = float(stats["shadow_fraction"])
    record.highlight_fraction = float(stats["highlight_fraction"])
    record.rgb_mean = stats["rgb_mean"]  # type: ignore[assignment]
    record.histogram = stats["histogram"]  # type: ignore[assignment]
    record.sharpness = float(stats["sharpness"])
    record.perceptual_hash = int(stats["perceptual_hash"])
    return record


def analyze_paths(
    paths: list[Path],
    root: Path,
    *,
    analysis_max_edge: int = 1024,
    model_max_edge: int = 512,
) -> list[ImageRecord]:
    return [
        analyze_file(
            path,
            root,
            analysis_max_edge=analysis_max_edge,
            model_max_edge=model_max_edge,
        )
        for path in paths
    ]
