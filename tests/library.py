"""Synthetic photos generated in tests. Nothing here is a committed binary."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def sharp_scene(shift: int = 0) -> Image.Image:
    """High-frequency checker on a mid-gray field. Exposure sits near the middle."""
    image = Image.new("RGB", (320, 240), (148, 144, 136))
    draw = ImageDraw.Draw(image)
    for y in range(36, 204, 8):
        for x in range(28, 292, 8):
            dark = ((x // 8) + (y // 8)) % 2 == 0
            color = (16, 16, 16) if dark else (236, 236, 236)
            draw.rectangle((x, y, x + 7, y + 7), fill=color)
    draw.rectangle((12 + shift, 8, 70 + shift, 28), fill=(176, 48, 42))
    return image


def stripe_scene() -> Image.Image:
    image = Image.new("RGB", (300, 220), (132, 142, 150))
    draw = ImageDraw.Draw(image)
    for x in range(0, 300, 10):
        draw.rectangle((x, 0, x + 4, 219), fill=(24, 28, 32))
    return image


def write_library(root: Path) -> dict[str, Path]:
    """Build the fixture folder used by the scan test. Returns paths by role."""
    root.mkdir(parents=True, exist_ok=True)
    nested = root / "nested"
    nested.mkdir()
    paths: dict[str, Path] = {}

    sharp = sharp_scene()
    buffer = _png_bytes(sharp)
    paths["clean"] = root / "clean.png"
    paths["clean_copy"] = root / "clean-copy.png"
    paths["clean"].write_bytes(buffer)
    paths["clean_copy"].write_bytes(buffer)

    paths["blurry"] = root / "blurry.png"
    sharp.filter(ImageFilter.GaussianBlur(radius=9)).save(paths["blurry"], format="PNG")

    paths["dark"] = root / "dark.png"
    sharp.point(lambda pixel: int(pixel * 0.05)).save(paths["dark"], format="PNG")

    paths["blown"] = root / "blown.png"
    sharp.point(lambda pixel: min(255, 250 + int(pixel * 0.02))).save(paths["blown"], format="PNG")

    paths["other"] = root / "other.png"
    stripe_scene().save(paths["other"], format="PNG")

    paths["nested"] = nested / "leaf.webp"
    stripe_scene().transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(paths["nested"], format="WEBP", quality=90)

    burst = sharp_scene(shift=4)
    paths["burst_a"] = root / "burst-a.jpg"
    paths["burst_b"] = root / "burst-b.jpg"
    _save_jpeg(burst, paths["burst_a"], "2026:05:01 12:00:00")
    _save_jpeg(sharp_scene(shift=8), paths["burst_b"], "2026:05:01 12:00:01")

    paths["flat_black"] = root / "flat-black.png"
    Image.new("RGB", (80, 60), (0, 0, 0)).save(paths["flat_black"], format="PNG")
    paths["flat_white"] = root / "flat-white.png"
    Image.new("RGB", (80, 60), (255, 255, 255)).save(paths["flat_white"], format="PNG")

    paths["broken"] = root / "broken.jpg"
    paths["broken"].write_bytes(b"this is not a jpeg")
    paths["heic"] = nested / "junk.heic"
    paths["heic"].write_bytes(b"not a heic either")
    (root / "notes.txt").write_text("ignore me\n", encoding="utf-8")

    wide = Image.new("RGB", (60, 20), (180, 20, 20))
    draw = ImageDraw.Draw(wide)
    draw.rectangle((0, 0, 29, 19), fill=(10, 10, 180))
    exif = Image.Exif()
    exif[274] = 6  # orientation: display rotated, so width and height swap
    paths["oriented"] = root / "oriented.jpg"
    wide.save(paths["oriented"], format="JPEG", quality=95, exif=exif)
    return paths


def _png_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _save_jpeg(image: Image.Image, path: Path, timestamp: str) -> None:
    exif = Image.Exif()
    exif[306] = timestamp
    exif[36867] = timestamp
    image.save(path, format="JPEG", quality=92, exif=exif)
