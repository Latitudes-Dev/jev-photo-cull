"""End-to-end scan on generated fixtures. Originals must stay byte-for-byte identical."""

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from jev_photo_cull.combine import combine_quality
from jev_photo_cull.errors import ConfigError
from jev_photo_cull.scan import resolve_backend_name, run_scan
from jev_photo_cull.verdicts import QualityVerdict
from tests.library import write_library


def _snapshot(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _by_path(report: dict) -> dict[str, dict]:
    return {image["path"]: image for image in report["images"]}


def test_heuristics_scan_flags_defects_and_groups_duplicates(tmp_path: Path):
    library = tmp_path / "photos"
    output = tmp_path / "out"
    write_library(library)
    before = _snapshot(library)

    def explode(*args, **kwargs):
        raise AssertionError("heuristics scan must not use the network")

    original = urllib.request.urlopen
    urllib.request.urlopen = explode
    try:
        report = run_scan(library, output, backend="heuristics", requests_per_minute=0)
    finally:
        urllib.request.urlopen = original

    assert _snapshot(library) == before
    rows = _by_path(report)
    assert "notes.txt" not in rows
    assert rows["blurry.png"]["label"] == "blurry"
    assert rows["dark.png"]["label"] == "too_dark"
    assert rows["blown.png"]["label"] == "overexposed"
    assert rows["clean.png"]["label"] == "clean"
    assert rows["other.png"]["label"] == "clean"
    assert rows["nested/leaf.webp"]["label"] == "clean"
    assert rows["clean.png"]["group_id"] == rows["clean-copy.png"]["group_id"]
    assert rows["flat-black.png"]["group_id"] != rows["flat-white.png"]["group_id"]
    assert rows["broken.jpg"]["label"] == "unknown"
    assert rows["broken.jpg"]["needs_review"] is True
    assert rows["nested/junk.heic"]["label"] == "unknown"
    assert rows["oriented.jpg"]["width"] == 20
    assert rows["oriented.jpg"]["height"] == 60
    assert rows["burst-a.jpg"]["captured_at"] == "2026-05-01T12:00:00"

    group_id = rows["clean.png"]["group_id"]
    members = [image for image in report["images"] if image["group_id"] == group_id]
    assert len(members) >= 2
    keepers = [image for image in members if image["keep"] is True]
    assert len(keepers) == 1
    assert keepers[0]["label"] == "clean"
    for name in ("blurry.png", "dark.png", "blown.png"):
        assert rows[name]["keep"] is not True
        assert rows[name]["heuristic_label"] == rows[name]["label"]
        assert rows[name]["backend"] == "heuristics"
        assert rows[name]["model"] is None
        assert "histogram" in rows[name]["scores"]

    assert (output / "report.json").is_file()
    assert (output / "report.csv").is_file()
    html = (output / "review.html").read_text(encoding="utf-8")
    assert "blurry.png" in html
    assert "data:image/jpeg;base64," in html
    assert "https://" not in html
    assert "http://" not in html
    assert 'data-filter="too_dark"' in html
    assert "Duplicates and bursts" in html
    csv_text = (output / "report.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("path,sha256,perceptual_hash")
    loaded = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert loaded["backend"] == "heuristics"
    assert loaded["image_count"] == len(rows)


def test_export_copies_keepers_and_leaves_originals(tmp_path: Path):
    library = tmp_path / "photos"
    output = tmp_path / "out"
    write_library(library)
    before = _snapshot(library)
    report = run_scan(library, output, backend="heuristics", export_mode="copy")
    assert _snapshot(library) == before
    exported = (output / "keepers.txt").read_text(encoding="utf-8").splitlines()
    assert exported
    for relative in exported:
        source = library / relative
        copied = output / "keepers" / relative
        assert copied.is_file()
        assert copied.read_bytes() == source.read_bytes()
        assert source.exists()
    rows = _by_path(report)
    assert "blurry.png" not in exported
    assert rows["blurry.png"]["absolute_path"].startswith(str(library))


def test_export_symlink_and_output_inside_input_is_refused(tmp_path: Path):
    library = tmp_path / "photos"
    output = tmp_path / "linked"
    write_library(library)
    before = _snapshot(library)
    run_scan(library, output, backend="heuristics", export_mode="symlink")
    listing = (output / "keepers.txt").read_text(encoding="utf-8").splitlines()
    link = output / "keepers" / listing[0]
    assert link.is_symlink()
    assert link.resolve() == (library / listing[0]).resolve()
    assert _snapshot(library) == before

    try:
        run_scan(library, library / "out", backend="heuristics")
    except ConfigError as exc:
        assert "outside" in str(exc).lower() or "inside" in str(exc).lower()
    else:
        raise AssertionError("output inside the input tree should be refused")
    assert not (library / "out").exists()


def test_cli_scan_and_missing_clef_credentials(tmp_path: Path, monkeypatch):
    library = tmp_path / "photos"
    output = tmp_path / "out"
    write_library(library)
    env = os.environ.copy()
    env.pop("CLOUDFLARE_ACCOUNT_ID", None)
    env.pop("CLOUDFLARE_API_TOKEN", None)
    env.pop("CLOUDFLARE_AUTH_TOKEN", None)
    completed = subprocess.run(
        [sys.executable, "-m", "jev_photo_cull", "scan", str(library), "--out", str(output), "--backend", "heuristics"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output / "review.html").is_file()

    failed = subprocess.run(
        [sys.executable, "-m", "jev_photo_cull", "scan", str(library), "--out", str(tmp_path / "clef-out"), "--backend", "clef"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert failed.returncode == 2
    assert "CLOUDFLARE_API_TOKEN" in failed.stderr
    assert "No request was sent" in failed.stderr
    assert not (tmp_path / "clef-out" / "report.json").exists()


def test_auto_backend_follows_credentials():
    assert resolve_backend_name("auto", {}) == "heuristics"
    assert resolve_backend_name("auto", {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}) == "clef"
    assert resolve_backend_name("heuristics", {"CLOUDFLARE_ACCOUNT_ID": "a", "CLOUDFLARE_API_TOKEN": "t"}) == "heuristics"


def test_disagreement_is_flagged_for_review():
    heuristic = QualityVerdict(label="blurry", probabilities={"blurry": 0.8}, confidence=0.8)
    model = QualityVerdict(label="clean", probabilities={"clean": 0.7}, confidence=0.7)
    combined = combine_quality(heuristic, model, review_below=0.6)
    assert combined.needs_review is True
    assert combined.label == "blurry"
    agreed = combine_quality(heuristic, QualityVerdict(label="blurry", probabilities={"blurry": 0.9}, confidence=0.9), review_below=0.6)
    assert agreed.label == "blurry"
    assert agreed.needs_review is False
