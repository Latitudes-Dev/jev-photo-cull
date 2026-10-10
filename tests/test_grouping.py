from datetime import datetime, timedelta

from jev_photo_cull.grouping import GroupingConfig, group_records, link_kind
from jev_photo_cull.records import ImageRecord


def _record(
    name: str,
    *,
    sha: str | None = None,
    phash: int = 0,
    mean: float = 128.0,
    std: float = 40.0,
    rgb: tuple[float, float, float] | None = None,
    captured_at: datetime | None = None,
    error: str | None = None,
) -> ImageRecord:
    return ImageRecord(
        relative_path=name,
        absolute_path="/virtual/" + name,
        sha256=sha or name,
        file_size=10,
        perceptual_hash=None if error else phash,
        mean_luminance=None if error else mean,
        luminance_std=None if error else std,
        rgb_mean=None if error else (rgb or (mean, mean, mean)),
        sharpness=None if error else 200.0,
        shadow_fraction=0.0,
        highlight_fraction=0.0,
        captured_at=captured_at,
        error=error,
        width=8,
        height=8,
    )


def test_flat_black_and_white_are_not_duplicates():
    black = _record("black.png", phash=0, mean=0, std=0, rgb=(0, 0, 0))
    white = _record("white.png", phash=0, mean=255, std=0, rgb=(255, 255, 255))
    assert link_kind(black, white, GroupingConfig()) is None
    groups = group_records([black, white])
    assert [group.size for group in groups] == [1, 1]


def test_identical_bytes_group_even_when_hashes_differ():
    left = _record("a.png", sha="same", phash=0)
    right = _record("b.png", sha="same", phash=0xFFFF)
    assert link_kind(left, right, GroupingConfig()) == "identical"
    group = group_records([right, left])[0]
    assert group.kind == "identical"
    assert group.id == "g0001"
    assert [member.relative_path for member in group.members] == ["a.png", "b.png"]


def test_burst_window_links_moderate_hashes_only_with_timestamps():
    start = datetime(2026, 5, 1, 12, 0, 0)
    # Hamming distance 10 sits between the near threshold (8) and the burst threshold (16).
    left = _record("a.jpg", phash=0, captured_at=start)
    right = _record("b.jpg", phash=(1 << 10) - 1, captured_at=start + timedelta(seconds=1))
    config = GroupingConfig()
    assert link_kind(left, right, config) == "burst"
    untimed = _record("c.jpg", phash=(1 << 10) - 1, captured_at=None)
    assert link_kind(left, untimed, config) is None
    late = _record("d.jpg", phash=(1 << 10) - 1, captured_at=start + timedelta(seconds=30))
    assert link_kind(left, late, config) is None
    groups = group_records([left, right], config)
    assert groups[0].kind == "burst"
    assert groups[0].size == 2


def test_unreadable_files_stay_singletons():
    broken = _record("broken.jpg", error="unreadable image")
    other = _record("ok.png", sha="same")
    broken.sha256 = "same"
    groups = group_records([broken, other])
    assert sorted(group.size for group in groups) == [1, 1]
