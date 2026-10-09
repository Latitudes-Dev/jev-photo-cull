"""Orchestrate a read-only scan: measure, group, decide, write a report."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jev_photo_cull import __version__
from jev_photo_cull.analyze import analyze_paths
from jev_photo_cull.backends.clef import ClefBackend, credentials_present, load_dotenv, require_credentials
from jev_photo_cull.backends.heuristics_backend import HeuristicsBackend
from jev_photo_cull.backends.local import LocalBackend
from jev_photo_cull.cache import DecisionCache
from jev_photo_cull.combine import DEFAULT_MODEL_MARGIN, combine_keeper, combine_quality, singleton_keep
from jev_photo_cull.discover import discover_images
from jev_photo_cull.errors import ConfigError
from jev_photo_cull.grouping import GroupingConfig, group_records
from jev_photo_cull.heuristics import HeuristicConfig, judge
from jev_photo_cull.records import ImageRecord
from jev_photo_cull.report import export_keepers, write_reports
from jev_photo_cull.verdicts import KeeperVerdict, QualityVerdict

_SEVERITY = {"overexposed": 0, "too_dark": 1, "blurry": 2, "unknown": 3, "clean": 4}


def resolve_backend_name(name: str, environ: dict[str, str] | None = None) -> str:
    """``auto`` uses Clef when Cloudflare credentials are present, otherwise heuristics."""
    if name != "auto":
        return name
    return "clef" if credentials_present(environ) else "heuristics"


def ensure_output_outside_input(input_dir: Path, output_dir: Path) -> None:
    """Refuse an output directory inside the input tree so a scan cannot ingest its own copies."""
    source = input_dir.resolve()
    target = output_dir.resolve()
    if target == source or source in target.parents:
        raise ConfigError(
            "Refusing to write the output directory inside the input folder. "
            "Originals are read-only; choose an output path outside that tree."
        )


def run_scan(
    input_dir: Path,
    output_dir: Path,
    *,
    backend: str = "auto",
    model: str | None = None,
    near_hamming: int = 8,
    burst_hamming: int = 16,
    burst_seconds: float = 2.0,
    review_below: float = 0.60,
    model_margin: float = DEFAULT_MODEL_MARGIN,
    max_edge: int = 512,
    requests_per_minute: float = 30.0,
    timeout: float = 120.0,
    cache_enabled: bool = True,
    export_mode: str | None = None,
    local_url: str | None = None,
    local_model: str | None = None,
    local_protocol: str | None = None,
    local_image_format: str | None = None,
    environ: dict[str, str] | None = None,
    post: Any = None,
    sleep: Any = None,
    clock: Any = None,
    load_env_file: bool = False,
) -> dict[str, Any]:
    if load_env_file:
        load_dotenv()
    source = input_dir.expanduser().resolve()
    if not source.is_dir():
        raise ConfigError(f"Input folder does not exist: {source}")
    destination = output_dir.expanduser().resolve()
    ensure_output_outside_input(source, destination)
    chosen = resolve_backend_name(backend, environ)
    if chosen not in {"heuristics", "clef", "local"}:
        raise ConfigError("backend must be auto, heuristics, clef, or local")
    if chosen == "clef":
        require_credentials(environ)
    destination.mkdir(parents=True, exist_ok=True)

    grouping = GroupingConfig(
        near_hamming=near_hamming,
        burst_hamming=burst_hamming,
        burst_seconds=burst_seconds,
    )
    heuristic_config = HeuristicConfig()
    paths = discover_images(source)
    records = analyze_paths(paths, source, model_max_edge=max_edge)
    groups = group_records(records, grouping)

    heuristics = HeuristicsBackend(
        heuristic_config=heuristic_config,
        grouping_config=grouping,
        review_below=review_below,
    )
    heuristic_quality = heuristics.classify(records)
    heuristic_keepers = heuristics.choose_keepers(groups)

    model_quality: dict[str, QualityVerdict] | None = None
    model_keepers: dict[str, KeeperVerdict] | None = None
    model_name: str | None = None
    usage: dict[str, Any] = {"calls": 0, "neurons": 0.0, "input_tokens": 0}
    cache = DecisionCache(destination / "cache" / "decisions.json", enabled=cache_enabled)
    if chosen == "clef":
        clef = ClefBackend(
            model=model or "clef-flash",
            cache=cache,
            requests_per_minute=requests_per_minute,
            timeout=timeout,
            review_below=review_below,
            environ=environ,
            post=post,
            sleep=sleep,
            clock=clock,
        )
        model_quality = clef.classify(records)
        model_keepers = clef.choose_keepers(groups)
        model_name = clef.model_id
        usage = dict(clef.usage)
    elif chosen == "local":
        local = LocalBackend(
            cache=cache,
            url=local_url,
            model=local_model or model,
            protocol=local_protocol,
            image_format=local_image_format,
            requests_per_minute=requests_per_minute,
            timeout=timeout,
            review_below=review_below,
            environ=environ,
            post=post,
            sleep=sleep,
            heuristics=heuristics,
        )
        model_quality = local.classify(records)
        model_keepers = local.choose_keepers(groups)
        model_name = local.model_id
        usage = dict(local.usage)

    report = _assemble(
        source=source,
        destination=destination,
        backend=chosen,
        model_name=model_name,
        records=records,
        groups=groups,
        heuristic_quality=heuristic_quality,
        heuristic_keepers=heuristic_keepers,
        model_quality=model_quality,
        model_keepers=model_keepers,
        review_below=review_below,
        model_margin=model_margin,
        heuristic_config=heuristic_config,
        usage=usage,
    )
    thumbs = {record.relative_path: record.thumb_jpeg for record in records if record.thumb_jpeg}
    write_reports(report, destination, thumbs)
    if export_mode:
        export_keepers(report, destination, export_mode)
    return report


def _assemble(
    *,
    source: Path,
    destination: Path,
    backend: str,
    model_name: str | None,
    records: list[ImageRecord],
    groups: list,
    heuristic_quality: dict[str, QualityVerdict],
    heuristic_keepers: dict[str, KeeperVerdict],
    model_quality: dict[str, QualityVerdict] | None,
    model_keepers: dict[str, KeeperVerdict] | None,
    review_below: float,
    model_margin: float,
    heuristic_config: HeuristicConfig,
    usage: dict[str, Any],
) -> dict[str, Any]:
    by_path = {record.relative_path: record for record in records}
    images: list[dict[str, Any]] = []
    group_payloads = []
    for group in groups:
        heuristic_keeper = heuristic_keepers.get(group.id)
        model_keeper = None if model_keepers is None else model_keepers.get(group.id)
        keeper = (
            combine_keeper(
                heuristic_keeper,
                model_keeper,
                review_below=review_below,
                model_margin=model_margin,
            )
            if group.size > 1
            else None
        )
        combined_by_path = {}
        for member in group.members:
            record = by_path[member.relative_path]
            heuristic = heuristic_quality[record.relative_path]
            model = None if model_quality is None else model_quality.get(record.relative_path)
            combined_by_path[record.relative_path] = (
                record,
                heuristic,
                model,
                combine_quality(
                    heuristic,
                    model,
                    review_below=review_below,
                    model_margin=model_margin,
                ),
            )
        keeper_label = None
        if keeper and keeper.keeper_path:
            keeper_label = combined_by_path[keeper.keeper_path][3].label
        # A frame that is still blurry or badly exposed is not exported, even if it
        # is the least-bad member of its group. The group payload still names it.
        keeper_worth_keeping = keeper_label == "clean"
        for member in group.members:
            record, heuristic, model, combined = combined_by_path[member.relative_path]
            judgement = judge(record, heuristic_config)
            needs_review = combined.needs_review or bool(keeper and keeper.needs_review)
            if group.size == 1:
                keep = singleton_keep(combined.label, needs_review)
                keep_reason = "only frame in its group"
            elif keeper is None or keeper.abstained or keeper.keeper_path is None:
                keep = None
                keep_reason = "no confident keeper for this group"
                needs_review = True
            elif record.relative_path == keeper.keeper_path and keeper_worth_keeping:
                keep = True
                keep_reason = "suggested keeper for the group"
            elif record.relative_path == keeper.keeper_path:
                keep = False
                keep_reason = f"best frame in the group is still {keeper_label}"
                needs_review = True
            else:
                keep = False
                keep_reason = f"drop in favor of {keeper.keeper_path}"
            reasons = list(combined.reasons)
            if keeper is not None:
                reasons.extend(keeper.reasons)
            reasons.append(keep_reason)
            scores = {
                "sharpness": record.sharpness,
                "mean_luminance": record.mean_luminance,
                "luminance_std": record.luminance_std,
                "shadow_fraction": record.shadow_fraction,
                "highlight_fraction": record.highlight_fraction,
                "histogram": record.histogram,
                "quality_score": judgement.quality_score,
            }
            images.append(
                {
                    "path": record.relative_path,
                    "absolute_path": record.absolute_path,
                    "sha256": record.sha256,
                    "perceptual_hash": record.perceptual_hash_hex,
                    "group_id": group.id,
                    "group_size": group.size,
                    "group_kind": group.kind,
                    "keep": keep,
                    "label": combined.label,
                    "confidence": round(combined.confidence, 4),
                    "needs_review": needs_review,
                    "heuristic_label": heuristic.label,
                    "heuristic_confidence": round(heuristic.confidence, 4),
                    "model_label": None if model is None else model.label,
                    "model_confidence": None if model is None else round(model.confidence, 4),
                    "model_confidence_raw": (
                        None
                        if model is None or model.confidence_raw is None
                        else round(model.confidence_raw, 4)
                    ),
                    "probabilities": {key: round(value, 4) for key, value in combined.probabilities.items()},
                    "heuristic_probabilities": {
                        key: round(value, 4) for key, value in heuristic.probabilities.items()
                    },
                    "model_probabilities": None
                    if model is None
                    else {key: round(value, 4) for key, value in model.probabilities.items()},
                    "scores": scores,
                    "backend": backend,
                    "model": model_name,
                    "reasons": reasons,
                    "captured_at": record.captured_at.isoformat(timespec="seconds") if record.captured_at else None,
                    "width": record.width,
                    "height": record.height,
                    "error": record.error,
                    "file_size": record.file_size,
                }
            )
        suggested = keeper.keeper_path if keeper and not keeper.abstained and keeper_worth_keeping else None
        rejected = (
            keeper.keeper_path if keeper and not keeper.abstained and not keeper_worth_keeping else None
        )
        group_payloads.append(
            {
                "id": group.id,
                "kind": group.kind,
                "size": group.size,
                "keeper": suggested,
                "rejected_best": rejected,
                "abstained": suggested is None and group.size > 1,
                "members": [member.relative_path for member in group.members],
            }
        )

    images.sort(key=lambda row: (_SEVERITY.get(row["label"], 9), row["confidence"], row["path"]))
    by_label: dict[str, int] = {}
    for image in images:
        by_label[image["label"]] = by_label.get(image["label"], 0) + 1
    return {
        "version": 1,
        "tool": f"jev-photo-cull {__version__}",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_dir": str(source),
        "output_dir": str(destination),
        "backend": backend,
        "model": model_name,
        "model_margin": model_margin,
        "usage": {
            "calls": int(usage.get("calls", 0)),
            "neurons": usage.get("neurons", 0.0),
            "input_tokens": int(usage.get("input_tokens", 0)),
        },
        "image_count": len(images),
        "summary": {
            "groups": len(group_payloads),
            "multi_groups": sum(1 for group in group_payloads if group["size"] > 1),
            "needs_review": sum(1 for image in images if image["needs_review"]),
            "keep": sum(1 for image in images if image["keep"] is True),
            "by_label": by_label,
        },
        "groups": group_payloads,
        "images": images,
    }
