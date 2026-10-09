"""Command line entry point for ``jev-photo-cull``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jev_photo_cull import __version__
from jev_photo_cull.errors import ConfigError, PhotoCullError
from jev_photo_cull.scan import run_scan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-photo-cull",
        description=(
            "Suggest which photos to keep. Reads an input folder and writes a report "
            "somewhere else. Originals are never modified, moved, or deleted."
        ),
    )
    parser.add_argument("--version", action="version", version=f"jev-photo-cull {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan a folder and write CSV, JSON, and an HTML review page")
    scan.add_argument("folder", type=Path, help="Folder of photos (subfolders included)")
    scan.add_argument("--out", type=Path, required=True, help="Output directory, outside the input folder")
    scan.add_argument(
        "--backend",
        choices=("auto", "heuristics", "clef", "local"),
        default="auto",
        help="auto uses clef when Cloudflare credentials are set, otherwise heuristics (default)",
    )
    scan.add_argument(
        "--model",
        choices=("clef-flash", "clef"),
        default=None,
        help="Clef model for --backend clef (default clef-flash). With --backend local, overrides the local model name.",
    )
    scan.add_argument("--near-hamming", type=int, default=8, help="dHash distance for near-duplicates")
    scan.add_argument("--burst-hamming", type=int, default=16, help="Looser dHash distance when EXIF times are close")
    scan.add_argument("--burst-seconds", type=float, default=2.0, help="EXIF time window for bursts")
    scan.add_argument("--review-below", type=float, default=0.60, help="Confidence under this is flagged for review")
    scan.add_argument("--max-edge", type=int, default=512, help="Longest edge of JPEGs sent to a decision model")
    scan.add_argument("--requests-per-minute", type=float, default=30.0, help="Client-side cap on model requests")
    scan.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    scan.add_argument("--no-cache", action="store_true", help="Do not read or write the decision cache")
    scan.add_argument(
        "--export-keep-list",
        nargs="?",
        const="copy",
        choices=("copy", "symlink"),
        help="Copy (default) or symlink suggested keepers into the output directory. Never moves originals.",
    )
    scan.add_argument("--local-url", default=None, help="Decision server for --backend local")
    scan.add_argument("--local-model", default=None, help="Model name sent to the local server")
    scan.add_argument("--local-protocol", choices=("systemone", "imajev"), default=None)
    scan.add_argument(
        "--local-image-format",
        choices=("raw", "data-url", "content-type"),
        default=None,
        help="How systemone image bytes are encoded. Ollama wants raw base64.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "scan":
        parser.error(f"unknown command {args.command}")
    try:
        report = run_scan(
            args.folder,
            args.out,
            backend=args.backend,
            model=args.model,
            near_hamming=args.near_hamming,
            burst_hamming=args.burst_hamming,
            burst_seconds=args.burst_seconds,
            review_below=args.review_below,
            max_edge=args.max_edge,
            requests_per_minute=args.requests_per_minute,
            timeout=args.timeout,
            cache_enabled=not args.no_cache,
            export_mode=args.export_keep_list,
            local_url=args.local_url,
            local_model=args.local_model,
            local_protocol=args.local_protocol,
            local_image_format=args.local_image_format,
            load_env_file=True,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except PhotoCullError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = report["summary"]
    labels = ", ".join(f"{name} {count}" for name, count in sorted(summary["by_label"].items()))
    print(
        f"{report['image_count']} images, {summary['groups']} groups, "
        f"backend {report['backend']}"
        + (f" ({report['model']})" if report["model"] else ""),
        file=sys.stderr,
    )
    print(f"labels: {labels or 'none'}", file=sys.stderr)
    print(f"needs review: {summary['needs_review']}", file=sys.stderr)
    print(f"wrote {report['output_dir']}/report.json", file=sys.stderr)
    print(f"wrote {report['output_dir']}/report.csv", file=sys.stderr)
    print(f"wrote {report['output_dir']}/review.html", file=sys.stderr)
    return 0
