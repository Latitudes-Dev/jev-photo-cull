"""CSV, JSON, and a single-file HTML review page. All writes stay in the output directory."""

from __future__ import annotations

import csv
import html
import io
import json
import shutil
from pathlib import Path
from typing import Any

SEVERITY = {"overexposed": 0, "too_dark": 1, "blurry": 2, "unknown": 3, "clean": 4}

_FILTERS = (
    ("all", "All"),
    ("duplicates", "Duplicates"),
    ("review", "Needs review"),
    ("clean", "Clean"),
    ("blurry", "Blurry"),
    ("too_dark", "Too dark"),
    ("overexposed", "Overexposed"),
    ("unknown", "Unknown"),
)


def write_reports(report: dict[str, Any], output_dir: Path, thumbs: dict[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(report, output_dir / "report.csv")
    (output_dir / "review.html").write_text(render_html(report, thumbs), encoding="utf-8")


def _write_csv(report: dict[str, Any], path: Path) -> None:
    fields = [
        "path",
        "sha256",
        "perceptual_hash",
        "group_id",
        "group_size",
        "group_kind",
        "keep",
        "label",
        "confidence",
        "needs_review",
        "heuristic_label",
        "heuristic_confidence",
        "model_label",
        "model_confidence",
        "model_confidence_raw",
        "probabilities",
        "heuristic_probabilities",
        "model_probabilities",
        "scores",
        "backend",
        "model",
        "reasons",
        "captured_at",
        "width",
        "height",
        "error",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for image in report["images"]:
        row = dict(image)
        for key in ("probabilities", "heuristic_probabilities", "model_probabilities", "scores"):
            row[key] = json.dumps(image.get(key), sort_keys=True)
        row["reasons"] = " | ".join(image.get("reasons") or [])
        row["keep"] = _csv_keep(image.get("keep"))
        writer.writerow(row)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _csv_keep(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def export_keepers(report: dict[str, Any], output_dir: Path, mode: str) -> list[str]:
    """Copy or symlink suggested keepers into the output dir. Inputs are never moved."""
    if mode not in {"copy", "symlink"}:
        raise ValueError("export mode must be copy or symlink")
    destination_root = output_dir / "keepers"
    destination_root.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    for image in report["images"]:
        if image.get("keep") is not True:
            continue
        source = Path(image["absolute_path"])
        destination = destination_root / image["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.exists():
            destination.unlink()
        if mode == "symlink":
            destination.symlink_to(source)
        else:
            shutil.copy2(source, destination)
        exported.append(image["path"])
    listing = "\n".join(exported)
    (output_dir / "keepers.txt").write_text(listing + ("\n" if listing else ""), encoding="utf-8")
    return exported


def render_html(report: dict[str, Any], thumbs: dict[str, bytes]) -> str:
    summary = report["summary"]
    counts = " · ".join(
        f"{name.replace('_', ' ')} {summary['by_label'].get(name, 0)}"
        for name in ("clean", "blurry", "too_dark", "overexposed", "unknown")
    )
    filters = "\n".join(
        f'<button type="button" data-filter="{key}" class="{"active" if key == "all" else ""}">{html.escape(label)}</button>'
        for key, label in _FILTERS
    )
    groups_html = _groups_html(report, thumbs)
    cards_html = "\n".join(_card_html(image, thumbs, keeper=image.get("keep") is True) for image in report["images"])
    model = report.get("model") or "none"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<title>Photo cull review</title>
<style>
:root {{
  color-scheme: light;
  --ink: #1c1915;
  --muted: #5e584f;
  --line: #e4ddd2;
  --paper: #f7f4ee;
  --card: #fffdf9;
  --accent: #1f6b4a;
  --warn: #8a5a12;
  --bad: #8d2f2f;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font: 15px/1.45 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  color: var(--ink);
  background: var(--paper);
}}
header, main {{ max-width: 1180px; margin: 0 auto; padding: 24px 20px; }}
header {{ padding-bottom: 8px; }}
h1 {{ font-weight: 600; font-size: 28px; margin: 0 0 6px; letter-spacing: -0.02em; }}
h2 {{ font-size: 18px; margin: 28px 0 12px; }}
.sub {{ color: var(--muted); margin: 0; }}
.filters {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0; }}
button {{
  font: inherit;
  border: 1px solid var(--line);
  background: var(--card);
  color: var(--ink);
  border-radius: 999px;
  padding: 6px 12px;
  cursor: pointer;
}}
button.active {{ background: var(--ink); color: white; border-color: var(--ink); }}
.dup {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px 14px 16px;
  margin-bottom: 14px;
}}
.dup h3 {{ margin: 0 0 10px; font-size: 15px; font-weight: 600; }}
.row, .grid {{ display: grid; gap: 12px; }}
.row {{ grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }}
.grid {{ grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }}
.card {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}
.card.keeper {{ outline: 3px solid var(--accent); outline-offset: -3px; }}
.card img, .placeholder {{
  width: 100%;
  aspect-ratio: 4 / 3;
  object-fit: cover;
  background: #ece7df;
  display: block;
}}
.placeholder {{
  display: grid;
  place-items: center;
  color: var(--muted);
  font-size: 13px;
}}
.meta {{ padding: 8px 10px 10px; }}
.file {{ display: block; font-size: 13px; word-break: break-all; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }}
.badge {{
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 11px;
  letter-spacing: 0.02em;
  border-radius: 999px;
  padding: 2px 7px;
  background: #efeae2;
}}
.badge.clean, .badge.keep {{ background: #e5f4ec; color: var(--accent); }}
.badge.blurry, .badge.too_dark, .badge.overexposed, .badge.drop {{ background: #f8e8e6; color: var(--bad); }}
.badge.unknown, .badge.review {{ background: #f8efd8; color: var(--warn); }}
.reasons {{ color: var(--muted); font-size: 12px; margin: 6px 0 0; }}
.empty {{ color: var(--muted); }}
</style>
</head>
<body>
<header>
  <h1>Photo cull</h1>
  <p class="sub">{html.escape(str(report["image_count"]))} photos · {html.escape(str(summary["groups"]))} groups · {html.escape(counts)}</p>
  <p class="sub">Backend {html.escape(str(report["backend"]))} · model {html.escape(str(model))} · needs review {html.escape(str(summary["needs_review"]))}</p>
  <p class="sub">Originals were not moved or modified. Suggested keepers are highlighted.</p>
  <div class="filters">{filters}</div>
</header>
<main>
  <section id="groups">
    <h2>Duplicates and bursts</h2>
    {groups_html}
  </section>
  <section id="grid">
    <h2>All photos, worst first</h2>
    <div class="grid">
      {cards_html}
    </div>
  </section>
</main>
<script>
const buttons = document.querySelectorAll("[data-filter]");
function matches(card, filter) {{
  if (filter === "all") return true;
  if (filter === "review") return card.dataset.review === "1";
  if (filter === "duplicates") return Number(card.dataset.size) > 1;
  return card.dataset.label === filter;
}}
buttons.forEach((button) => {{
  button.addEventListener("click", () => {{
    buttons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    const filter = button.dataset.filter;
    document.querySelectorAll("#grid .card").forEach((card) => {{
      card.hidden = !matches(card, filter);
    }});
    document.querySelectorAll("#groups .dup").forEach((group) => {{
      const cards = [...group.querySelectorAll(".card")];
      let any = false;
      cards.forEach((card) => {{
        const ok = filter === "all" || filter === "duplicates" || matches(card, filter);
        card.hidden = !ok;
        any = any || ok;
      }});
      group.hidden = !any;
    }});
    document.getElementById("grid").hidden = filter === "duplicates";
  }});
}});
</script>
</body>
</html>
"""


def _groups_html(report: dict[str, Any], thumbs: dict[str, bytes]) -> str:
    blocks = [group for group in report["groups"] if group["size"] > 1]
    if not blocks:
        return '<p class="empty">No duplicate or burst groups.</p>'
    by_path = {image["path"]: image for image in report["images"]}
    parts = []
    for group in blocks:
        keeper = group.get("keeper") or "none"
        cards = "\n".join(
            _card_html(by_path[path], thumbs, keeper=path == group.get("keeper"))
            for path in group["members"]
            if path in by_path
        )
        title = f"{group['id']} · {group['kind']} · {group['size']} frames · keep {keeper}"
        if group.get("rejected_best"):
            title += f" · best effort {group['rejected_best']} is not clean"
        parts.append(f'<section class="dup"><h3>{html.escape(title)}</h3><div class="row">{cards}</div></section>')
    return "\n".join(parts)


def _card_html(image: dict[str, Any], thumbs: dict[str, bytes], *, keeper: bool) -> str:
    label = str(image.get("label") or "unknown")
    keep = image.get("keep")
    keep_class = "keep" if keep is True else "drop" if keep is False else "review"
    keep_text = "keep" if keep is True else "drop" if keep is False else "review"
    review = "1" if image.get("needs_review") else "0"
    classes = "card keeper" if keeper else "card"
    blob = thumbs.get(image["path"])
    if blob:
        import base64

        src = "data:image/jpeg;base64," + base64.standard_b64encode(blob).decode("ascii")
        visual = f'<img alt="{html.escape(image["path"])}" src="{src}">'
    else:
        visual = '<div class="placeholder">unreadable</div>'
    reasons = " ".join(image.get("reasons") or [])
    confidence = image.get("confidence")
    confidence_text = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else ""
    badges = (
        f'<span class="badge {html.escape(label)}">{html.escape(label.replace("_", " "))}</span>'
        f'<span class="badge {keep_class}">{keep_text}</span>'
    )
    if image.get("needs_review"):
        badges += '<span class="badge review">needs review</span>'
    return (
        f'<article class="{classes}" data-label="{html.escape(label)}" data-review="{review}" '
        f'data-size="{int(image.get("group_size") or 1)}">'
        f"{visual}"
        f'<div class="meta"><span class="file">{html.escape(image["path"])}</span>'
        f'<div class="badges">{badges}<span class="badge">{html.escape(confidence_text)}</span></div>'
        f'<p class="reasons">{html.escape(reasons)}</p></div></article>'
    )
