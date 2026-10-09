# jev-photo-cull

Read-only photo cull. Point it at a folder of pictures and it writes a report somewhere else: which files look like duplicates or bursts, and which look clean, blurry, too dark, or overexposed.

Originals are never modified, moved, or deleted. The tool opens files read-only. Every artifact (CSV, JSON, HTML, thumbnail bytes inside that HTML, the decision cache, and an optional copy or symlink of suggested keepers) is created only under `--out`, and `--out` must sit outside the input tree.

## Install

Python 3.11 or newer.

```sh
pip install -e ".[dev]"
```

HEIC/HEIF is optional and needs libheif:

```sh
pip install -e ".[heif]"
```

Without that extra, `.heic` and `.heif` files are listed in the report as `unknown` with an install hint, and the scan continues.

JPEG, PNG, and WebP are read with Pillow. Subfolders are included. Other files are ignored. A file with an image extension that cannot be decoded is reported as `unknown` instead of failing the run.

## Quick start

```sh
jev-photo-cull scan ./photos --out ./cull --backend heuristics
```

That writes:

| File | What it is |
| --- | --- |
| `cull/report.csv` | One row per image |
| `cull/report.json` | The same records, plus group summaries |
| `cull/review.html` | One self-contained page: inline CSS and JS, thumbnails as JPEG data URIs, no network |

Open `review.html` in a browser. Duplicate and burst groups are shown side by side with the suggested keeper outlined. Filters cover label, needs-review, and duplicates. The grid is sorted worst-first (overexposed, too dark, blurry, unknown, then clean).

Copy suggested keepers (still not a move):

```sh
jev-photo-cull scan ./photos --out ./cull --backend heuristics --export-keep-list
jev-photo-cull scan ./photos --out ./cull --backend heuristics --export-keep-list symlink
```

Copies land in `cull/keepers/` and the relative paths are listed in `cull/keepers.txt`. A frame is exported only when `keep` is true. A group whose best frame is still a defect exports nothing from that group. Abstentions are not exported.

## How a scan decides

**Pass 1** is local and does not use the network.

- SHA-256 of the file bytes, and a 64-bit difference hash (dHash) after EXIF orientation is applied.
- Exact copies (same SHA-256) share a group.
- Near-duplicates share a group when the dHash distance is at most `--near-hamming` (default 8) and mean luminance is within 35. Blank frames hash like other blank frames, so a blank only pairs with another blank of a similar color. That keeps a black frame from matching a white one.
- Bursts add EXIF `DateTimeOriginal` (falling back to `DateTime`). Within `--burst-seconds` (default 2) the hash distance may be as loose as `--burst-hamming` (default 16), and the exposure gap may be wider, so a dark frame in a burst can still sit next to a better one.
- Sharpness is the variance of a Laplacian after a 1st/99th-percentile stretch, measured on a long edge of at most 1024 px. The stretch is what stops a dark-but-sharp photo from being called blurry.
- Exposure uses mean luminance, the fraction of pixels clipped near black (≤ 8) or white (≥ 247), and a 16-bin luminance histogram.
- Labels are `clean`, `blurry`, `too_dark`, `overexposed`, or `unknown`, each with a confidence and a probability spread.

**Pass 2** is a decision backend with the same job: a quality choice per image, and for each duplicate group a choice of which shot to keep, including an explicit abstain.

| Backend | When it runs | Network |
| --- | --- | --- |
| `heuristics` | Default when Cloudflare credentials are absent. Also the pass-1 signal when another backend is selected. | None |
| `clef` | Default when `CLOUDFLARE_ACCOUNT_ID` and a token are set. `--backend clef` requires them and exits 2 without sending a request if they are missing. | Downscaled JPEGs go to Cloudflare Workers AI |
| `local` | Optional open-weights server you run yourself. Not installed by this package. | Only to the URL you configure, default `127.0.0.1` |

`--backend auto` (the default) picks `clef` when those credentials are present and `heuristics` otherwise.

The two signals are merged. Agreement raises confidence. Disagreement, an `unknown` label, a model abstain, or confidence under `--review-below` (default 0.60) sets `needs_review`. Inside a group, the keeper is the frame both passes prefer. A tie on local scores among visually equivalent frames keeps the first path. If the winning frame's label is not `clean`, nothing in that group is marked `keep`.

Model answers are cached under `<out>/cache/decisions.json`, keyed by file hash, the exact JPEG that would be sent, the backend, and the question text. A rerun skips cached images. `--no-cache` turns that off.

## Clef (Cloudflare Workers AI)

Documented on the model pages (checked 2026-10-09):

| Model | Workers AI id | Body `model` field | List price |
| --- | --- | --- | --- |
| Clef-flash (default) | `@cf/cloudflare/clef-flash` | `clef-flash` | $0.09 / million input tokens |
| Clef | `@cf/cloudflare/clef` | `clef` | $0.24 / million input tokens |

```sh
export CLOUDFLARE_ACCOUNT_ID=your_account_id
export CLOUDFLARE_API_TOKEN=your_token   # Account → Workers AI → Read
jev-photo-cull scan ./photos --out ./cull --backend clef
jev-photo-cull scan ./photos --out ./cull --backend clef --model clef
```

The Workers AI examples use `CLOUDFLARE_AUTH_TOKEN`. This tool accepts that name when `CLOUDFLARE_API_TOKEN` is unset. A `.env` file in the working directory is loaded and does not override variables already set. `.env.example` lists the names. Do not commit a filled-in `.env`.

The client posts JSON to:

```text
https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/cloudflare/clef-flash
Authorization: Bearer $TOKEN
```

The body is the model input itself (`model`, `state`, `questions`, `images`), matching the Clef input schema. Quality is a `choice` question whose options are exactly `clean`, `blurry`, `too_dark`, `overexposed`, and `unknown`. A group keeper is a `choice` of `shot_1` … `shot_n` plus `abstain`. Images are downscaled JPEGs (long edge `--max-edge`, default 512) sent as `{"content_type": "image/jpeg", "base64": "..."}`, which is one of the two forms in the schema (the other is a `data:` URL). Remote image URLs are not accepted by the schema and are not sent.

Limits taken from that schema: at most 4 images per request, 4 MiB and 16 megapixels each, 8 MiB decoded in total, whole body at most 13 MiB, and 1–64 questions. Quality calls are batched 4 images at a time. A duplicate group larger than 4 images sends the 4 frames with the best local scores and names the rest in the text state. Requests are paced at `--requests-per-minute` (default 30) and retried on HTTP 429 and 5xx, honoring `Retry-After`.

The Workers AI REST API wraps model output as `{"success", "errors", "messages", "result"}`. The Clef result object is `model`, `answers`, and `usage`, per the output schema. A bare body that already has `answers` is also accepted.

### Cost and privacy

Heuristics never leave the machine. The local backend talks only to the server you point it at.

The clef backend uploads those downscaled JPEGs and the question text to Cloudflare Workers AI. It does not upload original files. Cloudflare's launch post says they do not read, store, or train on Workers AI requests or responses unless you opt into fine-tuning. That is their published claim; this tool cannot enforce it. Token use depends on image size. The published price is per million input tokens; a separate vision-token formula was not on the model page, which is why images are downscaled before they are sent. There is no live call in the test suite.

## Local backend

Nothing in this package downloads weights. `--backend local` expects a decision server that speaks a System One style API.

Ollama, raw base64 images (what the Ollama clef-flash card describes; it rejects data URLs):

```sh
ollama pull clef-flash
jev-photo-cull scan ./photos --out ./cull --backend local
```

Defaults: `JEV_PHOTO_CULL_LOCAL_URL=http://127.0.0.1:11434/v1/systemone`, model name `clef-flash`, image format `raw`. Override with `--local-url`, `--local-model`, and `--local-image-format raw|data-url|content-type`. If the port is closed, the error names `ollama pull` and does not pretend a model is installed.

[imajev](https://github.com/mohit67890/imajev) is a separate open-weights project (pairwise and single-image decisions, can abstain). Its README shows one multipart `image` file plus a `request` JSON field. That protocol is available, and keeper choice stays on the local scores because a multi-image field is not in that README:

```sh
jev-photo-cull scan ./photos --out ./cull --backend local \
  --local-protocol imajev \
  --local-url http://127.0.0.1:8765/v1/systemone \
  --local-model imajev-4b
```

If an imajev answer sets `abstained` or `unknown_probability` ≥ 0.5, the label becomes `unknown` and the photo is flagged for review.

## Report fields

Each image row includes `path`, `sha256`, `perceptual_hash` (`dhash:` plus 16 hex digits), `group_id`, `group_size`, `group_kind` (`single`, `identical`, `near-duplicate`, `burst`, or `mixed`), `keep` (`true`, `false`, or empty/null), `label`, `confidence`, `needs_review`, heuristic and model labels and probabilities, exposure and sharpness `scores` (including the histogram), `backend`, `model`, `reasons`, `captured_at`, dimensions, and `error` when the file could not be read.

`reasons` is a list of short measurement and decision notes, not a hidden chain of thought.

## Tests

Fixtures are drawn in the test process with Pillow. No photo binaries are committed.

```sh
pytest
```

The heuristics scan checks that generated blurry, dark, and overexposed images get those labels, that exact copies share a group, that a black frame is not grouped with a white frame, and that the input tree is byte-for-byte unchanged. The Clef tests mock HTTP: they check the documented URL, bearer auth, the choice schema, base64 JPEG payloads, batching at 4 images, the REST error envelope, retries, the response cache, and a clear error when credentials are missing. No test contacts Cloudflare.

## Open questions

These are limits of what the docs showed on 2026-10-09, not guesses filled in as facts.

- Workers AI account rate limits for Clef were not on the model page. The client caps itself at 30 requests per minute and backs off on 429. Confirm the account limit before a large library.
- The Clef Python sample prints `response.json()` and does not show the REST envelope. The client follows the usual Workers AI `/ai/run` envelope and also accepts a bare `{answers}` body. This was not executed against a live account.
- The input schema allows a data URL or `{content_type, base64}`. The client sends the object form. That was not executed live.
- Input-token accounting for vision, beyond the per-million-token price, was not specified.
- The Ollama card says images are raw base64 and that data URLs are rejected. The default local format follows that. It was not executed live.
- imajev's README shows a single image part. Keeper selection in that mode does not send a second image.
- Hosted System One documents a 32-question cap. The Workers AI Clef schema allows 64. This tool stays at or below 4 questions on a quality call.
