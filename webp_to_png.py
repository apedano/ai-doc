#!/usr/bin/env python3
"""Convert a WebP image to PNG. The input can be a URL or a local file path.

Usage:
    python webp2png.py SOURCE [OUTPUT] [--compress 6]

Example:
    python .\webp_to_png.py "C:\Users\...507aae7ad9ff_2198x856.webp" rl_training_diagram


SOURCE  an http(s) URL or a path to a local .webp file
OUTPUT  output file name; if omitted, it is derived from SOURCE
        (photo.webp / https://example.com/img/photo.webp -> photo.png,
        saved in the current directory)

Requires: pip install requests pillow
"""
import argparse
import io
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from PIL import Image


def is_url(source: str) -> bool:
    return urlparse(source).scheme in ("http", "https")


def default_filename(source: str) -> str:
    """Derive '<name>.png' from the URL path or local file name."""
    path = urlparse(source).path if is_url(source) else source
    name = Path(unquote(path) if is_url(source) else path).stem
    return f"{name or 'image'}.png"


def load_bytes(source: str) -> bytes:
    if is_url(source):
        resp = requests.get(source, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp.content
    return Path(source).expanduser().read_bytes()


def convert(source: str, output: str | None = None, compress: int = 6) -> Path:
    out_path = Path(output) if output else Path(default_filename(source))
    if not out_path.suffix:  # name given without extension -> add .png
        out_path = out_path.with_suffix(".png")

    img = Image.open(io.BytesIO(load_bytes(source)))

    # PNG supports transparency, so alpha is preserved. Only convert modes PNG can't store.
    if img.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "I;16"):
        has_alpha = "A" in img.getbands()
        img = img.convert("RGBA" if has_alpha else "RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", compress_level=compress)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a WebP (URL or local file) to PNG.")
    parser.add_argument("source", help="http(s) URL or local path of the .webp image")
    parser.add_argument("output", nargs="?", help="output file name (default: derived from source)")
    parser.add_argument(
        "-c", "--compress", type=int, choices=range(0, 10), default=6,
        help="PNG compression level 0-9, lossless either way (default: 6)",
    )
    args = parser.parse_args()

    try:
        path = convert(args.source, args.output, args.compress)
    except requests.RequestException as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(f"File not found: {args.source}", file=sys.stderr)
        return 1
    except OSError as e:  # includes Pillow's UnidentifiedImageError on bad data
        print(f"Could not read/convert image: {e}", file=sys.stderr)
        return 1

    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())