#!/usr/bin/env python3
"""
svg_to_png.py — Convert an SVG file from a web URL to a PNG image.

Usage:
    python svg_to_png.py <svg_url> [output.png] [--dpi N] [--scale N] [--width N]

Examples:
    python svg_to_png.py https://example.com/image.svg
    python svg_to_png.py https://example.com/image.svg output.png
    python svg_to_png.py https://example.com/image.svg output.png --dpi 300
    python svg_to_png.py https://example.com/image.svg output.png --scale 2.0
    python svg_to_png.py https://example.com/image.svg output.png --width 1920

Dependencies:
    pip install cairosvg
"""

import sys
import argparse
import urllib.request
import urllib.error
import os

try:
    import cairosvg
except ImportError:
    sys.exit(
        "Missing dependency: pip install cairosvg\n"
        "  Note: cairosvg also requires the Cairo system library.\n"
        "  Ubuntu/Debian: sudo apt install libcairo2\n"
        "  macOS:         brew install cairo\n"
        "  Windows:       https://cairosvg.org/documentation/"
    )


# ── Fetch ────────────────────────────────────────────────────────────────────

def fetch_svg(url: str) -> bytes:
    """Download SVG bytes from *url* with validation."""
    if not url.startswith(("http://", "https://")):
        sys.exit(f"Error: URL must start with http:// or https://\n  Got: {url}")

    print(f"Fetching: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "svg-to-png/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP error {e.code}: {e.reason}\n  URL: {url}")
    except urllib.error.URLError as e:
        sys.exit(f"Could not reach URL: {e.reason}\n  URL: {url}")

    # Validate content type
    is_svg_type = "svg" in content_type or "xml" in content_type
    is_svg_ext  = url.lower().split("?")[0].endswith(".svg")
    looks_svg   = raw.lstrip()[:5] in (b"<svg ", b"<?xml", b"<!DOC") or b"<svg" in raw[:512]

    if not (is_svg_type or is_svg_ext or looks_svg):
        preview = raw[:200].decode("utf-8", errors="replace").strip()
        sys.exit(
            f"Error: The URL does not appear to point to an SVG file.\n"
            f"  Content-Type: {content_type!r}\n"
            f"  Response preview:\n    {preview}\n"
            f"  Tip: Make sure the URL points directly to an .svg file."
        )

    print(f"Downloaded {len(raw):,} bytes  (Content-Type: {content_type!r})")
    return raw


# ── Convert ──────────────────────────────────────────────────────────────────

def convert(svg_bytes: bytes, output_path: str,
            dpi: int | None, scale: float | None, width: int | None) -> None:
    """Render SVG bytes to a PNG file."""
    kwargs: dict = {"write_to": output_path}

    if dpi   is not None: kwargs["dpi"]         = dpi
    if scale is not None: kwargs["scale"]        = scale
    if width is not None: kwargs["output_width"] = width

    try:
        cairosvg.svg2png(bytestring=svg_bytes, **kwargs)
    except Exception as e:
        sys.exit(f"Error converting SVG to PNG:\n  {e}")

    size = os.path.getsize(output_path)
    print(f"Saved → {output_path}  ({size:,} bytes)")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an SVG from a URL to a PNG image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Size options are mutually exclusive. Priority: --width > --scale > --dpi.\n"
            "Default (no size option): render at the SVG's native size."
        ),
    )
    parser.add_argument("url",    help="URL pointing to the SVG file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output PNG path (default: derived from SVG filename)")

    size = parser.add_mutually_exclusive_group()
    size.add_argument("--dpi",   type=int,   metavar="N",
                      help="Output resolution in DPI (e.g. 96, 150, 300)")
    size.add_argument("--scale", type=float, metavar="N",
                      help="Scale factor relative to native size (e.g. 2.0 = 2×)")
    size.add_argument("--width", type=int,   metavar="N",
                      help="Force output width in pixels (height scales proportionally)")

    return parser.parse_args()


def output_from_url(url: str) -> str:
    """Derive a PNG filename from the SVG URL, e.g. foo.svg → foo.png."""
    path = url.split("?")[0].rstrip("/")   # strip query string and trailing slash
    basename = path.split("/")[-1]         # last path segment
    name, _ = os.path.splitext(basename)
    return (name or "output") + ".png"


def main() -> None:
    args   = parse_args()
    output = args.output or output_from_url(args.url)
    raw    = fetch_svg(args.url)
    convert(raw, output, dpi=args.dpi, scale=args.scale, width=args.width)


if __name__ == "__main__":
    main()
