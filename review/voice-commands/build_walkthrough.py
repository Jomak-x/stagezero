#!/usr/bin/env python3
"""Build a labeled screenshot slideshow for the voice-command PR review."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    # Codex desktop bundles Pillow. An ordinary Python with Pillow also works.
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
    if bundled.is_file() and Path(sys.executable).resolve() != bundled.resolve():
        raise SystemExit(subprocess.call([str(bundled), __file__, *sys.argv[1:]]))
    raise SystemExit("Pillow is required (install pillow or use the Codex bundled Python).")

STEPS = (
    ("01-activate.png", "Activate Voice tools", "Find Voice tools in the Studio sidebar and open its controls."),
    ("02-command.png", "Enter a command", "Use the typed command path to review the routing used after transcription."),
    ("03-queue.png", "Check the queue", "Inspect the ordered requests and each request's cancellation control."),
    ("04-results.png", "Review the result", "Look at the completed scene or action in the Studio workflow."),
    ("05-refine.png", "Refine a named take", "Use the take name and movement controls to inspect the edit workflow."),
)

WIDTH, HEIGHT = 1920, 1080
BACKGROUND = "#0e1521"
ACCENT = "#59e0eb"
WHITE = "#f7f9fc"
MUTED = "#bec7d3"
FONT_PATHS = (
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    raise RuntimeError("No usable system font found")


def paragraph(draw: ImageDraw.ImageDraw, value: str, xy: tuple[int, int],
              max_width: int, typeface: ImageFont.FreeTypeFont, fill: str) -> None:
    lines: list[str] = []
    line = ""
    for word in value.split():
        candidate = f"{line} {word}" if line else word
        if line and draw.textlength(candidate, font=typeface) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    draw.multiline_text(xy, "\n".join(lines), font=typeface, fill=fill, spacing=12)


def render_slide(path: Path, title: str, detail: str, index: int, output: Path) -> None:
    slide = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(slide)
    with Image.open(path) as source:
        screenshot = source.convert("RGB")
    portrait = screenshot.height > screenshot.width
    if portrait:
        image_height = 960
        image_width = round(image_height * screenshot.width / screenshot.height)
        image_x, image_y = 84, 60
        screenshot = screenshot.resize((image_width, image_height), Image.Resampling.LANCZOS)
        draw.rounded_rectangle(
            (image_x - 8, image_y - 8, image_x + image_width + 8, image_y + image_height + 8),
            radius=14, outline="#46505d", width=3,
        )
        slide.paste(screenshot, (image_x, image_y))
        text_x = 760
        draw.text((text_x, 68), "Interface walkthrough • test motion", font=font(30), fill=ACCENT)
        draw.text((text_x, 180), f"{index:02d} / {len(STEPS):02d}", font=font(33), fill=MUTED)
        paragraph(draw, title, (text_x, 286), 1040, font(65), WHITE)
        paragraph(draw, detail, (text_x, 520), 980, font(37), MUTED)
        draw.text((text_x, 948), "Captured Studio UI • screenshot slideshow", font=font(27), fill=MUTED)
    else:
        scale = min(1320 / screenshot.width, 790 / screenshot.height)
        image_width = round(screenshot.width * scale)
        image_height = round(screenshot.height * scale)
        image_x = 60
        image_y = 190 + round((790 - image_height) / 2)
        screenshot = screenshot.resize((image_width, image_height), Image.Resampling.LANCZOS)
        draw.rounded_rectangle(
            (image_x - 8, image_y - 8, image_x + image_width + 8, image_y + image_height + 8),
            radius=14, outline="#46505d", width=3,
        )
        slide.paste(screenshot, (image_x, image_y))
        draw.text((60, 40), "Interface walkthrough • test motion", font=font(30), fill=ACCENT)
        text_x = 1430
        draw.text((text_x, 190), f"{index:02d} / {len(STEPS):02d}", font=font(32), fill=MUTED)
        paragraph(draw, title, (text_x, 285), 430, font(54), WHITE)
        paragraph(draw, detail, (text_x, 535), 430, font(32), MUTED)
        paragraph(draw, "Captured Studio UI • screenshot slideshow", (text_x, 913), 430, font(27), MUTED)
    slide.save(output, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "interface-walkthrough.mp4")
    parser.add_argument("--seconds-per-step", type=float, default=5.5)
    args = parser.parse_args()
    if args.seconds_per_step <= 0:
        parser.error("--seconds-per-step must be positive")
    missing = [name for name, _, _ in STEPS if not (args.input_dir / name).is_file()]
    if missing:
        parser.error("missing captured screenshots: " + ", ".join(missing))
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voice-walkthrough-") as scratch:
        slides = []
        for index, (name, title, detail) in enumerate(STEPS, start=1):
            slide_path = Path(scratch) / f"slide-{index:02d}.png"
            render_slide(args.input_dir / name, title, detail, index, slide_path)
            slides.append(slide_path)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for slide in slides:
            command += ["-loop", "1", "-framerate", "30", "-t", str(args.seconds_per_step), "-i", str(slide)]
        streams = "".join(f"[{i}:v]" for i in range(len(slides)))
        command += [
            "-filter_complex", f"{streams}concat=n={len(slides)}:v=1:a=0,format=yuv420p[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-movflags", "+faststart", str(args.output),
        ]
        subprocess.run(command, check=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
