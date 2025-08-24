# tag_generator.py
"""
Standalone Tag Generator for ForenSnap
- Uses ScreenshotAnalyzer for OCR, entity recognition, and tagging
- Optionally integrates BLIP captions as tags

Usage:
    python tag_generator.py <image_path> [credentials_path]
"""

import sys
from screenshot_analyzer import ScreenshotAnalyzer

# Optionally import BLIP captioning if available
try:
    from run_blip_demo import load_image, main as blip_main
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

def get_blip_caption(image_path):
    """Run BLIP captioning and return the main caption (if available)."""
    if not BLIP_AVAILABLE:
        return None
    # You may want to refactor run_blip_demo.py to return the caption instead of printing
    # For now, just run BLIP and capture output (not ideal, but works for demo)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        blip_main(image_path)
    output = buf.getvalue()
    # Try to extract the first caption from output
    for line in output.splitlines():
        if line.lower().startswith("conditional caption:"):
            return line.split(":", 1)[-1].strip()
    return None

def generate_tags(image_path, credentials_path=None, use_blip=True):
    analyzer = ScreenshotAnalyzer(credentials_path=credentials_path)
    result = analyzer.process_image(image_path)
    tags = result.get("tags", [])
    print(f"Tags for {image_path}:")
    for tag in tags:
        print(f" - {tag}")
    # Optionally add BLIP caption as a tag
    if use_blip and BLIP_AVAILABLE:
        blip_caption = get_blip_caption(image_path)
        if blip_caption:
            blip_tag = f"blip_caption.{blip_caption.lower().replace(' ', '_')}"
            print(f" - {blip_tag}")
            tags.append(blip_tag)
    return tags

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tag_generator.py <image_path> [credentials_path]")
        sys.exit(1)
    image_path = sys.argv[1]
    credentials_path = sys.argv[2] if len(sys.argv) > 2 else None
    generate_tags(image_path, credentials_path)
