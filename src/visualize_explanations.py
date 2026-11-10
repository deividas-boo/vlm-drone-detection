"""
Visualize VLM hybrid explanations.

Left side: original image with numbered red bounding boxes.
Right side: panel listing each detection's class, confidence, and explanation.
Panel expands vertically to fit all explanations — nothing is cut off.

Usage:
    python src/visualize_explanations.py
    python src/visualize_explanations.py --max-images 10
"""

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGES_DIR    = Path("data/visdrone_raw/VisDrone2019-DET-val/images")
OUTPUT_DIR    = Path("outputs/vlm_explanation_visuals")
EXPLANATIONS  = Path("outputs/VLM/qwen3_explanations/qwen3_hybrid_explanations.json")

BOX_COLOUR    = (255, 0, 0)
PANEL_BG      = (30, 30, 30)
TEXT_COLOUR   = (255, 255, 255)
HEADER_COLOUR = (255, 180, 0)
PANEL_WIDTH   = 680
WRAP_WIDTH    = 55
LINE_H        = 26
PADDING       = 20


def load_font(size):
    for name in ["DejaVuSansMono.ttf", "DejaVuSans.ttf", "cour.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


BOX_ALPHA   = 220   # 0-255; raise to make lines more opaque
BOX_LINE_W  = 1     # pixel width of each bounding box line


def annotate_image(image_path, detections, font, font_bold):
    # Draw boxes on a transparent overlay then alpha-composite, so the original is untouched
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    img_w, img_h = img.size

    box_colour_a  = BOX_COLOUR + (BOX_ALPHA,)
    label_fill_a  = BOX_COLOUR + (BOX_ALPHA,)

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        number = str(idx + 1)
        draw.rectangle([x1, y1, x2, y2], outline=box_colour_a, width=BOX_LINE_W)
        tw = font_bold.getbbox(number)[2]
        th = font_bold.getbbox(number)[3]
        pad = 3
        draw.rectangle([x1, y1, x1 + tw + pad * 2, y1 + th + pad * 2], fill=label_fill_a)
        draw.text((x1 + pad, y1 + pad), number, fill=(255, 255, 255, 255), font=font_bold)

    img = Image.alpha_composite(img, overlay).convert("RGB")

    # Pre-calculate total height needed so nothing gets cut off
    needed_h = PADDING + LINE_H + 4 + 8
    for det in detections:
        expl = (det.get("explanation") or "").strip()
        if expl:
            n_lines = len(textwrap.wrap(expl, width=WRAP_WIDTH))
            needed_h += LINE_H + 2 + n_lines * LINE_H + 8 + 8
        else:
            needed_h += LINE_H + 6  # compact: header only, no explanation
    panel_h = max(img_h, needed_h + PADDING)

    panel = Image.new("RGB", (PANEL_WIDTH, panel_h), PANEL_BG)
    pdraw = ImageDraw.Draw(panel)

    y = PADDING
    pdraw.text((PADDING, y), "DETECTIONS", fill=HEADER_COLOUR, font=font_bold)
    y += LINE_H + 4
    pdraw.line([(PADDING, y), (PANEL_WIDTH - PADDING, y)], fill=(80, 80, 80), width=1)
    y += 8

    for idx, det in enumerate(detections):
        class_name  = det["class_name"]
        conf        = det["confidence"]
        explanation = (det.get("explanation") or "").strip()

        header = f"[{idx+1}] {class_name}  conf={conf:.2f}"
        pdraw.text((PADDING, y), header, fill=HEADER_COLOUR, font=font_bold)
        y += LINE_H + 2

        if not explanation:
            # No explanation (e.g. capped by MAX_DETS_PER_IMAGE) -> compact, no divider
            y += 4
            continue

        for line in textwrap.wrap(explanation, width=WRAP_WIDTH):
            pdraw.text((PADDING, y), line, fill=TEXT_COLOUR, font=font)
            y += LINE_H

        y += 8
        pdraw.line([(PADDING, y), (PANEL_WIDTH - PADDING, y)], fill=(60, 60, 60), width=1)
        y += 8

    combined_h = max(img_h, panel_h)
    combined = Image.new("RGB", (img_w + PANEL_WIDTH, combined_h), PANEL_BG)
    combined.paste(img, (0, 0))
    combined.paste(panel, (img_w, 0))
    return combined


SETTING_DEFAULTS = {
    "hybrid": {
        "explanations": "outputs/VLM/qwen3_explanations/qwen3_hybrid_explanations.json",
        "output_dir":   "outputs/VLM/qwen3_explanations/visuals",
    },
    "pure-vlm": {
        "explanations": "outputs/VLM/qwen3_explanations+detection/qwen3vl_pure_explanations.json",
        "output_dir":   "outputs/VLM/qwen3_explanations+detection/visuals",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["hybrid", "pure-vlm"], default="hybrid",
                        help="Which pipeline to visualize (hybrid or pure-vlm)")
    parser.add_argument("--explanations", default=None,
                        help="Override explanations JSON path")
    parser.add_argument("--images-dir", default=str(IMAGES_DIR))
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    defaults      = SETTING_DEFAULTS[args.setting]
    explanations  = Path(args.explanations or defaults["explanations"])
    output_dir    = Path(args.output_dir   or defaults["output_dir"])
    images_dir    = Path(args.images_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting: {args.setting}")
    print(f"Explanations: {explanations}")

    with open(explanations) as f:
        data = json.load(f)

    # Handle both flat {"filename": [...]} and nested {"images": {...}} formats
    images = data["images"] if "images" in data else data
    names = list(images.keys())
    if args.max_images > 0:
        names = names[:args.max_images]

    font      = load_font(18)
    font_bold = load_font(19)

    print(f"Annotating {len(names)} images -> {output_dir}\n")

    for i, name in enumerate(names):
        detections = images[name]
        image_path = images_dir / name

        if not image_path.exists():
            print(f"[{i+1}/{len(names)}] MISSING: {name}")
            continue

        if not detections:
            print(f"[{i+1}/{len(names)}] {name} — no detections, skipping")
            continue

        annotated = annotate_image(image_path, detections, font, font_bold)
        out_path = output_dir / name
        annotated.save(out_path)
        print(f"[{i+1}/{len(names)}] {name} — {len(detections)} detections saved")

    print(f"\nDone. Open {output_dir} to review.")


if __name__ == "__main__":
    main()
