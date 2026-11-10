"""
Visualize VisDrone ground-truth bounding boxes on a sample of val images.

Usage:
    python src/prepare_dataset.py --max-images 10
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


VISDRONE_CLASSES = {
    0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "tricycle",
    8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others",
}


def parse_visdrone_annotation(annotation_path: Path):
    """Parse one VisDrone .txt annotation file into a list of box dicts."""
    boxes = []
    if not annotation_path.exists():
        print(f"Missing annotation file: {annotation_path}")
        return boxes

    with annotation_path.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            x, y, w, h = map(float, parts[0:4])
            boxes.append({
                "x_min": x, "y_min": y, "width": w, "height": h,
                "x_max": x + w, "y_max": y + h,
                "score":         float(parts[4]),
                "category_id":   int(parts[5]),
                "category_name": VISDRONE_CLASSES.get(int(parts[5]), "unknown"),
                "truncation":    int(parts[6]),
                "occlusion":     int(parts[7]),
            })

    return boxes


def draw_boxes(image_path: Path, annotation_path: Path, output_path: Path, include_ignored: bool = False):
    image = Image.open(image_path).convert("RGB")
    draw  = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for box in parse_visdrone_annotation(annotation_path):
        if box["category_id"] == 0 and not include_ignored:
            continue

        x1, y1 = int(box["x_min"]), int(box["y_min"])
        x2, y2 = int(box["x_max"]), int(box["y_max"])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        # Clamp to y=0 so the label never renders above the image edge
        draw.text((x1, max(0, y1 - 12)), box["category_name"], fill="red", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def visualize_dataset(dataset_root: Path, output_dir: Path, max_images: int, include_ignored: bool):
    images_dir      = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations folder not found: {annotations_dir}")

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in: {images_dir}")

    print(f"Found {len(image_paths)} images. Saving to: {output_dir}")

    for i, image_path in enumerate(image_paths[:max_images]):
        annotation_path = annotations_dir / f"{image_path.stem}.txt"
        output_path     = output_dir / image_path.name
        draw_boxes(image_path, annotation_path, output_path, include_ignored)
        print(f"[{i + 1}/{max_images}] Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize VisDrone ground-truth bounding boxes.")
    parser.add_argument("--dataset-root", type=str, default="data/visdrone_raw/VisDrone2019-DET-val")
    parser.add_argument("--output-dir", type=str, default="outputs/ground_truth_examples")
    parser.add_argument("--max-images", type=int, default=10)
    parser.add_argument("--include-ignored", action="store_true", help="Draw ignored regions as well.")
    args = parser.parse_args()

    visualize_dataset(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        max_images=args.max_images,
        include_ignored=args.include_ignored,
    )

if __name__ == "__main__":
    main()
