"""
Convert VisDrone-DET annotations to COCO-format ground truth JSON.

Usage:
    python src/visdrone_to_coco.py \
        --dataset-root data/visdrone_raw/VisDrone2019-DET-val \
        --output data/coco_gt/visdrone_val.json
"""

from pathlib import Path
from PIL import Image
import argparse
import json


# VisDrone category IDs 1-10 are the official evaluation classes
VISDRONE_CLASSES = {
    0: "ignored", 1: "pedestrian", 2: "people", 3: "bicycle",
    4: "car", 5: "van", 6: "truck", 7: "tricycle",
    8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others",
}

EVAL_CATEGORY_IDS = list(range(1, 11))


def parse_visdrone_annotation(annotation_path: Path):
    """Parse one VisDrone .txt annotation file into a list of box dicts."""
    boxes = []
    if not annotation_path.exists():
        return boxes

    with annotation_path.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            x, y, w, h = map(float, parts[0:4])
            boxes.append({
                "x": x, "y": y, "w": w, "h": h,
                "score":       float(parts[4]),
                "category_id": int(parts[5]),
                "truncation":  int(parts[6]),
                "occlusion":   int(parts[7]),
            })

    return boxes


def build_coco(dataset_root: Path, include_ignored_and_others: bool = False):
    images_dir      = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"

    if not images_dir.exists() or not annotations_dir.exists():
        raise FileNotFoundError(f"Expected images/ and annotations/ under {dataset_root}")

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images in {images_dir}")

    coco_images      = []
    coco_annotations = []
    ann_id = 1

    for image_id, image_path in enumerate(image_paths, start=1):
        # Read image dimensions only; no pixel data is loaded into memory
        with Image.open(image_path) as im:
            width, height = im.size

        coco_images.append({
            "id": image_id, "file_name": image_path.name,
            "width": width, "height": height,
        })

        ann_path = annotations_dir / f"{image_path.stem}.txt"
        for box in parse_visdrone_annotation(ann_path):
            cat = box["category_id"]
            if not include_ignored_and_others and cat not in EVAL_CATEGORY_IDS:
                continue

            coco_annotations.append({
                "id":          ann_id,
                "image_id":    image_id,
                "category_id": cat,
                "bbox":        [box["x"], box["y"], box["w"], box["h"]],
                "area":        box["w"] * box["h"],  # required by pycocotools
                "iscrowd":     0,  # 0 = individual instance; VisDrone has no crowd regions
                "truncation":  box["truncation"],
                "occlusion":   box["occlusion"],
            })
            ann_id += 1

    cat_ids = EVAL_CATEGORY_IDS if not include_ignored_and_others else sorted(VISDRONE_CLASSES.keys())
    categories = [{"id": cid, "name": VISDRONE_CLASSES[cid]} for cid in cat_ids]

    return {
        "info": {"description": f"VisDrone-DET ground truth: {dataset_root.name}", "source": str(dataset_root)},
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone-DET annotations to COCO JSON.")
    parser.add_argument("--dataset-root", type=str, default="data/visdrone_raw/VisDrone2019-DET-val")
    parser.add_argument("--output", type=str, default="data/coco_gt/visdrone_val.json")
    parser.add_argument("--include-ignored-and-others", action="store_true",
                        help="Include category 0 (ignored) and 11 (others). Off by default.")
    args = parser.parse_args()

    coco = build_coco(
        dataset_root=Path(args.dataset_root),
        include_ignored_and_others=args.include_ignored_and_others,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(coco, f)

    print(f"Wrote {len(coco['images'])} images and {len(coco['annotations'])} annotations to {out_path}")

if __name__ == "__main__":
    main()
