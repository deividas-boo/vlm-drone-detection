"""
Evaluate model predictions against VisDrone ground truth.

Usage:
    python src/evaluate.py \
        --predictions outputs/YOLO/yolo_predictions/yolo_finetuned_predictions.json \
        --gt data/coco_gt/visdrone_val.json \
        --report-out outputs/eval_reports/yolo11s_finetuned_visdrone.json
"""

from pathlib import Path
import argparse
import json
import sys


def load_class_map(path):
    if path is None:
        return None
    with Path(path).open("r") as f:
        return json.load(f)


def load_coco_gt(gt_path):
    from pycocotools.coco import COCO

    coco_gt = COCO(str(gt_path))
    # Build lookup tables: class name → category ID and filename → image ID
    name_to_cat_id = {c["name"]: c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
    filename_to_image_id = {img["file_name"]: img["id"] for img in coco_gt.loadImgs(coco_gt.getImgIds())}

    return coco_gt, name_to_cat_id, filename_to_image_id


def convert_predictions(predictions_dict, filename_to_image_id, name_to_cat_id, class_map=None):
    """Convert predictions from project schema to COCO result format."""
    coco_results = []
    dropped_unknown_class = 0
    dropped_unknown_image = 0

    for filename, preds in predictions_dict.items():
        image_id = filename_to_image_id.get(filename)
        if image_id is None:
            dropped_unknown_image += 1
            continue

        for det in preds:
            class_name = det.get("class_name")
            if class_name is None:
                continue

            if class_map is not None:
                class_name = class_map.get(class_name)
                if class_name is None:
                    dropped_unknown_class += 1
                    continue

            category_id = name_to_cat_id.get(class_name)
            if category_id is None:
                dropped_unknown_class += 1
                continue

            # Convert from [x1,y1,x2,y2] to COCO's [x,y,w,h]
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1

            if w <= 0 or h <= 0:
                continue

            coco_results.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": float(det.get("confidence", 1.0)),
            })

    return coco_results, dropped_unknown_class, dropped_unknown_image


def coco_eval(coco_gt, coco_results, image_ids):
    from pycocotools.cocoeval import COCOeval

    if not coco_results:
        return None

    coco_dt = coco_gt.loadRes(coco_results)
    e = COCOeval(coco_gt, coco_dt, iouType="bbox")
    # Restrict evaluation to the images we actually predicted
    e.params.imgIds = image_ids
    e.evaluate()
    e.accumulate()
    e.summarize()

    return e


def per_class_ap(coco_eval_obj, coco_gt):
    """Extract AP@0.5:0.95 per category from a completed COCO evaluation."""
    if coco_eval_obj is None:
        return {}

    # COCO precision shape: [T, R, K, A, M]
    precision = coco_eval_obj.eval["precision"]
    cat_ids = coco_gt.getCatIds()
    cats = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}

    out = {}
    for k, cat_id in enumerate(cat_ids):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]  # COCO uses -1 as a sentinel for missing entries
        out[cats[cat_id]] = float(p.mean()) if p.size else float("nan")

    return out


def precision_recall_at_iou(coco_eval_obj, iou_idx=0):
    """Mean precision and recall at one IoU threshold (iou_idx=0 -> IoU 0.50)."""
    if coco_eval_obj is None:
        return None, None

    p = coco_eval_obj.eval["precision"][iou_idx, :, :, 0, -1]
    r = coco_eval_obj.eval["recall"][iou_idx, :, 0, -1]
    p = p[p > -1]
    r = r[r > -1]

    return (
        float(p.mean()) if p.size else float("nan"),
        float(r.mean()) if r.size else float("nan"),
    )


def run(predictions_path, gt_path, class_map_path=None, report_out=None):
    coco_gt, name_to_cat_id, filename_to_image_id = load_coco_gt(gt_path)
    class_map = load_class_map(class_map_path)

    with Path(predictions_path).open("r") as f:
        predictions_dict = json.load(f)

    coco_results, dropped_class, dropped_img = convert_predictions(
        predictions_dict, filename_to_image_id, name_to_cat_id, class_map,
    )

    # Only score images present in the predictions file so partial runs aren't penalised
    image_ids_covered = sorted({
        filename_to_image_id[name]
        for name in predictions_dict
        if name in filename_to_image_id
    })

    print(f"Predictions: {sum(len(v) for v in predictions_dict.values())} boxes across {len(predictions_dict)} images")
    print(f"After filtering: {len(coco_results)} boxes kept")
    print(f"Dropped: {dropped_class} unknown class, {dropped_img} unknown image")
    print(f"Evaluating on {len(image_ids_covered)} images")
    print("-" * 60)

    e = coco_eval(coco_gt, coco_results, image_ids_covered)

    if e is None:
        print("\nNo valid predictions to evaluate.")
        return

    map_50_95  = float(e.stats[0])
    map_50     = float(e.stats[1])
    map_75     = float(e.stats[2])
    map_small  = float(e.stats[3])
    map_medium = float(e.stats[4])
    map_large  = float(e.stats[5])

    # NB: precision averaged over all recall thresholds is, by the COCO
    # definition, the area under the PR curve, which equals AP@0.5. It
    # therefore duplicates mAP@0.5 and is not reported as a separate metric.
    _p_at_50, r_at_50 = precision_recall_at_iou(e, iou_idx=0)
    per_class = per_class_ap(e, coco_gt)

    print("\nPer-class AP @ IoU 0.5:0.95:")
    for name, ap in sorted(per_class.items(), key=lambda kv: -kv[1] if kv[1] == kv[1] else 0):
        print(f"  {name:18s}  AP = {ap:.4f}")

    print(f"\nSummary:  mAP@0.5={map_50:.4f}   mAP@0.5:0.95={map_50_95:.4f}   R@0.5={r_at_50:.4f}")

    report = {
        "predictions_file": str(predictions_path),
        "gt_file": str(gt_path),
        "class_map_file": str(class_map_path) if class_map_path else None,
        "num_images_evaluated": len(image_ids_covered),
        "num_predictions_after_mapping": len(coco_results),
        "metrics": {
            "mAP_0.5": map_50,
            "mAP_0.5:0.95": map_50_95,
            "mAP_0.75": map_75,
            "mAP_small": map_small,
            "mAP_medium": map_medium,
            "mAP_large": map_large,
            "recall_at_0.5": r_at_50,
            "per_class_AP_0.5:0.95": per_class,
        },
    }

    if report_out is not None:
        out_path = Path(report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions JSON against VisDrone COCO ground truth.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--class-map", default=None)
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args()

    try:
        run(args.predictions, args.gt, args.class_map, args.report_out)
    except ModuleNotFoundError as exc:
        if "pycocotools" in str(exc):
            print("pycocotools is not installed. Run:\n    pip install pycocotools\n", file=sys.stderr)
            sys.exit(1)
        raise

if __name__ == "__main__":
    main()
