"""
Compare evaluation results across all models.

Usage:
    python src/compare_models.py
    python src/compare_models.py --reports-dir outputs/eval_reports
"""

import argparse
import json
from pathlib import Path


OVERALL_METRICS = [
    ("mAP@0.5",       "mAP_0.5"),
    ("mAP@0.5:0.95",  "mAP_0.5:0.95"),
    ("mAP@0.75",      "mAP_0.75"),
    ("mAP small",     "mAP_small"),
    ("mAP medium",    "mAP_medium"),
    ("mAP large",     "mAP_large"),
    ("Recall@0.5",    "recall_at_0.5"),
]

ALL_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def load_reports(reports_dir: Path):
    # File stem (filename without extension) becomes the model name in the table
    reports = {}
    for path in sorted(reports_dir.glob("*.json")):
        with path.open() as f:
            data = json.load(f)
        reports[path.stem] = data["metrics"]
    return reports


def col_width(models):
    return max(12, max(len(m) for m in models) + 2)


def print_table(title, rows, models, cw):
    label_w = max(16, max(len(r[0]) for r in rows) + 2)
    total_w = label_w + len(models) * (cw + 3) + 1

    print(f"\n{'=' * total_w}")
    print(f"  {title}")
    print(f"{'=' * total_w}")

    header = f"{'Metric':<{label_w}}" + "".join(f"  {m:>{cw}}" for m in models)
    print(header)
    print("-" * total_w)

    for label, values in rows:
        row = f"{label:<{label_w}}"
        for v in values:
            # Missing metric prints as a dash rather than crashing
            if v is None:
                row += f"  {'—':>{cw}}"
            else:
                row += f"  {v:>{cw}.4f}"
        print(row)

    print("=" * total_w)


def main():
    parser = argparse.ArgumentParser(description="Compare model evaluation reports side by side.")
    parser.add_argument("--reports-dir", type=str, default="outputs/eval_reports")
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)

    if not reports_dir.exists():
        print(f"No reports folder found at {reports_dir}. Run evaluate.py first.")
        return

    reports = load_reports(reports_dir)

    if not reports:
        print(f"No JSON files found in {reports_dir}.")
        return

    models = list(reports.keys())
    cw = col_width(models)

    overall_rows = []
    for label, key in OVERALL_METRICS:
        values = [reports[m].get(key) for m in models]
        overall_rows.append((label, values))

    print_table("OVERALL METRICS", overall_rows, models, cw)

    # Collect all classes that appear in at least one report
    classes_present = set()
    for m in models:
        classes_present.update(reports[m].get("per_class_AP_0.5:0.95", {}).keys())

    # Keep standard VisDrone order; append any extras alphabetically
    ordered_classes = [c for c in ALL_CLASSES if c in classes_present]
    ordered_classes += sorted(classes_present - set(ALL_CLASSES))

    per_class_rows = []
    for cls in ordered_classes:
        values = [reports[m].get("per_class_AP_0.5:0.95", {}).get(cls) for m in models]
        per_class_rows.append((cls, values))

    print_table("PER-CLASS AP @ IoU 0.5:0.95", per_class_rows, models, cw)

    # When exactly two models are compared, also print the absolute delta
    if len(models) == 2:
        m1, m2 = models
        print(f"\n  Change from '{m1}' -> '{m2}':")
        print(
            f"  {'mAP@0.5':20s} {reports[m1]['mAP_0.5']:.4f} -> "
            f"{reports[m2]['mAP_0.5']:.4f}  "
            f"({(reports[m2]['mAP_0.5'] - reports[m1]['mAP_0.5']):+.4f})"
        )
        print(
            f"  {'mAP@0.5:0.95':20s} {reports[m1]['mAP_0.5:0.95']:.4f} -> "
            f"{reports[m2]['mAP_0.5:0.95']:.4f}  "
            f"({(reports[m2]['mAP_0.5:0.95'] - reports[m1]['mAP_0.5:0.95']):+.4f})"
        )
        print()

if __name__ == "__main__":
    main()
