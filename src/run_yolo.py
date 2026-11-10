"""
Run a YOLO model on VisDrone val images and save predictions in the shared schema.

Usage:
    python src/run_yolo.py \
        --model-path models/weights/fine-tuned.pt \
        --max-images 0 \
        --predictions-path outputs/YOLO/yolo_predictions/yolo_finetuned_predictions.json
"""

from pathlib import Path
import argparse
import json


def run_yolo_on_visdrone(
    images_dir: Path,
    output_dir: Path,
    predictions_path: Path,
    model_path: str = "yolo11n.pt",
    max_images: int = 20,
    confidence_threshold: float = 0.25,
    save_visualizations: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    # Import inside the function so --help works without ultralytics installed
    from ultralytics import YOLO
    model = YOLO(model_path)

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    if max_images and max_images > 0:
        image_paths = image_paths[:max_images]

    all_predictions = {}

    for i, image_path in enumerate(image_paths):
        if (i + 1) % 25 == 0 or i == 0 or i == len(image_paths) - 1:
            print(f"[{i + 1}/{len(image_paths)}] {image_path.name}")

        results = model.predict(
            source=str(image_path),
            conf=confidence_threshold,
            save=False,
            verbose=False,
        )

        result = results[0]  # predict() returns a list; we passed one image
        image_predictions = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            image_predictions.append({
                "class_id":   int(box.cls[0]),
                "class_name": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox":       [x1, y1, x2, y2],
            })

        all_predictions[image_path.name] = image_predictions

        if save_visualizations:
            import cv2
            plotted_image = result.plot(line_width=1, font_size=4)  # returns BGR numpy array
            cv2.imwrite(str(output_dir / image_path.name), plotted_image)

    with predictions_path.open("w") as f:
        json.dump(all_predictions, f, indent=2)

    print(f"\nWrote predictions for {len(all_predictions)} images to {predictions_path}")
    if save_visualizations:
        print(f"Wrote visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLO on VisDrone images.")
    parser.add_argument("--images-dir", type=str, default="data/visdrone_raw/VisDrone2019-DET-val/images")
    parser.add_argument("--output-dir", type=str, default="outputs/yolo_visualizations")
    parser.add_argument("--predictions-path", type=str, default="outputs/yolo_predictions/yolo_predictions.json")
    parser.add_argument("--model-path", type=str, default="yolo11n.pt")
    parser.add_argument("--max-images", type=int, default=20, help="0 = all images")
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--no-visualizations", action="store_true")
    args = parser.parse_args()

    run_yolo_on_visdrone(
        images_dir=Path(args.images_dir),
        output_dir=Path(args.output_dir),
        predictions_path=Path(args.predictions_path),
        model_path=args.model_path,
        max_images=args.max_images,
        confidence_threshold=args.confidence_threshold,
        save_visualizations=not args.no_visualizations,
    )

if __name__ == "__main__":
    main()
