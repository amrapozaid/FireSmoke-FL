"""
FireSmoke-FL Inference Script

Install:
    pip install ultralytics opencv-python pandas numpy openpyxl

Examples:
    python infer_firesmoke_fl.py --model firesmoke_fl_global.pt --source test_images/
    python infer_firesmoke_fl.py --model firesmoke_fl_global.pt --source video.mp4
    python infer_firesmoke_fl.py --model firesmoke_fl_global.pt --source 0
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def is_webcam_source(source):
    return str(source).isdigit()


def is_video_source(source):
    source = str(source)
    return (
        Path(source).suffix.lower() in VIDEO_EXTS
        or source.startswith(("rtsp://", "http://", "https://"))
        or is_webcam_source(source)
    )


def collect_images(source):
    path = Path(source)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    if path.is_dir():
        return sorted([p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    return []


def save_summary(rows, times, output_dir, unit_name):
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "detections.csv", index=False)
    df.to_excel(output_dir / "inference_summary.xlsx", index=False)

    total_time = float(sum(times))
    count = len(times)
    fps = count / total_time if total_time > 0 else 0.0
    latency_ms = float(np.mean(times) * 1000) if times else 0.0

    with open(output_dir / "fps_latency.txt", "w", encoding="utf-8") as f:
        f.write(f"Number of {unit_name}: {count}\n")
        f.write(f"Total inference time (sec): {total_time:.4f}\n")
        f.write(f"Average latency (ms/{unit_name[:-1]}): {latency_ms:.2f}\n")
        f.write(f"FPS: {fps:.2f}\n")

    print("\nInference completed.")
    print(f"Processed {unit_name}: {count}")
    print(f"FPS: {fps:.2f}")
    print(f"Average latency: {latency_ms:.2f} ms")
    print(f"Results saved to: {output_dir}")


def run_image_inference(model, source, output_dir, conf, iou, imgsz, save_images=True):
    image_paths = collect_images(source)
    if not image_paths:
        raise FileNotFoundError(f"No images found in source: {source}")

    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    times = []

    for img_path in image_paths:
        start = time.time()
        result = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=False,
            verbose=False
        )[0]
        elapsed = time.time() - start
        times.append(elapsed)

        if save_images:
            annotated = result.plot()
            cv2.imwrite(str(pred_dir / img_path.name), annotated)

        if result.boxes is None or len(result.boxes) == 0:
            rows.append({
                "file": img_path.name,
                "class_id": None,
                "class_name": "No detection",
                "confidence": None,
                "x1": None, "y1": None, "x2": None, "y2": None,
                "inference_time_sec": elapsed
            })
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(boxes, confs, clss):
            rows.append({
                "file": img_path.name,
                "class_id": int(cls_id),
                "class_name": model.names.get(int(cls_id), str(cls_id)),
                "confidence": float(score),
                "x1": float(box[0]), "y1": float(box[1]),
                "x2": float(box[2]), "y2": float(box[3]),
                "inference_time_sec": elapsed
            })

    save_summary(rows, times, output_dir, "images")


def run_video_inference(model, source, output_dir, conf, iou, imgsz, show=False):
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    video_source = int(source) if is_webcam_source(source) else source
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0 or np.isnan(input_fps):
        input_fps = 25

    out_video = pred_dir / "firesmoke_fl_output.mp4"
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        input_fps,
        (width, height)
    )

    rows = []
    times = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        result = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False
        )[0]
        elapsed = time.time() - start
        times.append(elapsed)

        annotated = result.plot()
        writer.write(annotated)

        if show:
            cv2.imshow("FireSmoke-FL Inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if result.boxes is None or len(result.boxes) == 0:
            rows.append({
                "frame": frame_id,
                "class_id": None,
                "class_name": "No detection",
                "confidence": None,
                "x1": None, "y1": None, "x2": None, "y2": None,
                "inference_time_sec": elapsed
            })
        else:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)

            for box, score, cls_id in zip(boxes, confs, clss):
                rows.append({
                    "frame": frame_id,
                    "class_id": int(cls_id),
                    "class_name": model.names.get(int(cls_id), str(cls_id)),
                    "confidence": float(score),
                    "x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                    "inference_time_sec": elapsed
                })

        frame_id += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    save_summary(rows, times, output_dir, "frames")
    print(f"Output video saved to: {out_video}")


def main():
    parser = argparse.ArgumentParser(description="FireSmoke-FL inference")
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--source", required=True, help="Image, folder, video, webcam index, or stream URL")
    parser.add_argument("--output", default="inference_outputs", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--show", action="store_true", help="Show video/webcam output")
    args = parser.parse_args()

    output_dir = Path(args.output)
    model = YOLO(args.model)

    print("FireSmoke-FL Inference")
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}")
    print(f"Image size: {args.imgsz}")

    if is_video_source(args.source):
        run_video_inference(model, args.source, output_dir, args.conf, args.iou, args.imgsz, args.show)
    else:
        run_image_inference(model, args.source, output_dir, args.conf, args.iou, args.imgsz)


if __name__ == "__main__":
    main()
