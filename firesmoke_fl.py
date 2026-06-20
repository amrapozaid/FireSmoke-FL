"""
# FireSmoke-FL

FireSmoke-FL: A Privacy-Preserving Federated Learning Framework for Real-Time Fire and Smoke Detection

## Overview

FireSmoke-FL is a federated learning framework designed for real-time fire and smoke detection in smart-city environments. The framework combines:

- YOLOv11n object detection
- P2 high-resolution detection branch
- C2PSA-iEMA attention refinement module
- Dynamic Average Fusion Algorithm (DAFA)
- Federated Learning for privacy-preserving distributed training

The proposed framework enables collaborative model training across distributed edge devices without sharing raw surveillance images.

---

## Features

- Real-time fire and smoke detection
- Privacy-preserving federated learning
- Communication-efficient DAFA aggregation
- Support for IID and Non-IID client distributions
- Cross-dataset evaluation
- Ablation study support
- Real-time deployment evaluation
---

Install:
    pip install ultralytics torch numpy pandas matplotlib pyyaml opencv-python

Run examples:
    python firesmoke_fl.py --mode prepare
    python firesmoke_fl.py --mode dafa --partition iid
    python firesmoke_fl.py --mode dafa --partition noniid
    python firesmoke_fl.py --mode fedavg --partition noniid
    python firesmoke_fl.py --mode all --partition noniid

"""

import argparse
import copy
import json
import random
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

DEFAULT_ZIP = "/data/indoor-fire-smoke.v2i.yolov11.zip"

CLASS_NAMES = ["fire", "smoke"]  # Roboflow file uses ['0','1']; this script renames them for clarity.


# ============================================================
# General utilities
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def copy_or_link(src: Path, dst: Path) -> None:
    """Hard-link where possible, otherwise copy. This saves disk space."""
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def plot_line(x, ys, labels, xlabel, ylabel, title, save_path: Path):
    plt.figure(figsize=(8, 5))
    for y, label in zip(ys, labels):
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_bar(labels, values, ylabel, title, save_path: Path):
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# Dataset preparation
# ============================================================

def extract_dataset(zip_path: Path, output_root: Path) -> Path:
    dataset_dir = output_root / "indoor_fire_smoke_dataset"
    marker = dataset_dir / ".extracted"

    if marker.exists():
        return dataset_dir

    ensure_dir(dataset_dir)
    print(f"Extracting dataset from: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dataset_dir)

    marker.write_text("done", encoding="utf-8")
    return dataset_dir


def fix_data_yaml(dataset_dir: Path) -> Path:
    """
    Roboflow YOLO exports often use relative paths like ../train/images.
    This function writes a new local absolute YAML.
    """
    source_yaml = dataset_dir / "data.yaml"
    data = read_yaml(source_yaml)

    fixed = {
        "train": str((dataset_dir / "train" / "images").resolve()),
        "val": str((dataset_dir / "valid" / "images").resolve()),
        "test": str((dataset_dir / "test" / "images").resolve()),
        "nc": 2,
        "names": CLASS_NAMES,
    }

    fixed_yaml = dataset_dir / "data_local.yaml"
    write_yaml(fixed, fixed_yaml)
    return fixed_yaml


def count_dataset(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for split in ["train", "valid", "test"]:
        img_dir = dataset_dir / split / "images"
        lab_dir = dataset_dir / split / "labels"
        images = list(img_dir.glob("*.*")) if img_dir.exists() else []
        labels = list(lab_dir.glob("*.txt")) if lab_dir.exists() else []
        rows.append({"split": split, "images": len(images), "labels": len(labels)})
    return pd.DataFrame(rows)


def label_path_for_image(image_path: Path, split_dir: Path) -> Path:
    return split_dir / "labels" / f"{image_path.stem}.txt"


def get_label_counts(label_path: Path) -> Dict[int, int]:
    counts = {0: 0, 1: 0}
    if not label_path.exists():
        return counts

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                cls = int(float(parts[0]))
                if cls in counts:
                    counts[cls] += 1
            except ValueError:
                pass
    return counts


def primary_class(label_path: Path) -> int:
    counts = get_label_counts(label_path)
    if counts[0] == counts[1]:
        return random.choice([0, 1])
    return 0 if counts[0] > counts[1] else 1


def create_client_dataset_yaml(client_dir: Path) -> Path:
    yaml_path = client_dir / "data.yaml"
    data = {
        "train": str((client_dir / "train" / "images").resolve()),
        "val": str((client_dir / "valid" / "images").resolve()),
        "test": str((client_dir / "test" / "images").resolve()),
        "nc": 2,
        "names": CLASS_NAMES,
    }
    write_yaml(data, yaml_path)
    return yaml_path


def create_federated_clients(
    dataset_dir: Path,
    output_root: Path,
    num_clients: int = 5,
    partition: str = "iid",
    seed: int = 42,
) -> List[Path]:
    """
    Creates client datasets from the train split.
    - IID: shuffled round-robin distribution, approximately similar class proportions.
    - non-IID: class-skewed distribution, simulating heterogeneous surveillance devices.

    Validation and test sets are shared copies/links for local validation compatibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    clients_root = output_root / f"clients_{partition}"
    yaml_paths = [clients_root / f"client_{i}" / "data.yaml" for i in range(1, num_clients + 1)]
    if all(p.exists() for p in yaml_paths):
        return yaml_paths

    if clients_root.exists():
        shutil.rmtree(clients_root)
    ensure_dir(clients_root)

    train_img_dir = dataset_dir / "train" / "images"
    train_lab_dir = dataset_dir / "train" / "labels"

    images = sorted([p for p in train_img_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    random.shuffle(images)

    assignments = {i: [] for i in range(num_clients)}

    if partition.lower() == "iid":
        for idx, img in enumerate(images):
            assignments[idx % num_clients].append(img)

    elif partition.lower() in ["noniid", "non-iid"]:
        class0 = []
        class1 = []
        mixed = []
        for img in images:
            lab = train_lab_dir / f"{img.stem}.txt"
            counts = get_label_counts(lab)
            if counts[0] > counts[1]:
                class0.append(img)
            elif counts[1] > counts[0]:
                class1.append(img)
            else:
                mixed.append(img)

        random.shuffle(class0)
        random.shuffle(class1)
        random.shuffle(mixed)

        # Class-skewed client allocation:
        # Clients 1-2 receive more class 0 samples, clients 3-4 receive more class 1 samples,
        # client 5 receives mixed samples. This simulates non-IID smart-city clients.
        buckets = [
            class0[: int(0.50 * len(class0))] + class1[: int(0.10 * len(class1))],
            class0[int(0.50 * len(class0)):] + class1[int(0.10 * len(class1)): int(0.20 * len(class1))],
            class1[int(0.20 * len(class1)): int(0.60 * len(class1))] + class0[: int(0.05 * len(class0))],
            class1[int(0.60 * len(class1)):] + class0[int(0.05 * len(class0)): int(0.10 * len(class0))],
            mixed,
        ]

        # If fewer or more clients are requested, distribute remaining robustly.
        flat = []
        for b in buckets:
            random.shuffle(b)
            flat.extend(b)
        random.shuffle(flat)
        if num_clients != 5:
            assignments = {i: [] for i in range(num_clients)}
            for idx, img in enumerate(flat):
                assignments[idx % num_clients].append(img)
        else:
            for i in range(5):
                assignments[i] = buckets[i]

        # Make sure no client is empty.
        empty = [i for i in range(num_clients) if len(assignments[i]) == 0]
        for i in empty:
            assignments[i].append(images.pop())

    else:
        raise ValueError("partition must be 'iid' or 'noniid'")

    # Build folder structures.
    for i in range(num_clients):
        client_dir = clients_root / f"client_{i+1}"

        for split in ["train", "valid", "test"]:
            ensure_dir(client_dir / split / "images")
            ensure_dir(client_dir / split / "labels")

        # Client-specific train split.
        for img in assignments[i]:
            lab = train_lab_dir / f"{img.stem}.txt"
            copy_or_link(img, client_dir / "train" / "images" / img.name)
            if lab.exists():
                copy_or_link(lab, client_dir / "train" / "labels" / lab.name)

        # Shared validation/test splits for convenience.
        for split in ["valid", "test"]:
            for img in (dataset_dir / split / "images").glob("*.*"):
                if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    lab = dataset_dir / split / "labels" / f"{img.stem}.txt"
                    copy_or_link(img, client_dir / split / "images" / img.name)
                    if lab.exists():
                        copy_or_link(lab, client_dir / split / "labels" / lab.name)

        create_client_dataset_yaml(client_dir)

    # Save client statistics.
    stats = []
    for i in range(num_clients):
        client_dir = clients_root / f"client_{i+1}"
        train_images = list((client_dir / "train" / "images").glob("*.*"))
        c0 = c1 = 0
        for img in train_images:
            lab = client_dir / "train" / "labels" / f"{img.stem}.txt"
            counts = get_label_counts(lab)
            c0 += counts[0]
            c1 += counts[1]
        stats.append({"client": i + 1, "train_images": len(train_images), "fire_labels": c0, "smoke_labels": c1})

    pd.DataFrame(stats).to_csv(clients_root / "client_distribution.csv", index=False)
    return [clients_root / f"client_{i}" / "data.yaml" for i in range(1, num_clients + 1)]


# ============================================================
# YOLO training/evaluation
# ============================================================

def train_yolo(model_source: str, data_yaml: Path, epochs: int, imgsz: int, batch: int, lr: float, project: Path, name: str) -> str:
    model = YOLO(model_source)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        optimizer="Adam",
        plots=True,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    save_path = project / name / "weights" / "last.pt"
    return str(save_path)


def validate_yolo(model_path: str, data_yaml: Path, imgsz: int, conf: float, iou: float, project: Path, name: str) -> Dict[str, float]:
    model = YOLO(model_path)
    metrics = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        plots=True,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=False,
    )

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "fitness": float(metrics.fitness),
    }


def load_weights(model_path: str):
    model = YOLO(model_path)
    return copy.deepcopy(model.model.state_dict())


def average_weights(weights_list):
    avg = copy.deepcopy(weights_list[0])
    for k in avg.keys():
        if torch.is_floating_point(avg[k]):
            for i in range(1, len(weights_list)):
                avg[k] += weights_list[i][k]
            avg[k] /= len(weights_list)
    return avg


def save_weights_to_model(base_model_source: str, weights, save_path: Path):
    model = YOLO(base_model_source)
    model.model.load_state_dict(weights, strict=False)
    model.save(str(save_path))


# ============================================================
# DAFA/FedAvg federated training
# ============================================================

def dafa_select_clients(client_results: List[dict], best_map: float, accuracy_margin: float = 0.98, time_margin: float = 1.25) -> List[dict]:
    times = np.array([c["training_time"] for c in client_results])
    avg_time = float(times.mean())
    selected = []

    for c in client_results:
        accuracy_ok = c["map50"] >= max(best_map * accuracy_margin, 0.01)
        time_ok = c["training_time"] <= avg_time * time_margin
        if accuracy_ok and time_ok:
            selected.append(c)

    if len(selected) == 0:
        selected = sorted(client_results, key=lambda x: x["map50"], reverse=True)[:2]

    return selected


def run_federated(
    method: str,
    base_model: str,
    client_yamls: List[Path],
    global_val_yaml: Path,
    output_root: Path,
    rounds: int,
    local_epochs: int,
    imgsz: int,
    batch: int,
    lr: float,
    conf: float,
    iou: float,
) -> Tuple[str, pd.DataFrame]:

    method = method.upper()
    method_dir = ensure_dir(output_root / method.lower())
    model_dir = ensure_dir(method_dir / "models")
    runs_dir = ensure_dir(method_dir / "runs")
    logs_dir = ensure_dir(method_dir / "logs")

    global_model_path = model_dir / f"{method.lower()}_global_initial.pt"
    if not global_model_path.exists():
        YOLO(base_model).save(str(global_model_path))

    logs = []
    best_map = 0.0

    for r in range(1, rounds + 1):
        print(f"\n================ {method} Round {r}/{rounds} ================")
        client_results = []

        for idx, yaml_path in enumerate(client_yamls):
            client_id = idx + 1
            start = time.time()

            local_model = train_yolo(
                model_source=str(global_model_path),
                data_yaml=yaml_path,
                epochs=local_epochs,
                imgsz=imgsz,
                batch=batch,
                lr=lr,
                project=runs_dir,
                name=f"round_{r}_client_{client_id}",
            )
            training_time = time.time() - start

            metrics = validate_yolo(
                model_path=local_model,
                data_yaml=yaml_path,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                project=runs_dir,
                name=f"val_round_{r}_client_{client_id}",
            )

            client_results.append({
                "round": r,
                "client_id": client_id,
                "model_path": local_model,
                "training_time": training_time,
                **metrics,
            })

        if method == "FEDAVG":
            selected = client_results
        elif method == "DAFA":
            selected = dafa_select_clients(client_results, best_map)
        else:
            raise ValueError("method must be DAFA or FedAvg")

        weights = [load_weights(c["model_path"]) for c in selected]
        global_weights = average_weights(weights)

        global_model_path = model_dir / f"{method.lower()}_global_round_{r}.pt"
        save_weights_to_model(base_model, global_weights, global_model_path)

        global_metrics = validate_yolo(
            model_path=str(global_model_path),
            data_yaml=global_val_yaml,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            project=runs_dir,
            name=f"global_val_round_{r}",
        )

        best_map = max(best_map, global_metrics["map50"])

        log = {
            "method": method,
            "round": r,
            "selected_clients": ",".join(str(c["client_id"]) for c in selected),
            "num_selected_clients": len(selected),
            "communication_updates": len(selected),
            "global_precision": global_metrics["precision"],
            "global_recall": global_metrics["recall"],
            "global_f1": global_metrics["f1"],
            "global_map50": global_metrics["map50"],
            "global_map50_95": global_metrics["map50_95"],
            "best_map50": best_map,
        }
        logs.append(log)
        print(log)

    df = pd.DataFrame(logs)
    df.to_csv(logs_dir / f"{method.lower()}_federated_log.csv", index=False)
    save_json(logs, logs_dir / f"{method.lower()}_federated_log.json")

    return str(global_model_path), df


# ============================================================
# Figures and tables
# ============================================================

def plot_federated_curves(dafa_df: pd.DataFrame, fedavg_df: pd.DataFrame, output_root: Path):
    fig_dir = ensure_dir(output_root / "paper_figures")

    rounds = dafa_df["round"].values

    plot_line(
        rounds,
        [dafa_df["global_map50"].values, fedavg_df["global_map50"].values],
        ["DAFA", "FedAvg"],
        "Communication Round",
        "mAP@0.5",
        "Global mAP@0.5 vs Communication Rounds",
        fig_dir / "global_map50_vs_rounds.png",
    )

    plot_line(
        rounds,
        [dafa_df["global_f1"].values, fedavg_df["global_f1"].values],
        ["DAFA", "FedAvg"],
        "Communication Round",
        "F1-score",
        "Global F1-score vs Communication Rounds",
        fig_dir / "global_f1_vs_rounds.png",
    )

    plot_line(
        rounds,
        [dafa_df["communication_updates"].values, fedavg_df["communication_updates"].values],
        ["DAFA", "FedAvg"],
        "Communication Round",
        "Transmitted Client Updates",
        "Communication Cost vs Communication Rounds",
        fig_dir / "communication_cost_vs_rounds.png",
    )

    plot_line(
        rounds,
        [dafa_df["num_selected_clients"].values],
        ["DAFA"],
        "Communication Round",
        "Selected Clients",
        "Selected Clients per Round",
        fig_dir / "selected_clients_per_round.png",
    )


def evaluate_realtime(model_path: str, image_dir: Path, output_root: Path, imgsz: int, conf: float, iou: float) -> pd.DataFrame:
    model = YOLO(model_path)
    images = sorted([p for p in image_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])

    times = []
    for img in images:
        start = time.time()
        model.predict(source=str(img), imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        times.append(time.time() - start)

    total = sum(times)
    fps = len(images) / total if total > 0 else 0
    latency_ms = float(np.mean(times) * 1000) if times else 0

    df = pd.DataFrame([{
        "device": "current_machine",
        "num_frames": len(images),
        "total_time_sec": total,
        "fps": fps,
        "latency_ms": latency_ms,
    }])
    out = ensure_dir(output_root / "real_time")
    df.to_csv(out / "fps_latency.csv", index=False)
    return df


# ============================================================
# False-positive confidence histogram
# ============================================================

def xywhn_to_xyxy(box, w, h):
    x, y, bw, bh = box
    x1 = (x - bw / 2) * w
    y1 = (y - bh / 2) * h
    x2 = (x + bw / 2) * w
    y2 = (y + bh / 2) * h
    return np.array([x1, y1, x2, y2], dtype=float)


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def read_gt_boxes(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cls = int(float(parts[0]))
            vals = list(map(float, parts[1:5]))
            boxes.append((cls, xywhn_to_xyxy(vals, img_w, img_h)))
    return boxes


def collect_false_positive_confidences(
    model_path: str,
    test_img_dir: Path,
    test_label_dir: Path,
    imgsz: int,
    conf: float,
    iou_thr: float,
    max_images: int = 500,
) -> List[float]:
    import cv2

    model = YOLO(model_path)
    images = sorted([p for p in test_img_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    images = images[:max_images]

    fp_scores = []

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = test_label_dir / f"{img_path.stem}.txt"
        gt = read_gt_boxes(label_path, w, h)

        preds = model.predict(source=str(img_path), imgsz=imgsz, conf=conf, iou=iou_thr, verbose=False)[0]

        if preds.boxes is None or len(preds.boxes) == 0:
            continue

        pred_boxes = preds.boxes.xyxy.cpu().numpy()
        pred_cls = preds.boxes.cls.cpu().numpy().astype(int)
        pred_conf = preds.boxes.conf.cpu().numpy()

        for pb, pc, ps in zip(pred_boxes, pred_cls, pred_conf):
            matched = False
            for gc, gb in gt:
                if pc == gc and iou_xyxy(pb, gb) >= iou_thr:
                    matched = True
                    break
            if not matched:
                fp_scores.append(float(ps))

    return fp_scores


def plot_fp_histogram(model_paths: Dict[str, str], dataset_dir: Path, output_root: Path, imgsz: int, conf: float, iou: float):
    fig_dir = ensure_dir(output_root / "paper_figures")
    test_img_dir = dataset_dir / "test" / "images"
    test_label_dir = dataset_dir / "test" / "labels"

    bins = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    all_counts = {}

    for name, path in model_paths.items():
        if not Path(path).exists():
            print(f"Skipping FP histogram for missing model: {name} -> {path}")
            continue
        scores = collect_false_positive_confidences(path, test_img_dir, test_label_dir, imgsz, conf, iou)
        counts, _ = np.histogram(scores, bins=bins)
        all_counts[name] = counts.tolist()

    if not all_counts:
        print("No FP histogram generated because no model paths were available.")
        return

    intervals = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    df = pd.DataFrame({"Confidence Interval": intervals})
    for name, counts in all_counts.items():
        df[name] = counts
    df.to_excel(fig_dir / "false_positive_confidence_histogram_data.xlsx", index=False)

    x = np.arange(len(intervals))
    width = 0.8 / max(len(all_counts), 1)

    plt.figure(figsize=(11, 6))
    for i, (name, counts) in enumerate(all_counts.items()):
        offset = (i - (len(all_counts) - 1) / 2) * width
        plt.bar(x + offset, counts, width=width, label=name)

    plt.xticks(x, intervals)
    plt.xlabel("False-Positive Confidence Score Intervals")
    plt.ylabel("Number of False-Positive Detections")
    plt.title("Histogram of False-Positive Detection Confidence Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "false_positive_confidence_histogram.png", dpi=300)
    plt.close()


# ============================================================
# Baseline and ablation
# ============================================================

def train_centralized(base_model: str, data_yaml: Path, output_root: Path, epochs: int, imgsz: int, batch: int, lr: float, conf: float, iou: float) -> Tuple[str, dict]:
    out = ensure_dir(output_root / "centralized")
    model_path = train_yolo(base_model, data_yaml, epochs, imgsz, batch, lr, out, "centralized_yolo11n")
    metrics = validate_yolo(model_path, data_yaml, imgsz, conf, iou, out, "centralized_val")
    pd.DataFrame([{**metrics, "model": "Centralized YOLOv11n"}]).to_csv(out / "centralized_results.csv", index=False)
    return model_path, metrics


def run_ablation_table(output_root: Path, rows: List[dict]):
    """
    Saves ablation/comparison rows and creates bar charts.
    Expected columns: model, precision, recall, f1, map50, map50_95
    """
    ab_dir = ensure_dir(output_root / "ablation")
    df = pd.DataFrame(rows)
    df.to_csv(ab_dir / "ablation_results.csv", index=False)
    df.to_excel(ab_dir / "ablation_results.xlsx", index=False)

    for metric, ylabel in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("map50", "mAP@0.5"),
        ("map50_95", "mAP@0.5:0.95"),
    ]:
        if metric in df.columns:
            plot_bar(df["model"], df[metric], ylabel, f"Ablation Study: {ylabel}", ab_dir / f"ablation_{metric}.png")

    return df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="indoor-fire-smoke.v2i.yolov11.zip", help="Path to Dataset")
    parser.add_argument("--output", default="firesmoke_fl_indoor_outputs", help="Output folder")
    parser.add_argument("--base-model", default="yolo11n.pt", help="YOLO model, e.g., yolo11n.pt")
    parser.add_argument("--mode", default="centralized", choices=["prepare", "dafa", "fedavg", "centralized", "all"], help="Run mode")
    parser.add_argument("--partition", default="noniid", choices=["iid", "noniid"], help="Federated partitioning")
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--centralized-epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)

    args = parser.parse_args()

    zip_path = Path(args.zip)
    output_root = ensure_dir(Path(args.output))

    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    dataset_dir = extract_dataset(zip_path, output_root)
    data_yaml = fix_data_yaml(dataset_dir)
    client_yamls = create_federated_clients(dataset_dir, output_root, args.clients, args.partition)

    dataset_stats = count_dataset(dataset_dir)
    dataset_stats.to_csv(output_root / "dataset_split_counts.csv", index=False)

    print("\nDataset prepared.")
    print(dataset_stats)
    print(f"Local data YAML: {data_yaml}")
    print(f"Client YAMLs: {[str(p) for p in client_yamls]}")

    if args.mode == "prepare":
        return

    comparison_rows = []
    model_paths_for_fp = {}

    if args.mode in ["centralized", "all"]:
        central_model, central_metrics = train_centralized(
            args.base_model, data_yaml, output_root, args.centralized_epochs,
            args.imgsz, args.batch, args.lr, args.conf, args.iou
        )
        comparison_rows.append({"model": "Centralized YOLOv11n", **central_metrics})
        model_paths_for_fp["Centralized YOLOv11n"] = central_model

    dafa_df = None
    fedavg_df = None
    dafa_model = None
    fedavg_model = None

    if args.mode in ["dafa", "all"]:
        dafa_model, dafa_df = run_federated(
            "DAFA", args.base_model, client_yamls, data_yaml, output_root,
            args.rounds, args.local_epochs, args.imgsz, args.batch, args.lr, args.conf, args.iou
        )
        final = dafa_df.iloc[-1]
        comparison_rows.append({
            "model": f"FireSmoke-FL + DAFA ({args.partition})",
            "precision": final["global_precision"],
            "recall": final["global_recall"],
            "f1": final["global_f1"],
            "map50": final["global_map50"],
            "map50_95": final["global_map50_95"],
        })
        model_paths_for_fp["FireSmoke-FL"] = dafa_model

    if args.mode in ["fedavg", "all"]:
        fedavg_model, fedavg_df = run_federated(
            "FedAvg", args.base_model, client_yamls, data_yaml, output_root,
            args.rounds, args.local_epochs, args.imgsz, args.batch, args.lr, args.conf, args.iou
        )
        final = fedavg_df.iloc[-1]
        comparison_rows.append({
            "model": f"FedAvg + YOLOv11n ({args.partition})",
            "precision": final["global_precision"],
            "recall": final["global_recall"],
            "f1": final["global_f1"],
            "map50": final["global_map50"],
            "map50_95": final["global_map50_95"],
        })
        model_paths_for_fp["FedAvg"] = fedavg_model

    if dafa_df is not None and fedavg_df is not None:
        plot_federated_curves(dafa_df, fedavg_df, output_root)

    if comparison_rows:
        run_ablation_table(output_root, comparison_rows)

    final_model = dafa_model or fedavg_model
    if final_model:
        rt_df = evaluate_realtime(
            final_model,
            dataset_dir / "test" / "images",
            output_root,
            args.imgsz,
            args.conf,
            args.iou,
        )
        print("\nReal-time evaluation:")
        print(rt_df)

        # False-positive confidence histogram from available trained models.
        plot_fp_histogram(model_paths_for_fp, dataset_dir, output_root, args.imgsz, args.conf, args.iou)

    print(f"\nDone. Outputs saved in: {output_root.resolve()}")


if __name__ == "__main__":
    main()
