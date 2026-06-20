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

## Repository Structure

```text
FireSmoke-FL/
│
├── firesmoke_fl.py
├── requirements.txt
├── README.md
│
├── datasets/
│   └── indoor-fire-smoke.v2i.yolov11.zip
│
├── outputs/
│   ├── models/
│   ├── logs/
│   ├── figures/
│   ├── curves/
│   ├── ablation/
│   ├── federated/
│   └── real_time/
│
└── paper/
```

---

## Dataset

This repository was evaluated using:

Indoor Fire Smoke Dataset (Roboflow Universe)

https://universe.roboflow.com/firedet-uk6sb/indoor-fire-smoke-eqopn

The framework also supports additional fire and smoke datasets exported in YOLO format.

---

## Installation

Create a new environment:

```bash
conda create -n firesmoke python=3.10
conda activate firesmoke
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Preparing the Dataset

Place the dataset ZIP file in:

```text
datasets/
```

Then run:

```bash
python firesmoke_fl.py --mode prepare
```

---

## Training

### FireSmoke-FL (DAFA)

```bash
python firesmoke_fl.py \
    --mode dafa \
    --partition noniid
```

### FedAvg Baseline

```bash
python firesmoke_fl.py \
    --mode fedavg \
    --partition noniid
```

### Centralized Baseline

```bash
python firesmoke_fl.py \
    --mode centralized
```

### Full Experimental Pipeline

```bash
python firesmoke_fl.py \
    --mode all \
    --partition noniid
```

---

## Citation

If you use this code, please cite:

Fahad Alblehai, et al.

FireSmoke-FL: A Privacy-Preserving Federated Learning Framework for Real-Time Fire and Smoke Detection.

Scientific Reports, 2026.

---

## License

MIT License
