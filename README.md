# Video Anomaly Detection with CNN-LSTM | Surveillance & Federated Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Deep learning pipeline for automatic video anomaly detection** using a hybrid **CNN-LSTM** architecture. Trains on benchmark surveillance datasets (**UCSD**, **UCF-Crime**, **UBnormal**), then combines site-specific models with **federated learning** and **particle swarm optimization (PSO)** for a stronger global detector.

> **Freelance project** — End-to-end computer vision solution delivered for a client: dataset integration, model training, evaluation, and multi-model fusion for real-world surveillance scenarios.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Supported Datasets](#supported-datasets)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Federated Learning & Swarm Intelligence](#federated-learning--swarm-intelligence)
- [Results](#results)
- [Technologies](#technologies)
- [Author](#author)
- [Keywords](#keywords)

---

## Overview

This repository implements **video-based anomaly detection** for security and surveillance use cases. Short clips are represented as sequences of grayscale frames; a **time-distributed CNN** extracts spatial features per frame, and an **LSTM** models temporal patterns to classify sequences as **normal** or **anomalous**.

The workflow covers:

1. **Per-dataset training** — Independent CNN-LSTM models on UCSD, UCF-Crime, and UBnormal  
2. **Class balancing** — Augmentation and downsampling for imbalanced anomaly data  
3. **Model fusion** — Accuracy-weighted **federated averaging** and **PSO-based** weight search  
4. **Evaluation** — Accuracy, precision, recall, F1, and ROC-AUC on held-out validation sets  

---

## Key Features

| Feature | Description |
|--------|-------------|
| **CNN-LSTM hybrid** | Spatial CNN + temporal LSTM for clip-level binary classification |
| **Multi-dataset support** | UCSD (frames), UCF-Crime (optical flow `.npy`), UBnormal (MP4 videos) |
| **Data augmentation** | Rotation, shift, and zoom on minority (anomaly) sequences |
| **GPU training** | Automatic fallback to CPU if GPU is unavailable |
| **Federated learning** | Weighted fusion of three dataset-specific models into `global_fl.keras` |
| **Swarm intelligence** | Particle swarm optimization over model weights → `global_swarm.keras` |
| **Metrics & plots** | Confusion matrix and ROC curves (UBnormal training pipeline) |

---

## Model Architecture

```
Input: (batch, 10 frames, 64×64, 1 channel)
    │
    ▼
TimeDistributed Conv2D(32) → MaxPool
TimeDistributed Conv2D(64) → MaxPool
    │  (+ Flatten on UCF / UBnormal variants)
    ▼
LSTM(100 units) → Dropout(0.5)
    ▼
Dense(1, sigmoid)  →  Normal (0) / Anomaly (1)
```

| Hyperparameter | Value |
|----------------|-------|
| Sequence length | 10 frames |
| Frame size | 64 × 64 grayscale |
| Optimizer | Adam (lr = 1e-4) |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Early stopping | Validation accuracy / recall; target 98% val accuracy |

---

## Supported Datasets

| Dataset | Folder | Input format | Notes |
|---------|--------|----------------|-------|
| [UCSD Pedestrian](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) | `UCSD/` | Image paths in `normal.txt`, `anomalies.txt` | Surveillance pedestrian anomalies |
| [UCF-Crime](https://www.crcv.ucf.edu/projects/real-world/) | `UCF/` | Optical flow `.npy` under `all_flows/` | Split lists in `splits/` |
| [UBnormal](https://github.com/luizgh/UBnormal) | `UBnormal/` | MP4 paths in `normal.txt`, `abnormal.txt` | Synthetic abnormal/normal scenes |

Update path lists in each dataset’s `.txt` files to match your local download locations before training.

---

## Project Structure

```
Anomalies-Detection-Using-CNN-LSTM/
├── UCSD/
│   ├── train.py          # Train on UCSD frame sequences → ucsd.keras
│   ├── test.py           # Evaluate and write result.txt
│   ├── normal.txt
│   └── anomalies.txt
├── UCF/
│   ├── train.py          # Train on UCF optical flow → ucf.keras
│   ├── test.py
│   └── splits/           # train_001.txt, test_001.txt
├── UBnormal/
│   ├── train.py          # Train on UBnormal videos → UBnormal.keras
│   ├── test.py
│   ├── normal.txt
│   └── abnormal.txt
├── Faderated_Learning/
│   ├── faderated_learning_setup.py   # Federated weight averaging
│   ├── swarm.py                      # PSO over model weights
│   └── eval.py                       # Compare global FL vs swarm models
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, recommended for training)

### Dependencies

```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib
```

### Clone the repository

```bash
git clone https://github.com/danishjavedcodes/Anomalies-Detection-Using-CNN-LSTM.git
cd Anomalies-Detection-Using-CNN-LSTM
```

---

## Usage

### 1. Prepare dataset paths

Edit the `.txt` manifest files in `UCSD/`, `UCF/`, and `UBnormal/` so each line points to a valid image, `.npy`, or video on your machine.

For **UCF**, place preprocessed optical flow arrays under `UCF/all_flows/` as expected by `train.py`.

### 2. Train per-dataset models

```bash
cd UCSD && python train.py && python test.py && cd ..
cd UCF  && python train.py && python test.py && cd ..
cd UBnormal && python train.py && python test.py && cd ..
```

Each `test.py` writes validation **accuracy** to `result.txt` for federated weighting.

### 3. Federated learning fusion

```bash
cd Faderated_Learning
python faderated_learning_setup.py   # → global_fl.keras
```

### 4. Swarm intelligence optimization

```bash
python swarm.py                        # → global_swarm.keras
```

### 5. Evaluate global models

```bash
python eval.py
```

---

## Federated Learning & Swarm Intelligence

### Federated averaging

Per-dataset validation accuracies from `result.txt` define fusion weights:

\[
w_i = \frac{\text{accuracy}_i}{\sum_j \text{accuracy}_j}
\]

Layer weights from `UBnormal.keras`, `ucsd.keras`, and `ucf.keras` are combined into a single **global federated model** (`global_fl.keras`).

### Particle swarm optimization (PSO)

`swarm.py` treats each dataset-specific model as a particle, updates weights using inertia and cognitive/social terms, and tracks a **global best** configuration by validation accuracy across UBnormal, UCSD, and UCF validation sets. Output: `global_swarm.keras`.

---

## Results

Example validation accuracy on **UBnormal** (from committed `result.txt`):

| Model | Metric |
|-------|--------|
| UBnormal CNN-LSTM | **~95.6%** validation accuracy |

Run `test.py` in each dataset folder after training to regenerate `result.txt` and `eval.py` for global model comparison on your hardware.

---

## Technologies

- **TensorFlow / Keras** — CNN-LSTM model definition and training  
- **OpenCV** — Video frame extraction and resizing  
- **NumPy** — Sequence tensors and federated weight math  
- **scikit-learn** — Metrics, train/validation splits  
- **Matplotlib** — Confusion matrix and ROC visualizations  

---

## Author

**[danishjavedcodes](https://github.com/danishjavedcodes)**

Freelance **machine learning & computer vision** work: video anomaly detection, multi-dataset deep learning, and federated / swarm-based model aggregation for surveillance applications.

Questions or collaboration: open an [issue](https://github.com/danishjavedcodes/Anomalies-Detection-Using-CNN-LSTM/issues) or reach out via GitHub.

---

## Keywords

`video anomaly detection` · `CNN-LSTM` · `surveillance AI` · `deep learning` · `UCSD anomaly dataset` · `UCF-Crime` · `UBnormal` · `federated learning` · `particle swarm optimization` · `TensorFlow` · `computer vision` · `abnormal event detection` · `freelance ML project`

---

## License

This project is provided for research and portfolio purposes. Dataset licenses apply separately (UCSD, UCF, UBnormal). Add a `LICENSE` file if you distribute under a specific open-source terms.
