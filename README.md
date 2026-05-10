# ✋ Hand Sign Alphabet Recognition

## Project Overview

A deep learning system that classifies hand sign gestures corresponding to alphabetic characters (A–Z) using convolutional neural networks.

The model uses transfer learning with the MobileNetV2 architecture and is trained on the ASL Alphabet dataset available on Kaggle:

[ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

![Sign Language](data/The-26-letters-and-10-digits-of-American-Sign-Language-ASL.png)

---

# 🚀 Getting Started

## Requirements

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager

## Install uv

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

---

# 📦 Project Setup

## Clone the repository

```bash
git clone git@github.com:Ryane-S/hand-gesture-recognition.git
cd hand-gesture-recognition
```

## Create and sync the virtual environment

```bash
uv sync
```

## Activate the virtual environment

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

---

# ▶️ Run the Application

Launch the webcam interface:

### Linux / macOS

```bash
python main.py
```

### Windows

```powershell
python src\collection\data_collector.py
```

---

# 🧠 Model Architecture

- Transfer Learning with MobileNetV2
- TensorFlow / Keras
- Real-time webcam inference
- American Sign Language alphabet recognition (A–Z)

---

# 📁 Dataset

This project uses the ASL Alphabet dataset from Kaggle:

[Kaggle Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)