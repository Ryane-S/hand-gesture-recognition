# ✋ Hand Sign Alphabet Recognition

## Project Overview

A deep learning system that classifies hand sign gestures corresponding to alphabetic characters (A-Z) using convolutional neural networks. The model employs transfer learning with MobileNetV2 architecture, trained on a dataset available on Kaggle : https://www.kaggle.com/datasets/grassknoted/asl-alphabet

![Sign Language](data/The-26-letters-and-10-digits-of-American-Sign-Language-ASL.png) 

## Installation on Linux/Mac

### Python installation
This project requires **Python 3.11** or earlier versions.
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev python3.11-pip 
```

Naviguate to your desired working directory and clone this repository.
```bash
git clone git@github.com:Ryane-S/hand-gesture-recognition.git
```

### Setup
* Create a virtual environment and activate it:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

* Install dependencies:
```bash
pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

* Launch the webcam interface:
```bash
python3 ./src/collection/data_collector.py
```