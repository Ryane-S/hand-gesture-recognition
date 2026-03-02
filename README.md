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

### Setup
* Navigate to your desired working directory and clone this repository.
```bash
git clone git@github.com:Ryane-S/hand-gesture-recognition.git
cd hand-gesture-recognition
```

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
python ./src/collection/data_collector.py
```

## Installation on Windows

### Python installation
1. Download Python 3.11 from the [official website](https://www.python.org/downloads/windows/)
2. During installation, **check** "Add Python to PATH"
3. Verify installation:
```bash
python --version
```

### Setup
* Navigate to your desired working directory and clone this repository.
```bash
git clone git@github.com:Ryane-S/hand-gesture-recognition.git
cd hand-gesture-recognition
```

* Create a virtual environment and activate it:
```bash
python -m venv .venv
.venv\Scripts\activate
```

* Install dependencies:
```bash
python -m pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

* Launch the webcam interface:
```bash
python src\collection\data_collector.py
```

## Installation with Anaconda

### Anaconda installation (Windows)
1. Download and install [Anaconda](https://www.anaconda.com/download) for Windows
2. Open **Anaconda Prompt** as administrator

### Anaconda installation (Linux)
1. Download the Miniconda installer for Linux from the [official website](https://docs.conda.io/en/latest/miniconda.html)
2. Install via terminal :
```bash
chmod u+x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
3. Follow the prompts (accept license, confirm location, say "yes" to init)
4. Restart terminal or run :
```bash
source ~/.bashrc
```

### Setup with Conda
* Clone the repository:
```bash
git clone git@github.com:Ryane-S/hand-gesture-recognition.git
cd hand-gesture-recognition
```

* Create and configure the conda environment
```bash
conda create --name hgr python=3.11 -y
conda activate hgr
```

* Install dependencies
```bash
python -m pip install --upgrade pip
pip install --upgrade -r requirements.txt
```

* Depending on your OS, refer to previous sections to launch the webcam interface !