# AI-Powered Power Trace Prediction for Digital Circuits ⚡

This repository contains the source code, data structure, and documentation for a research project focused on predicting dynamic power consumption traces for digital circuits using a multimodal deep learning architecture. The project demonstrates a complete pipeline, from raw EDA data preprocessing to model training, evaluation, and inference on new, unseen data.

---

## 📝 Project Overview

Traditional power analysis in Electronic Design Automation (EDA) is a critical but computationally intensive process, often creating a significant bottleneck in the hardware design cycle.  
This project explores the feasibility of using an AI model to provide fast and accurate power waveform predictions directly from post-synthesis netlists and simulation activity files (VCDs).

The core of this project is a multimodal architecture that learns from two distinct data types simultaneously:

1. **Static Circuit Structure:** The physical blueprint of gates and wires, represented as a graph.
2. **Dynamic Switching Activity:** The real-time signal transitions during operation, represented as a time-series matrix.

By fusing these modalities, the model learns the complex relationship between a circuit's structure, its real-time behavior, and its resulting power consumption.

---

## 📂 Repository Structure

```bash
Power-Trace-Prediction/
│
├── data/
│   ├── raw/
│   │   ├── aes_original/           # Original 800-sample AES dataset
│   │   └── sayid_tests/            # New test/retraining VCDs
│   └── processed/
│       └── processed_dataset_retrain/ # Preprocessed .pt files
│
├── results/
│   ├── retrained_model/
│   │   ├── best_model_retrain.pth  # Final trained model weights
│   │   └── plots/                  # Visualization plots
│   └── surprise_test_predictions/
│       ├── prediction_trace_0.png  # Inference results
│       └── prediction_trace_0.data
│
├── src/
│   ├── utils.py             # Shared classes (Dataset, Model) and functions
│   ├── create_dataset.py    # Preprocess raw data
│   ├── train.py             # Train the model
│   ├── predict.py           # Inference on new data
│   └── visualize.py         # Generate comparison plots
│
├── scripts/
│   ├── organize_raw_data.py
│   ├── analyze_netlist_stats.py
│   └── compare_vcd_activity.py
│
├── requirements.txt         # Python dependencies
└── README.md
```

## 🛠️ Setup and Installation

This project was developed using **Python 3.8** and the **Conda** package manager.

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/Power-Trace-Prediction.git
cd Power-Trace-Prediction
```
### 2. Download the Dataset
The dataset (raw data, processed `.pt` files, and results) is hosted externally.  

- Download from: **[PASTE YOUR LINK HERE]**  
- Unzip the archive.  
- Place the `data/` and `results/` folders into the root of this repository.  

Your final local directory should look like:
```bash
Power-Trace-Prediction/
├── data/
├── results/
├── src/
├── scripts/
└── README.md
```
### 3. Set up the Environment
```bash
# Create and activate the environment
conda create --name power_trace_env python=3.8
conda activate power_trace_env

# Install dependencies
pip install -r requirements.txt
```

## 4.  Install EDA Tools

This project requires Yosys for Verilog parsing. Ensure the yosys executable is available in your system's PATH.


## 🚀 Usage Pipeline
All executable scripts are located in the src/ folder. They are designed to be run from the root directory of the repository.

### 1. Data Preprocessing (create_dataset.py)
This script is the first and most critical step. It takes all the raw data, preprocesses it, and saves it as a collection of PyTorch-ready .pt files.

Input: Raw data located in data/raw/.
Output: Processed .pt files, a splits.json file, a cell_types.json dictionary, and norm_stats.json for normalization, all saved in data/processed/processed_dataset_retrain.

To run:
```Bash
python src/create_dataset.py
```

### 2. Model Training (train.py)
This script loads the preprocessed data, defines the PowerPredictor model architecture, and runs the training and validation loops.

Input: The processed dataset from the previous step.
Output: The best trained model weights (best_model_retrain.pth) and evaluation metrics, saved in the results/retrained_model/ folder.

To run:
```Bash
python src/train.py
```

### 3. Inference on New Data (predict.py)
This script loads a trained model and uses it to generate power trace predictions for new, unseen VCD files.

Input: A trained model (.pth), the cell_types.json dictionary, and new VCD files. (Paths are configured inside the script).
Output: Predicted power traces saved as both .data files and .png plots.

To run:
```Bash
python src/predict.py
```
