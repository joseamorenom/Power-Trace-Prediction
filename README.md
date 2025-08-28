AI-Powered Power Trace Prediction for Digital Circuits
This repository contains the source code, data structure, and documentation for a research project focused on predicting dynamic power consumption traces for digital circuits using a multimodal deep learning architecture. The project demonstrates a complete pipeline, from raw EDA data preprocessing to model training, evaluation, and inference on new, unseen data.

📝 Project Overview
Traditional power analysis in Electronic Design Automation (EDA) is a critical but computationally intensive process, often creating a significant bottleneck in the hardware design cycle. This project explores the feasibility of using an AI model to provide fast and accurate power waveform predictions directly from post-synthesis netlists and simulation activity files (VCDs).

The core of this project is a multimodal architecture that learns from two distinct data types simultaneously to form a holistic understanding of power consumption:

Static Circuit Structure: The physical blueprint of gates and wires, represented as a graph.

Dynamic Switching Activity: The real-time signal transitions during operation, represented as a time-series matrix.

By fusing these modalities, the model learns the complex relationship between a circuit's structure, its real-time behavior, and its resulting power consumption.

📂 Repository Structure
The project is organized into a standard machine learning structure for clarity and reproducibility.

Power-Trace-Prediction/
│
├── data/
│   ├── raw/
│   │   ├── aes_original/      # Original 800-sample AES dataset
│   │   └── sayid_tests/       # New test/retraining VCDs
│   └── processed/
│       └── processed_dataset_retrain/ # Preprocessed .pt files
│
├── results/
│   ├── retrained_model/
│   │   ├── best_model_retrain.pth   # The final trained model weights
│   │   └── plots/                 # Visualization plots
│   └── surprise_test_predictions/
│       ├── prediction_trace_0.png # Inference results
│       └── prediction_trace_0.data
│
├── src/
│   ├── utils.py             # Shared classes (Dataset, Model) and functions
│   ├── create_dataset.py    # Script to preprocess raw data
│   ├── train.py             # Script to train the model
│   ├── predict.py           # Script for running inference on new data
│   └── visualize.py         # Script to generate comparison plots
│
├── scripts/
│   ├── organize_raw_data.py
│   ├── analyze_netlist_stats.py
│   └── compare_vcd_activity.py
│
├── requirements.txt         # List of Python dependencies
└── README.md                # This file

🛠️ Setup and Installation
This project was developed using Python 3.8 and the Conda package manager.

Clone the repository:

git clone https://github.com/[YourUsername]/Power-Trace-Prediction.git
cd Power-Trace-Prediction

Create and activate the Conda environment:

conda create --name power_trace_env python=3.8
conda activate power_trace_env

Install required Python packages:

pip install -r requirements.txt

Install EDA Tools: This project requires Yosys for Verilog parsing. Please follow the official installation guide for your operating system (e.g., via MSYS2 on Windows). Ensure that the yosys executable is available in your system's PATH.

🚀 Usage Pipeline
The project is divided into three main executable scripts located in the src/ directory. They should be run in the following order.

1. Data Preprocessing (create_dataset.py)
This script is the first and most critical step. It takes all the raw data, preprocesses it, and saves it as a collection of PyTorch-ready .pt files.

Input: Raw data located in data/raw/.

Output: Processed .pt files, a splits.json file, a cell_types.json dictionary, and norm_stats.json for normalization, all saved in data/processed/processed_dataset_retrain.

To run:

python src/create_dataset.py

2. Model Training (train.py)
This script loads the preprocessed data, defines the PowerPredictor model architecture, and runs the training and validation loops.

Input: The processed dataset from the previous step.

Output: The best trained model weights (best_model_retrain.pth) and evaluation metrics, saved in the results/retrained_model/ folder.

To run:

python src/train.py

3. Inference on New Data (predict.py)
This script loads a trained model and uses it to generate power trace predictions for new, unseen VCD files.

Input: A trained model (.pth), the cell_types.json dictionary, and new VCD files. (Paths are configured inside the script).

Output: Predicted power traces saved as both .data files and .png plots.

To run:

python src/predict.py

🤖 Model Architecture
The PowerPredictor is a multimodal architecture composed of three main blocks:

GNNEncoder (Structural Analyst):

Architecture: A two-layer Graph Attention Network (GAT).

Function: It processes the circuit graph, using an attention mechanism to learn the importance of different nodes and connections. It aggregates this information into a single 128-dimensional vector (graph embedding) that summarizes the circuit's static structure.

ActivityEncoder (Behavioral Analyst):

Architecture: A hybrid 1D CNN + Bidirectional LSTM.

Function: A 1D CNN first scans the activity matrix at each time step to find important local spatial patterns. An LSTM then processes the sequence of these patterns to understand the overall temporal behavior.

Output: A 128-dimensional vector (activity embedding) summarizing the circuit's dynamic activity.

PredictionHead (Decision Maker):

Architecture: A Multi-Layer Perceptron (MLP).

Function: It concatenates the graph and activity embeddings into a single 256-dimensional vector. This vector is then passed through two fully connected layers to produce the final 400-point power trace prediction.

📈 Experiments & Results
Our final retrained model, trained on a diverse dataset of 810 samples, achieved the following performance on the test set:

Mean Squared Error (MSE): [Introduce aquí el MSE final de tu modelo re-entrenado]

Dynamic Time Warping (DTW): [Introduce aquí el DTW final de tu modelo re-entrenado]

This demonstrates the model's ability to accurately predict power traces with high fidelity in both value and shape, while generalizing across different activity distributions.