# visualize.py
#
# This script is our visual check on the model's performance.
# It loads a trained model, runs inference on a few random samples from the
# test set, and generates plots comparing the predicted power trace against the
# actual ground truth trace.
#


import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader

# Import our custom tools from the toolbox
from utils import PowerTraceDataset, PowerPredictor


# --- Paths ---
# Pointing to the folders for our retrained model experiment
RUTA_BASE_NUEVO_MODELO = r'C:\Users\amesa\OneDrive\Delaware\Power Traces\Data\Power_trace_jose\Sayid Tests\New Model'
CARPETA_DATASET_PROCESADO = os.path.join(RUTA_BASE_NUEVO_MODELO, 'processed_dataset_retrain')
RUTA_SPLITS_JSON = os.path.join(CARPETA_DATASET_PROCESADO, 'splits.json')
RUTA_MODELO_GUARDADO = os.path.join(RUTA_BASE_NUEVO_MODELO, 'best_model_retrain.pth')

# --- Output Folder ---
# A new folder to save the comparison plots
CARPETA_GRAFICOS = os.path.join(RUTA_BASE_NUEVO_MODELO, 'results_visualization')
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

# --- Hyperparameters (must match the model we are loading) ---
HIDDEN_DIM_GNN = 64 
HIDDEN_DIM_LSTM = 128 

#Main processing functions

def visualize_predictions(model, dataset, device, num_samples=5):
    """
    Takes a trained model and a dataset, and generates comparative plots.
    """
    model.eval() # Put the model in "testing" mode
    print(f"Generating {num_samples} example plots from the test set...")

    # Pick a few random samples from the dataset to visualize
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, sample_idx in enumerate(sample_indices):
        data = dataset[sample_idx]
        
        # Use a DataLoader to create a batch of size 1. This is the format
        # the model expects, even for a single prediction.
        loader = DataLoader([data], batch_size=1)
        batch = next(iter(loader))
        batch = batch.to(device)

        with torch.no_grad(): # Turn off learning for inference
            prediction = model(batch)
            
        # Move data to the CPU and convert to NumPy for plotting
        actual_trace = batch.power_trace.cpu().numpy().flatten()
        predicted_trace = prediction.cpu().numpy().flatten()
        
        time_axis = np.arange(len(actual_trace))

        # Plotting
        plt.figure(figsize=(14, 7))
        plt.plot(time_axis, actual_trace, label='Actual Power Trace', color='blue', linewidth=2, alpha=0.8)
        plt.plot(time_axis, predicted_trace, label='Predicted Power Trace', color='red', linestyle='--', linewidth=2)
        plt.title(f'Power Trace Comparison (Test Sample #{sample_idx})')
        plt.xlabel('Time Steps')
        plt.ylabel('Power')
        plt.legend()
        plt.grid(True)
        
        # Save the figure to the output folder
        save_path = os.path.join(CARPETA_GRAFICOS, f'comparison_plot_sample_{sample_idx}.png')
        plt.savefig(save_path)
        plt.close() # Close the plot to free up memory
        
    print(f"Plots saved successfully in the '{CARPETA_GRAFICOS}' folder!")

def main():
    """
    Main function to run the visualization pipeline.
    """
    # Load the test set indices
    with open(RUTA_SPLITS_JSON, 'r') as f:
        splits = json.load(f)
    test_indices = splits['test']
    
    # Create the test dataset object
    test_dataset = PowerTraceDataset(CARPETA_DATASET_PROCESADO, test_indices)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the model architecture. This MUST match the architecture
    # of the model we saved during training.
    first_sample = test_dataset[0]
    model = PowerPredictor(
        gnn_in=first_sample.num_node_features,
        gnn_hidden=HIDDEN_DIM_GNN,
        num_nodes=first_sample.activity.shape[1],
        cnn_out=32,
        lstm_hidden=HIDDEN_DIM_LSTM,
        common_embedding_dim=128,
        output_len=len(first_sample.power_trace)
    ).to(device)
    
    # Load the trained weights into the model structure
    print(f"Loading best model from: {RUTA_MODELO_GUARDADO}")
    model.load_state_dict(torch.load(RUTA_MODELO_GUARDADO))
    
    # Generate the visualizations
    visualize_predictions(model, test_dataset, device)

if __name__ == "__main__":
    # Check if matplotlib is installed before running
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed. Please install it with: pip install matplotlib")
    else:
        main()
