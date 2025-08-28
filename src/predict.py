# predict.py
#
# This script performs inference using a pre-trained PowerPredictor model.
# It is designed to take new VCD activity files for a known circuit,
# preprocess them, and generate the corresponding power trace predictions.
#
# Workflow:
#   1. Load the pre-trained model weights and the original circuit's graph structure.
#   2. Scan a specified directory for new .vcd files to process.
#   3. For each VCD, create the corresponding activity matrix.
#   4. Feed the graph and activity matrix to the model to get a prediction.
#   5. Save the predicted trace as a .data file and a .png plot.
#

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from vcdvcd import VCDVCD
import pandas as pd
from tqdm import tqdm

# Import from the toolbox
from utils import PowerPredictor, crear_matriz_actividad

# --- Paths ---
# Base project folder
RUTA_BASE_PROYECTO = r'C:\Users\amesa\OneDrive\Delaware\Power Traces'

# Path to the trained model file we want to use for inference
RUTA_MODELO_GUARDADO = os.path.join(RUTA_BASE_PROYECTO, 'Data', 'Power_trace_jose', 'best_model_cnn_lstm.pth')

# Paths to the original dataset assets needed for context
CARPETA_DATASET_PROCESADO = os.path.join(RUTA_BASE_PROYECTO, 'Data', 'Power_trace_jose', '4_processed_dataset')
RUTA_JSON_ORIGINAL = os.path.join(RUTA_BASE_PROYECTO, 'Data', 'Power_trace_jose', '1_source_files', 'aes_netlist.json')

# --- INPUT/OUTPUT FOLDERS ---
# This is the folder where you should place the new VCD files for prediction
CARPETA_NUEVOS_VCD = os.path.join(RUTA_BASE_PROYECTO, 'Data', 'Power_trace_jose', 'Sayid Tests', 'Third Batch')

# The script will create this subfolder to save the results
CARPETA_PREDICCIONES = os.path.join(CARPETA_NUEVOS_VCD, 'predictions')
os.makedirs(CARPETA_PREDICCIONES, exist_ok=True)

# --- Model Hyperparameters (must match the trained model) ---
TRACE_LENGTH = 400


#HELPER FUNCTIONS

def save_power_trace_data(predicted_trace, output_path, start_time=325.0, time_step=650.0):
    """
    Saves the predicted power trace to a .data file in the original format.
    
    Args:
        predicted_trace (np.array): The 1D array of predicted power values.
        output_path (str): The file path to save the data to.
        start_time (float): The initial timestamp for the trace.
        time_step (float): The time increment between points.
    """
    with open(output_path, 'w') as f:
        f.write("# Predicted Power Profile using AI Model\n")
        current_time = start_time
        for power_value in predicted_trace:
            f.write(f"{current_time:.1f} {power_value:.12f}\n")
            current_time += time_step


#Main executn
def main():
    """
    Main function to run the inference pipeline.
    """
    # Set up the device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Load the static graph structure and net map from the original dataset.
    # We only need to do this once, as all predictions are for the same circuit.
    print("Loading original graph structure and net map...")
    try:
        # Load the net map from the original JSON for VCD signal matching
        with open(RUTA_JSON_ORIGINAL, 'r') as f:
            module_data = next(iter(json.load(f)['modules'].values()))
        netnames_json = list(module_data['netnames'].keys())
        net_map_original = {i: name.split('.')[-1] for i, name in enumerate(netnames_json)}
        
        # Load a template data object to get the graph structure (x, edge_index)
        data_template = torch.load(os.path.join(CARPETA_DATASET_PROCESADO, 'sample_000.pt'))
        graph_base = data_template.clone()
        del graph_base.activity
        del graph_base.power_trace
    except FileNotFoundError as e:
        print(f"Error: Could not find the processed dataset or JSON file. Please run create_dataset.py first.")
        print(f"Details: {e}")
        return

    # Step 2: Initialize the model architecture and load the trained weights.
    print(f"Loading trained model from: {RUTA_MODELO_GUARDADO}")
    model = PowerPredictor(
        gnn_in=graph_base.num_node_features,
        gnn_hidden=64,
        num_nodes=graph_base.num_nodes,
        cnn_out=32,
        lstm_hidden=128,
        common_embedding_dim=128,
        output_len=TRACE_LENGTH
    ).to(device)
    
    model.load_state_dict(torch.load(RUTA_MODELO_GUARDADO))
    model.eval() # Set the model to evaluation mode
    print(" -> Model loaded successfully.")

    # Step 3: Find all new .vcd files in the target directory.
    vcds_to_process = [f for f in os.listdir(CARPETA_NUEVOS_VCD) if f.endswith('.vcd')]
    if not vcds_to_process:
        print(f"\nNo .vcd files found in the folder: '{CARPETA_NUEVOS_VCD}'")
        return

    print(f"\n--- Found {len(vcds_to_process)} new VCDs to process ---")

    # Step 4: Loop through each new VCD and generate a prediction.
    for vcd_filename in tqdm(vcds_to_process, desc="Generating predictions"):
        vcd_path = os.path.join(CARPETA_NUEVOS_VCD, vcd_filename)
        
        # Preprocess the VCD into an activity matrix
        num_cells = (graph_base.x.sum(dim=1) > 0).sum().item()
        dummy_df_power = pd.DataFrame({'Time': np.linspace(0, (TRACE_LENGTH-1)*650, TRACE_LENGTH)})
        
        vcd = VCDVCD(vcd_path)
        activity_matrix = crear_matriz_actividad(vcd, dummy_df_power, graph_base.num_nodes, num_cells, net_map_original)
        
        # Assemble the final data packet for the model
        prediction_packet = graph_base.clone()
        prediction_packet.activity = activity_matrix
        
        # Use a DataLoader to create a batch of size 1, which is what the model expects
        loader = DataLoader([prediction_packet], batch_size=1)
        batch = next(iter(loader)).to(device)

        # Get the prediction from the model
        with torch.no_grad():
            predicted_trace = model(batch).cpu().numpy().flatten()
            
        base_filename = os.path.splitext(vcd_filename)[0]
        
        # Save the plot
        plot_path = os.path.join(CARPETA_PREDICCIONES, f'{base_filename}_prediction.png')
        plt.figure(figsize=(14, 7))
        plt.plot(predicted_trace, label='Predicted Power Trace', color='green')
        plt.title(f'Power Prediction for {vcd_filename}')
        plt.xlabel('Time Steps')
        plt.ylabel('Predicted Power')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()

        # Save the .data file
        data_path = os.path.join(CARPETA_PREDICCIONES, f'{base_filename}_prediction.data')
        save_power_trace_data(predicted_trace, data_path)

    print(f"\n--- Predictions complete ---")
    print(f"Results have been saved in the folder: '{CARPETA_PREDICCIONES}'")

if __name__ == "__main__":
    main()
