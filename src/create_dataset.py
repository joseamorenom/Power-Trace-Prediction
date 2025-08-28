# create_dataset.py
#
# This is the main preprocessing script. It takes all the raw engineering data
# (Verilog, VCDs, power .data files) and turns it into a clean, processed dataset
# of .pt files that our PyTorch model can actually use.
#
# Main steps:
#   1. Use Yosys to parse the netlist into a simple JSON.
#   2. Build the master circuit graph and cell type dictionary from that JSON.
#   3. Combine all our VCD/power trace samples into one big list.
#   4. Process every single sample into a final data packet.
#   5. Save everything out, including the train/val/test splits.
#

import os
import json
import subprocess
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vcdvcd import VCDVCD

# Import the custom tools from the toolbox
from utils import construir_grafo_desde_json, crear_matriz_actividad


# --- Input Paths ---
# The main folder for the original AES dataset (This will have to be changed for a new PC)
RUTA_BASE_ORIGINAL = r'C:\Users\amesa\OneDrive\Delaware\Power Traces\Data\Power_trace_jose'
CARPETA_FUENTE_ORIGINAL = os.path.join(RUTA_BASE_ORIGINAL, '1_source_files')
CARPETA_MUESTRAS_ORIGINALES = os.path.join(RUTA_BASE_ORIGINAL, '3_dataset_samples')
RUTA_JSON_NETLIST = os.path.join(CARPETA_FUENTE_ORIGINAL, 'aes_netlist.json')

# The folder with the 10 new VCDs we want to add to our training data
CARPETA_NUEVOS_VCD_TRAIN = os.path.join(RUTA_BASE_ORIGINAL, 'Sayid Tests', 'Third Batch')

# --- Output Paths ---
# Where all the final processed data for our new experiment will live
RUTA_BASE_NUEVO_MODELO = r'C:\Users\amesa\OneDrive\Delaware\Power Traces\Data\Power_trace_jose\Sayid Tests\New Model'
CARPETA_DATASET_PROCESADO = os.path.join(RUTA_BASE_NUEVO_MODELO, 'processed_dataset_retrain')


def run_yosys(verilog_path, json_output_path):
    """
    Uses Yosys to do the heavy lifting of parsing the Verilog netlist.
    It takes the .v file and spits out a much friendlier .json file.
    """
    print("\n--- Running Yosys to convert Verilog to JSON ---")
    # The command tells Yosys to read, process, and write the JSON.
    yosys_script = f'read_verilog "{verilog_path}"; proc; write_json "{json_output_path}"'
    
    try:
        subprocess.run(['yosys', '-p', yosys_script], check=True, capture_output=True, text=True)
        print(f" -> Yosys finished successfully.")
        return True
    except Exception as e:
        print(f"CRITICAL ERROR: Yosys failed. Make sure it's installed and in the system PATH.")
        print(f"Error details: {e}")
        return False

def main():
    """
    Main function that runs the entire dataset creation pipeline.
    """
    print("--- Starting the creation of the new, combined dataset ---")
    
    # First, we need the netlist in a simple JSON format.
    # We check if it exists, and if not, we run Yosys to create it.
    if not os.path.exists(RUTA_JSON_NETLIST):
        run_yosys(os.path.join(CARPETA_FUENTE_ORIGINAL, 'aes_netlist.v'), RUTA_JSON_NETLIST)

    # Now, build the base graph structure and, most importantly, get the
    # "cell type dictionary" from the original AES circuit.
    graph_base, cell_map, net_map, cell_type_map = construir_grafo_desde_json(RUTA_JSON_NETLIST)
    if not graph_base:
        print("Failed to build the base graph. Aborting.")
        return

    # Create the output directories for our new experiment and save the dictionary.
    os.makedirs(CARPETA_DATASET_PROCESADO, exist_ok=True)
    cell_types_path = os.path.join(CARPETA_DATASET_PROCESADO, 'cell_types.json')
    with open(cell_types_path, 'w') as f:
        json.dump(cell_type_map, f)
    print(f"Saved our universal dictionary of {len(cell_type_map)} cell types.")

    # Create a master list of all the samples we want to process.
    master_sample_list = []
    
    # Add the original 800 samples
    for i in range(800):
        master_sample_list.append({
            'vcd_path': os.path.join(CARPETA_MUESTRAS_ORIGINALES, f'sample_{i:03d}', 'activity.vcd'),
            'power_path': os.path.join(CARPETA_MUESTRAS_ORIGINALES, f'sample_{i:03d}', 'power.data'),
            'id': i
        })
    
    # Add the 10 new samples from the "Third Batch" to make our dataset more diverse
    new_vcds = [f for f in os.listdir(CARPETA_NUEVOS_VCD_TRAIN) if f.endswith('.vcd')]
    for i, vcd_file in enumerate(new_vcds):
        base_name = os.path.splitext(vcd_file)[0]
        master_sample_list.append({
            'vcd_path': os.path.join(CARPETA_NUEVOS_VCD_TRAIN, vcd_file),
            'power_path': os.path.join(CARPETA_NUEVOS_VCD_TRAIN, f'{base_name}.data'),
            'id': 800 + i # Give them new IDs
        })

    print(f"\nFound a total of {len(master_sample_list)} samples to process.")

    # Before processing, we need to figure out how to normalize the power traces.
    # It's critical to calculate the min/max values ONLY from the training data
    # to avoid "leaking" information from the test set.
    indices = list(range(len(master_sample_list)))
    random.shuffle(indices)
    train_split_index = int(0.7 * len(indices))
    train_indices = indices[:train_split_index]

    print("\n--- Calculating normalization stats from the new training set ---")
    all_power_values = []
    for idx in tqdm(train_indices, desc="Scanning training traces"):
        power_path = master_sample_list[idx]['power_path']
        if os.path.exists(power_path):
            df = pd.read_csv(power_path, comment='#', delim_whitespace=True, names=['Time', 'Power'])
            all_power_values.append(df['Power'].values)
    
    all_power_values = np.concatenate(all_power_values)
    power_min, power_max = all_power_values.min(), all_power_values.max()
    
    norm_stats = {'min': float(power_min), 'max': float(power_max)}
    with open(os.path.join(CARPETA_DATASET_PROCESADO, 'norm_stats.json'), 'w') as f:
        json.dump(norm_stats, f)
    print("Normalization stats saved.")

    # Process every single sample. This is the heavy lifting.
    print("\n--- Processing and normalizing all samples ---")
    for sample_info in tqdm(master_sample_list, desc="Processing samples"):
        if not os.path.exists(sample_info['power_path']) or not os.path.exists(sample_info['vcd_path']):
            print(f"WARNING: Missing files for sample ID {sample_info['id']}. Skipping.")
            continue
            
        df_potencia = pd.read_csv(sample_info['power_path'], comment='#', delim_whitespace=True, names=['Time', 'Power'])
        vcd = VCDVCD(sample_info['vcd_path'])
        
        activity_matrix = crear_matriz_actividad(vcd, df_potencia, graph_base.num_nodes, len(cell_map), net_map)
        
        # Normalize the power trace to the [-1, 1] range
        power_values = df_potencia['Power'].values
        normalized_power = 2 * (power_values - power_min) / (power_max - power_min) - 1
        
        final_packet = graph_base.clone()
        final_packet.activity = activity_matrix
        final_packet.power_trace = torch.tensor(normalized_power, dtype=torch.float)
        
        torch.save(final_packet, os.path.join(CARPETA_DATASET_PROCESADO, f"sample_{sample_info['id']:03d}.pt"))

    # Finally, create the definitive splits.json based on the files that were actually created.
    print("\n--- Verifying processed files and creating final data splits ---")
    processed_files = os.listdir(CARPETA_DATASET_PROCESADO)
    valid_indices = []
    for f in processed_files:
        if f.startswith('sample_') and f.endswith('.pt'):
            try:
                idx = int(f.replace('sample_', '').replace('.pt', ''))
                valid_indices.append(idx)
            except ValueError:
                continue
    
    print(f"Found {len(valid_indices)} successfully processed samples.")

    random.shuffle(valid_indices)
    train_split = int(0.7 * len(valid_indices))
    val_split = int(0.85 * len(valid_indices))
    
    train_indices = valid_indices[:train_split]
    val_indices = valid_indices[train_split:val_split]
    test_indices = valid_indices[val_split:]
    
    splits = {'train': train_indices, 'validation': val_indices, 'test': test_indices}
    with open(os.path.join(CARPETA_DATASET_PROCESADO, 'splits.json'), 'w') as f:
        json.dump(splits, f)
        
    print(f"Data splits created successfully:")
    print(f" -> Training samples: {len(train_indices)}")
    print(f" -> Validation samples: {len(val_indices)}")
    print(f" -> Test samples: {len(test_indices)}")

if __name__ == "__main__":
    main()
