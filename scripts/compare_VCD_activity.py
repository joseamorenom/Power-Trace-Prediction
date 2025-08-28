# compare_vcd_activity.py
#
# A diagnostic script to investigate why a trained model might not generalize
# well to new VCD files.
#
# Hypothesis: The new VCDs, while for the same circuit, might have a very
# different activity distribution (i.e., they exercise the circuit differently).
#
# This script tests a simple proxy for this: the total number of switching events.
# It calculates and compares the average switch count between the original dataset
# and a new set of VCDs to quickly spot major differences.

import os
import numpy as np
from vcdvcd import VCDVCD
from tqdm import tqdm


# CONFIGURATION

# --- Paths ---
# Path to the base project folder
RUTA_BASE_PROYECTO = r'C:\Users\amesa\OneDrive\Delaware\Power Traces'

# Path to the original 800 preprocessed samples
CARPETA_DATASET_ORIGINAL = os.path.join(RUTA_BASE_PROYECTO, 'Data', 'Power_trace_jose', '3_dataset_samples')

# Path to the new VCDs we are testing against
CARPETA_NUEVOS_VCD = os.path.join(RUTA_BASE_PROYECTO, 'Data', 'Power_trace_jose', 'Sayid Tests', 'Second Batch')

# ANALYSIS LOGIC

def calculate_total_switches(vcd_path):
    """
    Open a VCD file and count the total number of signal changes across all signals.
    
    Args:
        vcd_path (str): The path to the .vcd file.

    Returns:
        int: The total number of switching events.
    """
    try:
        vcd = VCDVCD(vcd_path)
        # Sum the length of the (time, value) list for every signal in the file.
        total_switches = sum(len(vcd[signal].tv) for signal in vcd.signals)
        return total_switches
    except Exception as e:
        print(f"Error processing {vcd_path}: {e}")
        return 0

def main():
    """
    Main function to run the comparative analysis.
    """
    print("--- Analyzing activity of the ORIGINAL Dataset ---")
    original_switches = []
    # Analyze the first 50 samples from the original set for a quick comparison.
    for i in tqdm(range(50), desc="Processing original VCDs"):
        path = os.path.join(CARPETA_DATASET_ORIGINAL, f'sample_{i:03d}', 'activity.vcd')
        if os.path.exists(path):
            original_switches.append(calculate_total_switches(path))

    print("\n--- Analyzing activity of the NEW Dataset ---")
    new_switches = []
    new_vcds = [f for f in os.listdir(CARPETA_NUEVOS_VCD) if f.endswith('.vcd')]
    for filename in tqdm(new_vcds, desc="Processing new VCDs"):
        path = os.path.join(CARPETA_NUEVOS_VCD, filename)
        new_switches.append(calculate_total_switches(path))

    # --- Print a comparative summary ---
    print("\n" + "="*50)
    print("           VCD ACTIVITY COMPARISON SUMMARY")
    print("="*50)
    if original_switches:
        print(f"\nOriginal Dataset ({len(original_switches)} samples):")
        print(f" -> Avg Switches per Trace: {np.mean(original_switches):,.0f}")
        print(f" -> Min Switches:           {np.min(original_switches):,.0f}")
        print(f" -> Max Switches:           {np.max(original_switches):,.0f}")
    
    if new_switches:
        print(f"\nNew Dataset ({len(new_switches)} samples):")
        print(f" -> Avg Switches per Trace: {np.mean(new_switches):,.0f}")
        print(f" -> Min Switches:           {np.min(new_switches):,.0f}")
        print(f" -> Max Switches:           {np.max(new_switches):,.0f}")
    
    print("\n--- Conclusion ---")
    if new_switches and original_switches:
        mean_new = np.mean(new_switches)
        mean_orig = np.mean(original_switches)
        # Check if the new average is drastically different (e.g., >50% change)
        if mean_new > mean_orig * 1.5 or mean_new < mean_orig * 0.5:
            print("Hypothesis Confirmed: The activity level in the new VCDs is SIGNIFICANTLY DIFFERENT.")
            print("This explains the poor generalization. The model needs to be retrained with more diverse data.")
        else:
            print("The average activity level is similar.")
            print("The generalization issue is likely more subtle, related to *which specific parts*")
            print("of the circuit are being activated, confirming a data distribution shift.")

if __name__ == "__main__":
    main()
