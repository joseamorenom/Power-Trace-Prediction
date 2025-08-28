# organize_raw_data.py
#
# This is a one-time utility script to organize the initial raw dataset
# into a clean, logical, sample-centric structure.
#
# It takes the scattered source files, VCDs, and power traces and sorts
# them into the following structure:
#   - 1_source_files/ (netlist, .lib)
#   - 2_context_data/ (keys, plaintexts)
#   - 3_dataset_samples/ (sample_000/, sample_001/, etc.)
#
# This makes the dataset much easier for our main preprocessing script to handle.

import os
import shutil
from tqdm import tqdm


# --- Base Path ---
BASE_PATH = r'C:\Users\amesa\OneDrive\Delaware\Power Traces\Data\Power_trace_jose'

# --- New Directory Names ---
SOURCE_DIR = os.path.join(BASE_PATH, '1_source_files')
CONTEXT_DIR = os.path.join(BASE_PATH, '2_context_data')
SAMPLES_DIR = os.path.join(BASE_PATH, '3_dataset_samples')

# --- Original File and Folder Paths ---
# Static files in the base directory
ORIGINAL_NETLIST = os.path.join(BASE_PATH, 'aes_syn_LLL.v')
ORIGINAL_LIBRARY = os.path.join(BASE_PATH, 'NangateOpenCellLibrary_slow_ccs.lib')
ORIGINAL_PLAINTEXTS = os.path.join(BASE_PATH, 'plaintexts.txt')
ORIGINAL_KEYS = os.path.join(BASE_PATH, 'keys.txt')
ORIGINAL_CIPHERTEXTS = os.path.join(BASE_PATH, 'ciphertexts.txt')

# Folders containing the 800 sample files
ORIGINAL_VCD_DIR = os.path.join(BASE_PATH, r'sim_results_AES_nyu_LLL_800\sim_results')
ORIGINAL_POWER_DIR = os.path.join(BASE_PATH, r'power_traces_AES_nyu_full_library_LLL_800\power_traces')


def organize_dataset():
    """
    Main function to run the automated file organization process.
    """
    print("--- Starting automatic dataset organization ---")

    # Step 1: Create the new, clean directory structure.
    print("\n[Step 1/4] Creating new directory structure...")
    os.makedirs(SOURCE_DIR, exist_ok=True)
    os.makedirs(CONTEXT_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    print(" -> Directories '1_source_files', '2_context_data', '3_dataset_samples' are ready.")

    # Step 2: Copy the source files (Netlist and Library).
    print("\n[Step 2/4] Copying source files...")
    try:
        shutil.copy(ORIGINAL_NETLIST, os.path.join(SOURCE_DIR, 'aes_netlist.v'))
        print(" -> Copied: aes_netlist.v")
        shutil.copy(ORIGINAL_LIBRARY, os.path.join(SOURCE_DIR, 'nangate_library.lib'))
        print(" -> Copied: nangate_library.lib")
    except FileNotFoundError as e:
        print(f"   -> WARNING: A source file was not found: {e}")

    # Step 3: Copy the context files (Keys, Plaintexts, etc.).
    print("\n[Step 3/4] Copying context files...")
    try:
        shutil.copy(ORIGINAL_PLAINTEXTS, os.path.join(CONTEXT_DIR, 'plaintexts.txt'))
        print(" -> Copied: plaintexts.txt")
        shutil.copy(ORIGINAL_KEYS, os.path.join(CONTEXT_DIR, 'keys.txt'))
        print(" -> Copied: keys.txt")
        shutil.copy(ORIGINAL_CIPHERTEXTS, os.path.join(CONTEXT_DIR, 'ciphertexts.txt'))
        print(" -> Copied: ciphertexts.txt")
    except FileNotFoundError as e:
        print(f"   -> WARNING: A context file was not found: {e}")

    # Step 4: Create individual sample folders and copy the VCD and power files.
    print("\n[Step 4/4] Processing and organizing the 800 samples...")
    processed_count = 0
    for i in tqdm(range(800), desc="Organizing samples"):
        # Format the sample number with leading zeros (e.g., 000, 001, 056, 123)
        sample_name = f"sample_{i:03d}"
        destination_sample_dir = os.path.join(SAMPLES_DIR, sample_name)
        os.makedirs(destination_sample_dir, exist_ok=True)

        # Define source and destination paths for this sample's files
        vcd_source = os.path.join(ORIGINAL_VCD_DIR, f'trace_{i}.vcd')
        power_source = os.path.join(ORIGINAL_POWER_DIR, f'trace_{i}.data')

        vcd_dest = os.path.join(destination_sample_dir, 'activity.vcd')
        power_dest = os.path.join(destination_sample_dir, 'power.data')

        # Copy the files, but only if both exist
        try:
            if os.path.exists(vcd_source) and os.path.exists(power_source):
                shutil.copy(vcd_source, vcd_dest)
                shutil.copy(power_source, power_dest)
                processed_count += 1
            else:
                print(f"   -> WARNING: Missing files for sample {i}. VCD exists: {os.path.exists(vcd_source)}, Power exists: {os.path.exists(power_source)}")
        except Exception as e:
            print(f"   -> ERROR processing sample {i}: {e}")

    print(f"\n--- Reorganization Complete! ---")
    print(f"Successfully processed and organized {processed_count} samples.")
    print(f"The clean dataset is now ready in: {SAMPLES_DIR}")

if __name__ == "__main__":
    organize_dataset()
