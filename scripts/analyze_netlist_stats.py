# analyze_netlist_stats.py
#
# A utility to get a statistical report of a Verilog netlist using Yosys.
# It calculates two key metrics:
#   1. Hierarchical Count: The true total of primitive cells across all sub-modules.
#   2. Flattened Count: The final cell count of the top-level module, which is
#      the structure our AI model is trained on.

import os
import subprocess
import re

# --- CONFIGURATION ---
NETLIST_PATH = r'C:\Users\amesa\OneDrive\Delaware\Power Traces\Data\Power_trace_jose\1_source_files\aes_netlist.v'

def analyze_netlist(verilog_path):
    """
    Run Yosys to get a statistical report and print a summary.
    """
    print(f"--- Analyzing: {os.path.basename(verilog_path)} ---")

    # Yosys command to read the Verilog file and get statistics.
    yosys_script = f'read_verilog "{verilog_path}"; stat'

    try:
        # Execute Yosys and capture its text output.
        result = subprocess.run(
            ['yosys', '-p', yosys_script], 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8'
        )
        
        # Parse the output to generate our summary.
        parse_and_print_summary(result.stdout)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

def parse_and_print_summary(report_text):
    """
    Parse the raw Yosys output to extract and print a clean summary.
    """
    total_hierarchical_cells = 0
    final_flat_cells = 0
    
    # Find all module sections in the report.
    module_sections = re.findall(r"===\s*(\w+)\s*===(.*?)(?===|$)", report_text, re.DOTALL)

    print("\n--- Hierarchical Breakdown ---")
    # First, sum up all primitive cells from all modules.
    for name, section in module_sections:
        num_cells_match = re.search(r"Number of cells:\s+(\d+)", section)
        if num_cells_match:
            num_cells_total = int(num_cells_match.group(1))
            
            # Find and subtract instances of sub-modules to get the primitive count.
            submodule_instance_count = 0
            instance_matches = re.findall(r"^\s+(sub\w+)\s+(\d+)", section, re.MULTILINE)
            for sub_name, count in instance_matches:
                submodule_instance_count += int(count)
            
            primitive_cells = num_cells_total - submodule_instance_count
            total_hierarchical_cells += primitive_cells
            
            print(f"- Module '{name}': {primitive_cells} primitive cells (+ {submodule_instance_count} sub-module instances)")

            # The 'aes' module's total cell count is the flattened count.
            if name == 'aes':
                final_flat_cells = num_cells_total

    # Finally, print a clear conclusion.
    print("\n" + "="*40)
    print("           FINAL SUMMARY")
    print("="*40)
    print(f"Hierarchical Cell Count: {total_hierarchical_cells}")
    print(f"  (This is the true sum of all primitive gates in the design.)\n")
    print(f"Flattened Cell Count:    {final_flat_cells}")
    print(f"  (This is the final structure that our AI model is trained on.)")
    print("="*40)


if __name__ == "__main__":
    if os.path.exists(NETLIST_PATH):
        analyze_netlist(NETLIST_PATH)
    else:
        print(f"ERROR: Netlist file not found at: '{NETLIST_PATH}'")
