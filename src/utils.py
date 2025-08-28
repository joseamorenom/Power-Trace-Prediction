# utils.py
#
# This script is the central toolbox for the Power Trace Prediction project.
# It contains all the shared, reusable components like the custom Dataset class,
# the model architecture definitions (GNN, ActivityEncoder, PowerPredictor),
# and the core data preprocessing functions.

import os
import json
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from vcdvcd import VCDVCD
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv, global_mean_pool
import numpy as np


#1. PYTORCH GEOMETRIC DATASET CLASS


class PowerTraceDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset for our project.
    It's responsible for loading the preprocessed .pt files from disk.
    """
    def __init__(self, root_dir, indices):
        """
        Initializes the dataset.
        Args:
            root_dir (str): The directory where the processed .pt files are.
            indices (list): A list of integer IDs for the samples to include.
        """
        super(PowerTraceDataset, self).__init__(root_dir)
        self.indices = indices

    def __len__(self):
        """Returns the total number of samples in this dataset split."""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Gets a single sample. This is what the DataLoader calls internally.
        """
        # The DataLoader gives us a relative index (0 to len-1).
        # We use it to get the absolute sample ID from our list.
        real_idx = self.indices[idx]
        return self.get(real_idx)

    def get(self, idx):
        """
        Loads a single preprocessed data sample from its .pt file.
        Args:
            idx (int): The absolute sample ID (e.g., from 0 to 809).
        """
        data_path = os.path.join(self.root, f'sample_{int(idx):03d}.pt')
        try:
            data = torch.load(data_path)
            return data
        except FileNotFoundError:
            # If a file is missing, we return None. Our DataLoader's collate_fn
            # will handle this gracefully to prevent a crash.
            print(f"Warning: Data file not found at {data_path}. Skipping sample.")
            return None


#2. MODEL ARCHITECTURE DEFINITIONS


class GNNEncoder(nn.Module):
    """
    The GNN Encoder, our "Structural Analyst".
    It processes the static circuit graph using Graph Attention Networks (GAT).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        # GAT layers learn to weigh the importance of neighboring nodes.
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1, concat=False)

    def forward(self, data):
        """Defines the forward pass for the GNN."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        
        # We aggregate all node features into a single vector representing the whole graph.
        graph_embedding = global_mean_pool(x, batch)
        return graph_embedding

class ActivityEncoder(nn.Module):
    """
    The Activity Encoder, our "Behavioral Analyst".
    Uses a hybrid 1D-CNN and Bidirectional LSTM to process the activity matrix.
    """
    def __init__(self, num_nodes, cnn_out_channels, lstm_hidden, output_dim):
        super(ActivityEncoder, self).__init__()
        # The 1D CNN is a great local pattern detector. It scans the activity
        # across all nodes at each time step.
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=128, stride=64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # We need to calculate the flattened output size of the CNN to correctly
        # initialize the LSTM layer.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, num_nodes)
            dummy_output = self.cnn(dummy_input)
            self.cnn_output_flat_size = dummy_output.shape[1] * dummy_output.shape[2]
        
        # The Bidirectional LSTM processes the sequence of patterns found by the CNN.
        self.lstm = nn.LSTM(self.cnn_output_flat_size, lstm_hidden, batch_first=True, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, output_dim) # *2 because it's bidirectional

    def forward(self, activity_matrix):
        # Input shape: [batch, seq_len, num_nodes]
        batch_size, seq_len, num_nodes = activity_matrix.shape
        
        # Reshape for the CNN: we treat each of the 400 time steps as a separate item in a big batch.
        activity_reshaped = activity_matrix.view(batch_size * seq_len, 1, num_nodes)
        
        # Get the spatial features from the CNN
        cnn_features = self.cnn(activity_reshaped)
        
        # Reshape back into a sequence for the LSTM
        cnn_features_flat = cnn_features.view(batch_size, seq_len, -1)
        
        # Process the sequence of features
        _, (hidden_state, _) = self.lstm(cnn_features_flat)
        
        # Concatenate the final forward and backward hidden states for a rich summary
        forward_hidden = hidden_state[-2,:,:]
        backward_hidden = hidden_state[-1,:,:]
        hidden_concat = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        return self.fc(hidden_concat)

class PowerPredictor(nn.Module):
    """
    The main multimodal model, our "Project Manager".
    It combines the GNN and Activity encoders to predict the final power trace.
    """
    def __init__(self, gnn_in, gnn_hidden, num_nodes, cnn_out, lstm_hidden, common_embedding_dim, output_len):
        super(PowerPredictor, self).__init__()
        self.gnn_encoder = GNNEncoder(gnn_in, gnn_hidden, common_embedding_dim)
        self.activity_encoder = ActivityEncoder(num_nodes, cnn_out, lstm_hidden, common_embedding_dim)
        
        # The final MLP that fuses the two expert opinions and makes the prediction.
        self.prediction_head = nn.Sequential(
            nn.Linear(common_embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout helps prevent overfitting
            nn.Linear(512, output_len)
        )

    def forward(self, data):
        # 1. Get the structural summary
        graph_embedding = self.gnn_encoder(data)
        
        # 2. Get the behavioral summary
        num_graphs = data.num_graphs
        activity_batched = data.activity.reshape(num_graphs, -1, data.activity.shape[-1])
        activity_embedding = self.activity_encoder(activity_batched)
        
        # 3. Combine summaries and predict
        combined_embedding = torch.cat([graph_embedding, activity_embedding], dim=1)
        predicted_trace = self.prediction_head(combined_embedding)
        return predicted_trace

# 3. DATA PREPROCESSING HELPER FUNCTIONS

def construir_grafo_desde_json(ruta_json_netlist, cell_type_map=None):
    """
    Builds a PyG graph from a Yosys JSON file.
    If a `cell_type_map` is provided, it uses that "universal dictionary"
    to ensure feature consistency. Otherwise, it creates a new one.
    """
    with open(ruta_json_netlist, 'r') as f:
        data = json.load(f)
    module_data = next(iter(data['modules'].values()))
    cells = module_data['cells']
    
    if cell_type_map is None:
        print("No cell type map provided. Creating a new one from this netlist...")
        cell_types = {cell_type: i for i, cell_type in enumerate(set(cell['type'] for cell in cells.values()))}
        return_new_map = True
    else:
        cell_types = cell_type_map
        return_new_map = False

    # Map cell and net names to unique integer IDs
    cell_to_id = {name: i for i, name in enumerate(cells)}
    
    # Robustly find the total number of nets (including unnamed ones)
    max_net_index = 0
    for cell_info in cells.values():
        for net_indices in cell_info['connections'].values():
            if net_indices:
                max_net_index = max(max_net_index, max(net_indices))
    num_nets = max_net_index + 1
    num_cells = len(cell_to_id)
    
    netnames_json = list(module_data['netnames'].keys())
    net_id_to_name = {i: name.split('.')[-1] for i, name in enumerate(netnames_json)}
    
    # Build the edge list for our bipartite graph (cell <-> net)
    edge_list = []
    for cell_name, cell_info in cells.items():
        cell_id = cell_to_id[cell_name]
        for port_name, net_indices in cell_info['connections'].items():
            for net_index in net_indices:
                net_node_id = net_index + num_cells
                edge_list.append([cell_id, net_node_id])
                edge_list.append([net_node_id, cell_id])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create one-hot encoded node features using the cell_types dictionary
    cell_features = torch.zeros(num_cells, len(cell_types))
    for cell_name, cell_info in cells.items():
        cell_id = cell_to_id[cell_name]
        cell_type = cell_info['type']
        if cell_type in cell_types:
            type_id = cell_types[cell_type]
            cell_features[cell_id, type_id] = 1
            
    net_features = torch.zeros(num_nets, len(cell_types))
    x = torch.cat([cell_features, net_features], dim=0)
    
    graph_data = Data(x=x, edge_index=edge_index)
    
    if return_new_map:
        return graph_data, cell_to_id, net_id_to_name, cell_types
    else:
        return graph_data, cell_to_id, net_id_to_name

def crear_matriz_actividad(vcd, df_potencia, num_total_nodos, num_cells, net_id_to_name):
    """
    Builds the activity matrix from a VCD file, aligned with the power trace.
    """
    num_timesteps = len(df_potencia)
    activity_matrix = np.zeros((num_timesteps, num_total_nodos), dtype=np.uint8)
    time_bins = df_potencia['Time'].values
    
    # Create an efficient lookup map of VCD signals for fast searching
    vcd_signal_map = {s.split('.')[-1]: s for s in vcd.signals}
    
    # Map our graph's nets to the VCD signals
    vcd_signal_to_net_id = {}
    for net_id, net_name in net_id_to_name.items():
        if net_name in vcd_signal_map:
            vcd_signal = vcd_signal_map[net_name]
            # Add the offset to distinguish net nodes from cell nodes
            vcd_signal_to_net_id[vcd_signal] = net_id + num_cells
            
    # Populate the activity matrix
    for vcd_signal, net_node_id in vcd_signal_to_net_id.items():
        # Safety check: ignore activity on nets that don't fit our model's expected size
        if net_node_id >= num_total_nodos:
            continue
        
        activity_signal = vcd[vcd_signal].tv
        switch_times = [t for t, v in activity_signal]
        
        # Find which time bin each switch belongs to
        time_indices = np.digitize(switch_times, bins=time_bins)
        
        for idx in time_indices:
            if idx < num_timesteps:
                activity_matrix[idx, net_node_id] = 1
                
    return torch.from_numpy(activity_matrix).float()
