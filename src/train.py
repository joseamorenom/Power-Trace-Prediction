# train.py
#
# This script takes our preprocessed dataset, defines the AI model architecture, and then trains it.
#
# The process is:
#   1. Load the processed data and the train/val/test splits.
#   2. Define the model architecture (the GNN and LSTM working together).
#   3. Loop for a set number of epochs, training the model on the training data.
#   4. After each epoch, check the model's performance on the validation data.
#   5. Save the model if it's the best one we've seen so far.
#   6. After all epochs, run a final test on the unseen test data for our final score.
#

import os
import json
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np
from tslearn.metrics import dtw

# Import our custom tools from the toolbox
from utils import PowerTraceDataset, PowerPredictor


# --- Paths ---
# Probably change these
RUTA_BASE_NUEVO_MODELO = r'C:\Users\amesa\OneDrive\Delaware\Power Traces\Data\Power_trace_jose\Sayid Tests\New Model'
CARPETA_DATASET_PROCESADO = os.path.join(RUTA_BASE_NUEVO_MODELO, 'processed_dataset_retrain')
RUTA_SPLITS_JSON = os.path.join(CARPETA_DATASET_PROCESADO, 'splits.json')
RUTA_GUARDADO_MODELO = os.path.join(RUTA_BASE_NUEVO_MODELO, 'best_model_retrain.pth')

# --- Hyperparameters ---
NUM_EPOCHS = 15       # How many times we'll show the entire dataset to the model.
BATCH_SIZE = 16       # How many samples the model looks at in one go.
LEARNING_RATE = 0.001 # How big of a step the model takes when it learns.
HIDDEN_DIM_GNN = 64   # Internal "brainpower" of the GNN.
HIDDEN_DIM_LSTM = 128 # Internal "brainpower" of the LSTM.

# =============================================================================
# ==  HELPER FUNCTIONS
# =============================================================================

def collate_fn(batch):
    """
    A helper for the DataLoader. If our dataset script failed to create a
    .pt file for some reason, this function catches it and filters it out
    so the training doesn't crash.
    """
    # Filter out any samples that might have failed during preprocessing (returned as None)
    batch = [data for data in batch if data is not None]
    if not batch:
        return None
    # Use the default PyG function to combine the valid samples into a single batch object
    return Batch.from_data_list(batch)

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    This is the "study session" for our model. It goes through one full
    pass of the training data.
    """
    model.train() # Put the model in "learning" mode
    total_loss = 0
    num_samples_processed = 0
    
    for data in tqdm(loader, desc="Training"):
        if data is None: # Skip any batches that were empty after filtering
            continue
            
        data = data.to(device) # Move data to the GPU
        optimizer.zero_grad()  # Reset gradients from the last step
        
        # 1. Make a prediction
        prediction = model(data)
        
        # 2. Check the answer
        target = data.power_trace.reshape(data.num_graphs, -1)
        loss = criterion(prediction, target)
        
        # 3. Learn from the mistake
        loss.backward()   # Calculate how much each weight contributed to the error
        optimizer.step()  # Update the weights to do better next time
        
        total_loss += loss.item() * data.num_graphs
        num_samples_processed += data.num_graphs
        
    return total_loss / num_samples_processed if num_samples_processed > 0 else 0

def evaluate(model, loader, device, desc="Evaluating"):
    """
    It goes through the validation or test data to see how well it's doing, without letting it learn.
    """
    model.eval() # Put the model in "testing" mode
    all_predictions, all_actuals = [], []
    
    with torch.no_grad(): # Turn off learning for this part
        for data in tqdm(loader, desc=desc):
            if data is None:
                continue
            data = data.to(device)
            prediction = model(data)
            target = data.power_trace.reshape(data.num_graphs, -1)
            all_predictions.append(prediction.cpu().numpy())
            all_actuals.append(target.cpu().numpy())
    
    if not all_predictions:
        return float('nan'), float('nan')

    # Combine all the batch results into one big array
    all_predictions = np.vstack(all_predictions)
    all_actuals = np.vstack(all_actuals)
    
    # Calculate our performance metrics
    mse = np.mean((all_predictions - all_actuals)**2)
    try:
        # DTW is great for checking if the *shape* of the wave is right.
        dtw_scores = [dtw(pred, act) for pred, act in zip(all_predictions, all_actuals)]
        dtw_score = np.mean(dtw_scores)
    except Exception:
        dtw_score = float('nan')
    
    return mse, dtw_score


# MAIN EXECUTION BLOCK
def main():
    
    # Load the train/validation/test splits we created earlier
    with open(RUTA_SPLITS_JSON, 'r') as f:
        splits = json.load(f)
    
    train_indices = splits['train']
    val_indices = splits['validation']
    test_indices = splits['test']
    
    # Create the Dataset objects
    train_dataset = PowerTraceDataset(CARPETA_DATASET_PROCESADO, train_indices)
    val_dataset = PowerTraceDataset(CARPETA_DATASET_PROCESADO, val_indices)
    test_dataset = PowerTraceDataset(CARPETA_DATASET_PROCESADO, test_indices)
    
    # Create the DataLoaders, which will feed data to the model in batches.
    # We use our custom collate_fn to handle any missing files gracefully.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Set up our device (use GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize our model architecture
    # We load one sample just to get the dimensions right for the model input/output layers.
    first_valid_sample = next(item for item in train_dataset if item is not None)
    if first_valid_sample is None:
        print("Error: No valid samples found in the training dataset.")
        return
        
    model = PowerPredictor(
        gnn_in=first_valid_sample.num_node_features,
        gnn_hidden=HIDDEN_DIM_GNN,
        num_nodes=first_valid_sample.activity.shape[1],
        cnn_out=32,
        lstm_hidden=HIDDEN_DIM_LSTM,
        common_embedding_dim=128,
        output_len=len(first_valid_sample.power_trace)
    ).to(device)
    
    # Set up the optimizer (Adam is a great general-purpose choice) and the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_val_mse = float('inf') # Keep track of the best score to save the best model

    print("\n--- Starting training with Combined Dataset ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Run one study session
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # quiz on the val set
        val_mse, val_dtw = evaluate(model, val_loader, device, desc="Validating")
        
        print(f"Epoch {epoch:02d} | Train Loss (MSE): {train_loss:.6f} | Val MSE: {val_mse:.6f} | Val DTW: {val_dtw:.4f}")

        #Save the best model seen
        if not np.isnan(val_mse) and val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), RUTA_GUARDADO_MODELO)
            print(f" -> New best model saved! (Validation MSE: {best_val_mse:.6f})")

    print("\n--- training Finished ---")
    
    # Now, for the final exam on the test set.
    print("\n--- Starting Final Evaluation on the Test Set ---")
    # Load the weights of the best model we saved
    model.load_state_dict(torch.load(RUTA_GUARDADO_MODELO))
    
    test_mse, test_dtw = evaluate(model, test_loader, device, desc="Testing")
    
    print("\n--- Final Results of the trained Model ---")
    print(f" -> Test Set MSE: {test_mse:.6f}")
    print(f" -> Test Set DTW: {test_dtw:.4f}")

if __name__ == "__main__":
    main()
