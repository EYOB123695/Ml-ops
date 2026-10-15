"""
Untracked Baseline Training Script.

This is a deliberately bare-bones training script that demonstrates the
NAIVE approach to model training WITHOUT any MLOps tooling.

=== LIMITATIONS (documented for Assignment 4 requirement) ===

1. EPHEMERAL METRICS: All training output is printed to the terminal.
   Once the terminal is closed, the entire training history is lost forever.
   There is no persistent record of what happened during training.

2. LOST HYPERPARAMETER PROVENANCE: Parameters are hardcoded directly in
   this file. If the model achieves 84% accuracy, there is no automated
   record of which exact seed, learning rate, or dropout produced it.
   A researcher must manually track these in a spreadsheet or notebook.

3. SILENT OVERWRITING: Running this script again overwrites model.pt
   without warning. The previous best model is permanently destroyed.
   There is no versioning, no rollback capability.

4. NO CODE/DATA LINEAGE: There is no link between the saved model file
   and the Git commit or dataset version that produced it. If the dataset
   changes subtly, we cannot trace which model was trained on which data.

5. NO SCHEMA PACKAGING: The saved model.pt contains only raw weight
   tensors. It does not record that the model expects 1433 input features,
   produces 7 output classes, or what those classes mean. Loading it
   requires manually knowing the exact architecture.

Compare this with train_mlflow.py which solves all of these problems.
"""

import os
import sys
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.cora_loader import load_cora
from src.models.gnn_models import GraphSAGE, count_parameters


def main():
    # ===== HARDCODED PARAMETERS (Limitation #2) =====
    SEED = 42
    HIDDEN_DIM = 64
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.5
    EPOCHS = 200
    NUM_LAYERS = 2
    MODEL_SAVE_PATH = "model.pt"  # Overwrites every run (Limitation #3)

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Load dataset
    print("Loading Cora dataset...")
    data = load_cora()
    print(f"  Nodes: {data['num_nodes']}, Edges: {data['num_edges']}")
    print(f"  Features: {data['num_features']}, Classes: {data['num_classes']}")

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(
        in_dim=data["num_features"],
        hidden_dim=HIDDEN_DIM,
        out_dim=data["num_classes"],
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    print(f"  Parameters: {count_parameters(model)}")

    x = data["x"].to(device)
    edge_index = data["edge_index"].to(device)
    y = data["y"].to(device)
    train_mask = data["train_mask"].to(device)
    val_mask = data["val_mask"].to(device)
    test_mask = data["test_mask"].to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # ===== TRAINING LOOP =====
    # Metrics are ONLY printed to terminal (Limitation #1)
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Training step
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation step
        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            pred = logits.argmax(dim=1)

            train_acc = float((pred[train_mask] == y[train_mask]).float().mean())
            val_acc = float((pred[val_mask] == y[val_mask]).float().mean())

        # Print to terminal only — ephemeral output (Limitation #1)
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

        # Track best model (in memory only)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    # ===== FINAL EVALUATION =====
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        pred = logits.argmax(dim=1)
        test_acc = float((pred[test_mask] == y[test_mask]).float().mean())

    print(f"\n{'='*50}")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Test Acc:     {test_acc:.4f}")
    print(f"{'='*50}")

    # ===== SAVE MODEL =====
    # Saves ONLY raw weights — no schema, no metadata (Limitation #5)
    # Silently overwrites previous model.pt (Limitation #3)
    # No link to Git commit or dataset hash (Limitation #4)
    torch.save(best_state, MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("WARNING: No metadata, no versioning, no experiment tracking.")


if __name__ == "__main__":
    main()
