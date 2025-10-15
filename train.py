# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# --- MODIFIED: Updated import paths ---
from models.model import Model
from utils.dataset import PKUMMDDataset # Look inside the 'utils' folder

def read_split_file(filepath):
    """Helper function to read the split files and return a list of filenames."""
    with open(filepath, 'r') as f:
        filenames = [line.strip().replace('.avi', '.txt') for line in f]
    return filenames

def train_and_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files_in_dir = set(f for f in os.listdir(args.skeleton_dir) if f.endswith('.txt'))
    train_files_from_split = read_split_file(args.split_file)
    train_files = list(set(train_files_from_split) & all_files_in_dir)
    val_files = list(all_files_in_dir - set(train_files))

    print(f"Found {len(all_files_in_dir)} total files.")
    print(f"Training on {len(train_files)} files, Validating on {len(val_files)} files.")

    train_dataset = PKUMMDDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        file_list=train_files,
        window_size=args.window_size
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_dataset = PKUMMDDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        file_list=val_files,
        window_size=args.window_size
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # The graph import path is handled inside the model's __init__
    model = Model(num_class=52, num_point=25, num_person=2, graph='utils.graph.Graph', graph_args=dict()).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        model.train()
        # (Your training and validation loops go here, unchanged)
        # ...
        # (For brevity, loop is omitted, but it stays the same as before)
        # ...
        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.save_path)
            print(f"New best model saved to {args.save_path} with accuracy: {best_val_accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CTR-GCN using official PKU-MMD splits")
    
    # --- MODIFIED: Updated default paths ---
    parser.add_argument('--skeleton_dir', type=str, default='data/PKU_Skeleton_Renew')
    parser.add_argument('--label_dir', type=str, default='data/Vectorized_Labels')
    parser.add_argument('--split_file', type=str, required=True, help="Path to the split file (e.g., data/splits/cross-subject.txt)")
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--window_size', type=int, default=300)
    parser.add_argument('--save_path', type=str, default='best_model.pth', help="Path to save the best model weights")
    
    args = parser.parse_args()
    # train_and_evaluate(args)
    print("✅ train.py is updated. Remember to paste your training/validation loop logic back into the train_and_evaluate function.")