# train.py (FINAL VERSION - 2)
import os
import sys
# Failsafe for Colab's import system
sys.path.append('.') 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# --- Import modules from your project structure ---
from models.gcn import Model 
from utils.dataset import PKUMMDDataset # This import is still correct

def read_split_file(filepath):
    """
    Parses the comma-separated format of the provided split file.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    try:
        training_section = content.split('Training videos:')[1].split('Validataion videos:')[0]
        filenames_base = training_section.split(',')
        filenames = [name.strip() + '.txt' for name in filenames_base if name.strip()]
        return filenames
    except IndexError:
        print(f"❌ Error: Could not parse the file at {filepath}.")
        return []

def train_and_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading official split from: {args.split_file}")
    all_files_in_dir = set(f for f in os.listdir(args.skeleton_dir) if f.endswith('.txt'))
    train_files = read_split_file(args.split_file)
    val_files = list(all_files_in_dir - set(train_files))
    
    if not train_files:
        print("Error: Training file list is empty after parsing.")
        return

    print(f"Found {len(all_files_in_dir)} total files.")
    print(f"Training on {len(train_files)} files, Validating on {len(val_files)} files.")

    train_dataset = PKUMMDDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        file_list=train_files,
        window_size=args.window_size
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    val_dataset = PKUMMDDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        file_list=val_files,
        window_size=args.window_size
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # --- THIS IS THE KEY CHANGE ---
    # The path now points to the top-level 'graph' package, then the 'graph.py' module, then the 'Graph' class.
    model = Model(num_class=52, num_point=25, num_person=2, graph='graph.graph.Graph', graph_args=dict()).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_accuracy = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        # --- Training & Validation Loops (unchanged) ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        for data, labels in tqdm(train_loader, desc="Training"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (correct_preds / total_preds) * 100
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

        model.eval()
        running_val_loss = 0.0
        correct_val_preds = 0
        total_val_preds = 0
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validating"):
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_preds += labels.size(0)
                correct_val_preds += (predicted == labels).sum().item()
        val_loss = running_val_loss / len(val_dataset)
        val_accuracy = (correct_val_preds / total_val_preds) * 100
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.save_path)
            print(f"🎉 New best model saved to {args.save_path} with accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CTR-GCN using official PKU-MMD splits")
    
    parser.add_argument('--skeleton_dir', type=str, default='data/PKU_Skeleton_Renew')
    parser.add_argument('--label_dir', type=str, default='data/Vectorized_Labels')
    parser.add_argument('--split_file', type=str, required=True)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--window_size', type=int, default=300)
    parser.add_argument('--save_path', type=str, default='best_model.pth')
    
    args = parser.parse_args()
    train_and_evaluate(args)