# train_classifier.py
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

# --- Import your project's modules ---
from models.gcn import Model 
from utils.dataset import PKUMMDDataset

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
    """
    Main function to train the MULTI-CLASS ACTION CLASSIFIER (Actions Only).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("--- Training Action Classifier (Multi-Class, Actions Only) ---")

    print(f"Loading official split from: {args.split_file}")
    all_files_in_dir = set(f for f in os.listdir(args.skeleton_dir) if f.endswith('.txt'))
    train_files = read_split_file(args.split_file)
    val_files = list(all_files_in_dir - set(train_files))
    
    if not train_files:
        print("Error: Training file list is empty.")
        return

    print(f"Found {len(all_files_in_dir)} total files.")
    print(f"Training on {len(train_files)} files, Validating on {len(val_files)} files.")

    # 1. Create Datasets (loading ALL samples initially)
    # --- CHANGE: Added stride=args.stride ---
    train_dataset = PKUMMDDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        file_list=train_files,
        window_size=args.window_size,
        stride=args.stride  # <--- PASSING THE STRIDE
    )
    # --- CHANGE: Added stride=args.stride ---
    val_dataset = PKUMMDDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        file_list=val_files,
        window_size=args.window_size,
        stride=args.stride  # <--- PASSING THE STRIDE
    )

    # 2. KEY CHANGE: Filter datasets to ONLY include samples where label > 0
    print(f"Original training samples: {len(train_dataset.samples)}")
    train_dataset.samples = [s for s in train_dataset.samples if s[2] > 0]
    print(f"Filtered training samples (actions only): {len(train_dataset.samples)}")
    
    print(f"Original validation samples: {len(val_dataset.samples)}")
    val_dataset.samples = [s for s in val_dataset.samples if s[2] > 0]
    print(f"Filtered validation samples (actions only): {len(val_dataset.samples)}")

    # 3. Calculate Class Weights for the 51 Action Classes
    print("Calculating class weights for action classification...")
    all_labels_actions = [(s[2] - 1) for s in train_dataset.samples]
    
    if not all_labels_actions:
        print("❌ Error: No action samples (label > 0) found in the training set. Check your data or stride.")
        return

    class_counts = torch.bincount(torch.tensor(all_labels_actions), minlength=51)
    class_weights = 1. / class_counts.float()
    class_weights[class_weights == float('inf')] = 0
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)
    print(f"Class Weights (for actions 1-51): {class_weights}")

    # --- CHANGE: Added num_workers=args.num_workers ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, # <--- USING NUM_WORKERS ARG
        pin_memory=True
    )
    # --- CHANGE: Added num_workers=args.num_workers ---
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, # <--- USING NUM_WORKERS ARG
        pin_memory=True
    )

    # 4. Instantiate Model for MULTI-CLASS classification (num_class=51)
    model = Model(num_class=51, num_point=25, num_person=2, graph='graph.graph.Graph', graph_args=dict()).to(device)
    
    # 5. Use the weighted loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_accuracy = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        for data, labels in tqdm(train_loader, desc="Training Classifier"):
            # 6. KEY CHANGE: Remap labels from [1...51] to [0...50]
            labels_remapped = labels - 1
            data, labels = data.to(device), labels_remapped.to(device)
            
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
            for data, labels in tqdm(val_loader, desc="Validating Classifier"):
                labels_remapped = labels - 1
                data, labels = data.to(device), labels_remapped.to(device)
                
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
            print(f"🎉 New best CLASSIFIER model saved to {args.save_path} with accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CTR-GCN Action Classifier (Actions Only)")
    parser.add_argument('--skeleton_dir', type=str, default='data/PKU_Skeleton_Renew')
    parser.add_argument('--label_dir', type=str, default='data/Vectorized_Labels')
    
    # --- CHANGE: Fixed the split_file argument ---
    parser.add_argument('--split_file', type=str, default='data/splits/cross-view.txt', help="Path to the official split file")
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--window_size', type=int, default=32)
    
    # --- NEW ARGUMENTS FOR OPTIMIZATION ---
    parser.add_argument('--stride', type=int, default=16, help="Stride for sliding window. 8 or 16 is good.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of CPU workers for data loading")
    
    parser.add_argument('--save_path', type=str, default='classifier_model.pth', help="Path to save the best classifier model")
    
    args = parser.parse_args()
    train_and_evaluate(args)
