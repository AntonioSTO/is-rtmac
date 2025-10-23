# evaluate_detector.py
import os
import sys
# Failsafe for Colab's import system
sys.path.append('.') 

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
# We'll use scikit-learn for detailed metrics
from sklearn.metrics import classification_report, confusion_matrix

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

def evaluate_model(args):
    """
    Main function to load and evaluate the trained detector model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")

    # 1. --- Get the VALIDATION/TEST file list ---
    # We must evaluate on the files the model has NOT seen.
    all_files_in_dir = set(f for f in os.listdir(args.skeleton_dir) if f.endswith('.txt'))
    train_files = set(read_split_file(args.split_file))
    val_files = list(all_files_in_dir - train_files)
    
    if not val_files:
        print("Error: Validation/Test file list is empty. Cannot evaluate.")
        return

    print(f"Evaluating on {len(val_files)} files (the validation split).")

    # 2. --- Create the Validation Dataset and DataLoader ---
    val_dataset = PKUMMDDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        file_list=val_files,
        window_size=args.window_size,
        stride=args.stride
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # No need to shuffle for evaluation
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 3. --- Load the Trained Model ---
    # Instantiate the model with num_class=2 (binary detector)
    model = Model(num_class=2, num_point=25, num_person=2, graph='graph.graph.Graph', graph_args=dict()).to(device)
    
    # Load the saved weights
    model.load_state_dict(torch.load(args.model_path))
    
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()

    # 4. --- Run Evaluation ---
    all_labels = []
    all_predictions = []

    with torch.no_grad(): # Disable gradient calculation
        for data, labels in tqdm(val_loader, desc="Evaluating"):
            # Binarize the labels (ground truth)
            labels_binary = (labels > 0).long()
            data, labels = data.to(device), labels_binary.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            # Collect all labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 5. --- Print Detailed Metrics ---
    
    # Convert lists to NumPy arrays for scikit-learn
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Calculate overall accuracy
    accuracy = (all_predictions == all_labels).mean() * 100
    
    print("\n" + "="*50)
    print(f"         Final Model Evaluation Results")
    print("="*50 + "\n")
    print(f"Overall Accuracy: {accuracy:.2f}%\n")

    # Classification Report
    print("--- Classification Report ---")
    target_names = ['No Action (0)', 'Action (1)']
    print(classification_report(all_labels, all_predictions, target_names=target_names))
    print("\n")
    
    # Confusion Matrix
    print("--- Confusion Matrix ---")
    matrix = confusion_matrix(all_labels, all_predictions)
    print("                 Predicted: No Action  |  Predicted: Action")
    print(f"Actual: No Action  | {matrix[0, 0]:<20} | {matrix[0, 1]:<20}")
    print(f"Actual: Action     | {matrix[1, 0]:<20} | {matrix[1, 1]:<20}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CTR-GCN Action Detector")
    
    parser.add_argument('--model_path', type=str, default='detector_model.pth', help="Path to the saved model weights (.pth file)")
    
    parser.add_argument('--skeleton_dir', type=str, default='data/PKU_Skeleton_Renew')
    parser.add_argument('--label_dir', type=str, default='data/Vectorized_Labels')
    parser.add_argument('--split_file', type=str, default='data/splits/cross-view.txt')
    
    # These args must match the ones used for training/validation
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    evaluate_model(args)
