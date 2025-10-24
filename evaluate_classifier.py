# evaluate_classifier.py
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
    Main function to load and evaluate the trained CLASSIFIER model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading classifier model from: {args.model_path}")

    # 1. --- Get the VALIDATION/TEST file list ---
    all_files_in_dir = set(f for f in os.listdir(args.skeleton_dir) if f.endswith('.txt'))
    train_files = set(read_split_file(args.split_file))
    val_files = list(all_files_in_dir - train_files)
    
    if not val_files:
        print("Error: Validation/Test file list is empty. Cannot evaluate.")
        return

    print(f"Evaluating on {len(val_files)} files (the validation split).")

    # 2. --- Create the Validation Dataset (with filtering) ---
    val_dataset = PKUMMDDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        file_list=val_files,
        window_size=args.window_size,
        stride=args.stride,
        transform=None # No augmentations during evaluation
    )

    # --- KEY CHANGE: Filter dataset to ONLY include action samples ---
    print(f"Original validation samples: {len(val_dataset.samples)}")
    val_dataset.samples = [s for s in val_dataset.samples if s[2] > 0]
    print(f"Filtered validation samples (actions only): {len(val_dataset.samples)}")

    if not val_dataset.samples:
        print("❌ Error: No action samples found in the validation set.")
        return

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 3. --- Load the Trained Model (Classifier) ---
    # --- KEY CHANGE: num_class is 51 ---
    model = Model(
        num_class=51, # 51 action classes
        num_point=25, 
        num_person=2, 
        graph='graph.graph.Graph', 
        graph_args=dict(),
        drop_out=0 # Dropout is off during eval anyway, but good practice
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # 4. --- Run Evaluation ---
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc="Evaluating Classifier"):
            # --- KEY CHANGE: Remap labels from [1...51] to [0...50] ---
            labels_remapped = labels - 1
            data, labels = data.to(device), labels_remapped.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 5. --- Print Detailed Metrics ---
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    accuracy = (all_predictions == all_labels).mean() * 100
    
    print("\n" + "="*50)
    print(f"       Final CLASSIFIER Model Evaluation")
    print("="*50 + "\n")
    print(f"Overall Accuracy: {accuracy:.2f}%\n")

    # Classification Report
    print("--- Classification Report (Top 5 & Bottom 5 classes by F1-score) ---")
    # Generate target names for 51 classes (e.g., "Action 1", "Action 2", ...)
    target_names = [f"Action {i+1}" for i in range(51)]
    
    report_dict = classification_report(
        all_labels, 
        all_predictions, 
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    # Get scores for each class and sort them
    class_scores = []
    for class_name, metrics in report_dict.items():
        if class_name in target_names:
            class_scores.append((class_name, metrics['f1-score'], metrics['support']))

    class_scores.sort(key=lambda x: x[1], reverse=True)

    print("\n--- BEST Performing Classes (by F1-score) ---")
    for name, f1, support in class_scores[:5]:
        print(f"{name:<12} | F1-Score: {f1:.2f} | Samples: {support}")
        
    print("\n--- WORST Performing Classes (by F1-score) ---")
    for name, f1, support in class_scores[-5:]:
        print(f"{name:<12} | F1-Score: {f1:.2f} | Samples: {support}")
    
    print("\n--- Full Report (for copy/paste) ---")
    print(classification_report(
        all_labels, 
        all_predictions, 
        target_names=target_names,
        zero_division=0
    ))
    
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained CTR-GCN Action Classifier")
    
    parser.add_argument('--model_path', type=str, default='classifier_model.pth', help="Path to the saved classifier weights (.pth file)")
    
    parser.add_argument('--skeleton_dir', type=str, default='data/PKU_Skeleton_Renew')
    parser.add_argument('--label_dir', type=str, default='data/Vectorized_Labels')
    parser.add_argument('--split_file', type=str, default='data/splits/cross-view.txt')
    
    # These args must match the ones used for training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=16) 
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    evaluate_model(args)
