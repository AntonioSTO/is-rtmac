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
# 1. Import the new augmentation
from augmentations import RandomRotation3DTransform, TranslateToOrigin

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

# 2. Add the AugmentationWrapper to handle tensor shape permutations
class AugmentationWrapper(nn.Module):
    """
    Wraps the augmentation functions to handle the (C, T, V, M) -> (M, T, V, C)
    shape permutation required by the augmentation code.
    """
    def __init__(self, transforms):
        super().__init__()
        # Use nn.Sequential to chain the transforms
        self.transforms = nn.Sequential(*transforms)

    def forward(self, x):
        # Input x shape is (C, T, V, M)
        # Permute to what augmentations expect: (M, T, V, C)
        x = x.permute(3, 1, 2, 0).contiguous()
        
        # Apply the sequence of transforms (e.g., Translate, then Rotate)
        x = self.transforms(x) 
        
        # Permute it back to the (C, T, V, M) shape the model expects
        x = x.permute(3, 1, 2, 0).contiguous()
        return x

def train_and_evaluate(args):
    """
    Main function to train the MULTI-CLASS ACTION CLASSIFIER (Actions Only).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"--- Training Action Classifier (Augmentations=ON, Dropout={args.dropout}, WeightDecay={args.weight_decay}) ---")

    print(f"Loading official split from: {args.split_file}")
    all_files_in_dir = set(f for f in os.listdir(args.skeleton_dir) if f.endswith('.txt'))
    train_files = read_split_file(args.split_file)
    val_files = list(all_files_in_dir - set(train_files))
    
    if not train_files:
        print("Error: Training file list is empty.")
        return

    print(f"Found {len(all_files_in_dir)} total files.")
    print(f"Training on {len(train_files)} files, Validating on {len(val_files)} files.")

    # 3. Instantiate the full augmentation pipeline
    # NOTE: Assuming joint 21 (index 20) is the 'SpineBase' for TranslateToOrigin
    train_transform = AugmentationWrapper([
        TranslateToOrigin(joint_idx=20),
        RandomRotation3DTransform(max_angle=args.rot_angle)
    ])

    # 4. Apply transforms to the training dataset ONLY
    train_dataset = PKUMMDDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        file_list=train_files,
        window_size=args.window_size,
        stride=args.stride,
        transform=train_transform  # <--- PASS THE AUGMENTATION CHAIN
    )
    val_dataset = PKUMMDDataset(
        skeleton_dir=args.skeleton_dir,
        label_dir=args.label_dir,
        file_list=val_files,
        window_size=args.window_size,
        stride=args.stride,
        transform=None  # <--- NO AUGMENTATION on validation data
    )

    print(f"Original training samples: {len(train_dataset.samples)}")
    train_dataset.samples = [s for s in train_dataset.samples if s[2] > 0]
    print(f"Filtered training samples (actions only): {len(train_dataset.samples)}")
    
    print(f"Original validation samples: {len(val_dataset.samples)}")
    val_dataset.samples = [s for s in val_dataset.samples if s[2] > 0]
    print(f"Filtered validation samples (actions only): {len(val_dataset.samples)}")

    print("Calculating class weights for action classification...")
    all_labels_actions = [(s[2] - 1) for s in train_dataset.samples]
    if not all_labels_actions:
        print("❌ Error: No action samples (label > 0) found in the training set.")
        return
    class_counts = torch.bincount(torch.tensor(all_labels_actions), minlength=51)
    class_weights = 1. / class_counts.float()
    class_weights[class_weights == float('inf')] = 0
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )

    model = Model(
        num_class=51, num_point=25, num_person=2, 
        graph='graph.graph.Graph', graph_args=dict(),
        drop_out=args.dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 5. Add the Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # Track max validation accuracy
        factor=0.1,      # Reduce LR by 10x
        patience=5,      # Wait 5 epochs with no improvement before dropping
            )

    best_val_accuracy = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for data, labels in tqdm(train_loader, desc="Training Classifier"):
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

        # 6. Step the scheduler with the validation accuracy
        scheduler.step(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.save_path)
            print(f"🎉 New best CLASSIFIER model saved to {args.save_path} with accuracy: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CTR-GCN Action Classifier (Actions Only)")
    parser.add_argument('--skeleton_dir', type=str, default='data/PKU_Skeleton_Renew')
    parser.add_argument('--label_dir', type=str, default='data/Vectorized_Labels')
    parser.add_argument('--split_file', type=str, default='data/splits/cross-subject.txt')
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # --- Regularization Arguments ---
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout probability (e.g., 0.5)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization (e.g., 0.0001)")
    parser.add_argument('--rot_angle', type=float, default=0.3, help="Max random rotation angle in radians (e.g., 0.3)")
    
    parser.add_argument('--save_path', type=str, default='classifier_model.pth')
    
    args = parser.parse_args()
    train_and_evaluate(args)
