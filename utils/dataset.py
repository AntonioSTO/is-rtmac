# utils/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List
from tqdm import tqdm

class PKUMMDDataset(Dataset):
    """
    Custom Dataset class for loading and processing PKU-MMD skeleton data.
    
    This version accepts a specific list of filenames, allowing for programmatic 
    train/val splits. It uses a sliding window approach to create fixed-size 
    samples from full video sequences.
    """
    def __init__(
        self,
        skeleton_dir: str,
        label_dir: str,
        file_list: List[str],
        window_size: int = 300,
        transform: Optional[Callable] = None,
    ) -> None:
        self.skeleton_dir = skeleton_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.window_size = window_size
        self.transform = transform
        
        # --- Constants for PKU-MMD data shape ---
        self.num_person_in = 2  # M
        self.num_point = 25     # V (joints)
        self.num_channels = 3   # C (x, y, z coordinates)

        self.samples = []
        self._create_sample_map()

    def _create_sample_map(self):
        """
        Scans the files specified in self.file_list and creates a master list of all
        possible training samples (windows). This is done only once during initialization.
        """
        print(f"Creating data samples for {len(self.file_list)} files...")
        
        for filename in tqdm(self.file_list, desc="Scanning files"):
            if not filename.endswith(".txt"):
                continue
                
            skeleton_path = os.path.join(self.skeleton_dir, filename)
            label_path = os.path.join(self.label_dir, filename)

            if not os.path.exists(label_path):
                print(f"Warning: Label file not found for {filename}. Skipping.")
                continue
            
            try:
                label_vector = np.loadtxt(label_path, dtype=int)
            except Exception as e:
                print(f"Warning: Could not load label file {label_path}. Skipping. Error: {e}")
                continue

            if len(label_vector) < self.window_size:
                continue

            # Slide a window across the video's frames to create samples
            num_frames = len(label_vector)
            for start_frame in range(num_frames - self.window_size + 1):
                center_frame_idx = start_frame + self.window_size // 2
                label = label_vector[center_frame_idx]
                
                # Each sample is a "recipe" tuple: (path, start_frame, label)
                self.samples.append((skeleton_path, start_frame, label))
                
        print(f"✅ Data map created. Found {len(self.samples)} total samples.")

    def __len__(self) -> int:
        """Returns the total number of samples (windows) in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Fetches and prepares a single data sample and its corresponding label.
        """
        skeleton_path, start_frame, label = self.samples[idx]
        
        full_skeleton_data = np.loadtxt(skeleton_path, dtype=np.float32)
        skeleton_window = full_skeleton_data[start_frame : start_frame + self.window_size]
        
        # --- Reshape data for the GCN model: from (T, 150) to (C, T, V, M) ---
        data = skeleton_window.reshape((self.window_size, self.num_person_in, self.num_point, self.num_channels))
        data = data.transpose((3, 0, 2, 1)) # (C, T, V, M)
        
        feature_tensor = torch.from_numpy(data)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)
            
        return feature_tensor, label