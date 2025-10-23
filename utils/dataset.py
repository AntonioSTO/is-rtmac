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
    
    This version includes a 'stride' parameter in its sliding window to
    efficiently sample the data and avoid massive redundancy.
    """
    def __init__(
        self,
        skeleton_dir: str,
        label_dir: str,
        file_list: List[str],
        window_size: int = 300,
        stride: int = 1,
        transform: Optional[Callable] = None,
    ) -> None:
        self.skeleton_dir = skeleton_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        
        self.num_person_in = 2
        self.num_point = 25
        self.num_channels = 3

        self.samples = []
        self._create_sample_map()

    def _create_sample_map(self):
        """
        Scans the files and creates a master list of all possible training samples (windows),
        using the specified stride to skip frames.
        """
        print(f"Creating data samples for {len(self.file_list)} files (window={self.window_size}, stride={self.stride})...")
        
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

            # --- THIS IS THE FIX ---
            # Define num_frames based on the length of the label vector
            num_frames = len(label_vector)
            
            # Now the 'num_frames' variable exists for this loop
            for start_frame in range(0, num_frames - self.window_size + 1, self.stride):
                center_frame_idx = start_frame + self.window_size // 2
                label = label_vector[center_frame_idx]
                self.samples.append((skeleton_path, start_frame, label))
                
        print(f"✅ Data map created. Found {len(self.samples)} total samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        skeleton_path, start_frame, label = self.samples[idx]
        
        full_skeleton_data = np.loadtxt(skeleton_path, dtype=np.float32)
        skeleton_window = full_skeleton_data[start_frame : start_frame + self.window_size]
        
        data = skeleton_window.reshape((self.window_size, self.num_person_in, self.num_point, self.num_channels))
        data = data.transpose((3, 0, 2, 1)) # (C, T, V, M)
        
        feature_tensor = torch.from_numpy(data)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)
            
        return feature_tensor, label