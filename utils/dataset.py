# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple, List
from tqdm import tqdm

class PKUMMDDataset(Dataset):
    """
    MODIFIED: This version accepts a specific list of filenames to use,
    allowing for programmatic train/val splits.
    """
    def __init__(
        self,
        skeleton_dir: str,
        label_dir: str,
        file_list: List[str], # <-- NEW: Pass the list of files to use
        window_size: int = 300,
        transform: Optional[Callable] = None,
    ) -> None:
        self.skeleton_dir = skeleton_dir
        self.label_dir = label_dir
        self.file_list = file_list # <-- NEW: Store the file list
        self.window_size = window_size
        self.transform = transform
        
        self.num_person_in = 2
        self.num_point = 25
        self.num_channels = 3

        self.samples = []
        self._create_sample_map()

    def _create_sample_map(self):
        print(f"Creating data samples for {len(self.file_list)} files...")
        # MODIFIED: We now iterate over the provided self.file_list
        # instead of listing the whole directory.
        for filename in tqdm(self.file_list, desc="Scanning files"):
            if not filename.endswith(".txt"): continue
            skeleton_path = os.path.join(self.skeleton_dir, filename)
            label_path = os.path.join(self.label_dir, filename)
            if not os.path.exists(label_path): continue
            
            try:
                label_vector = np.loadtxt(label_path, dtype=int)
            except Exception: continue
            if len(label_vector) < self.window_size: continue

            num_frames = len(label_vector)
            for start_frame in range(num_frames - self.window_size + 1):
                center_frame_idx = start_frame + self.window_size // 2
                label = label_vector[center_frame_idx]
                self.samples.append((skeleton_path, start_frame, label))
        print(f"Data map created. Found {len(self.samples)} total samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        skeleton_path, start_frame, label = self.samples[idx]
        full_skeleton_data = np.loadtxt(skeleton_path, dtype=np.float32)
        skeleton_window = full_skeleton_data[start_frame : start_frame + self.window_size]
        
        data = skeleton_window.reshape((self.window_size, self.num_person_in, self.num_point, self.num_channels))
        data = data.transpose((3, 0, 2, 1))
        feature_tensor = torch.from_numpy(data)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)
            
        return feature_tensor, label