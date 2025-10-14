import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple
from tqdm import tqdm

class PKUMMDDataset(Dataset):
    """
    Custom Dataset class for loading and processing PKU-MMD skeleton data
    for the CTR-GCN model. It uses a sliding window approach to create
    fixed-size samples from full video sequences.
    """

    def __init__(
        self,
        skeleton_dir: str,
        label_dir: str,
        window_size: int = 300,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            skeleton_dir (str): Path to the directory with skeleton files (e.g., 'PKU_Skeleton_Renew').
            label_dir (str): Path to the directory with vectorized label files (e.g., 'Vectorized_Labels').
            window_size (int): The number of frames for each sample clip. This is the 'T' dimension for the model.
            transform (Optional[Callable]): Optional transform to be applied on a data sample.
        """
        self.skeleton_dir = skeleton_dir
        self.label_dir = label_dir
        self.window_size = window_size
        self.transform = transform
        
        # --- Model & Data Shape Parameters ---
        self.num_person_in = 2  # M as in the paper
        self.num_point = 25     # V as in the paper
        self.num_channels = 3   # C as in the paper (x, y, z)

        # This list will store tuples of (file_path, start_frame, label) for each valid window
        self.samples = []
        
        self._create_sample_map()

    def _create_sample_map(self):
        """
        Private helper method to scan the data directories and create a map of all
        possible training samples (windows). This is done once during initialization.
        """
        print("Creating data samples map...")
        file_list = sorted(os.listdir(self.skeleton_dir))

        for filename in tqdm(file_list, desc="Scanning files"):
            if not filename.endswith(".txt"):
                continue

            skeleton_path = os.path.join(self.skeleton_dir, filename)
            label_path = os.path.join(self.label_dir, filename)

            if not os.path.exists(label_path):
                continue
            
            # Load the entire label vector for this video sequence
            try:
                label_vector = np.loadtxt(label_path, dtype=int)
            except Exception as e:
                print(f"Warning: Could not load label file {label_path}. Skipping. Error: {e}")
                continue

            # Videos with fewer frames than the window size are skipped
            if len(label_vector) < self.window_size:
                continue

            # Use a sliding window to create samples
            num_frames = len(label_vector)
            for start_frame in range(num_frames - self.window_size + 1):
                
                # We use the label of the center frame as the label for the entire window
                center_frame_idx = start_frame + self.window_size // 2
                label = label_vector[center_frame_idx]
                
                # OPTIONAL: You can add logic here to handle class imbalance. For example,
                # you might want to skip some windows with the "no action" (label 0) class.
                # Example: if label == 0 and np.random.rand() > 0.1: continue
                
                self.samples.append((skeleton_path, start_frame, label))
        
        print(f"✅ Data map created. Found {len(self.samples)} total samples.")

    def __len__(self) -> int:
        """Returns the total number of samples (windows) in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Fetches a single data sample (a window of skeleton data) and its corresponding label.
        
        This method performs the crucial step of reshaping the data to match the
        model's expected input tensor shape of (C, T, V, M).
        """
        skeleton_path, start_frame, label = self.samples[idx]

        # Load the full skeleton data from the text file
        # Note: For very large datasets, this I/O can be a bottleneck.
        # Consider converting data to a binary format like .npy for faster loading.
        full_skeleton_data = np.loadtxt(skeleton_path, dtype=np.float32)

        # Extract the frames corresponding to the window
        skeleton_window = full_skeleton_data[start_frame : start_frame + self.window_size]
        
        # --- Reshape the data for the CTR-GCN model ---
        # The raw data shape is (T, 150)
        # We need to reshape it to (C, T, V, M)
        
        # Reshape from (T, 150) to (T, M, V, C)
        # 150 = 2 persons * 25 joints * 3 coordinates
        data = skeleton_window.reshape((self.window_size, self.num_person_in, self.num_point, self.num_channels))
        
        # Permute the axes to get the final shape: (C, T, V, M)
        data = data.transpose((3, 0, 2, 1))
        
        # Convert the NumPy array to a PyTorch tensor
        feature_tensor = torch.from_numpy(data)

        if self.transform:
            feature_tensor = self.transform(feature_tensor)
            
        return feature_tensor, label