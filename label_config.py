import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
# These paths assume your script is in the same parent folder as the dataset folders.
# If not, you'll need to provide the full path to them.
SKELETON_DIR = Path("./Data/Skeleton/PKU_Skeleton_Renew")
LABEL_DIR = Path("./Label/Train_Label_PKU_final")
OUTPUT_DIR = Path("Vectorized_Labels")


def create_vectorized_labels():
    """
    Reads skeleton and label files to create frame-wise label vectors.
    """
    # 1. Create the output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✅ Output directory '{OUTPUT_DIR}' is ready.")

    label_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")])
    
    if not label_files:
        print(f"❌ Error: No label files found in '{LABEL_DIR}'. Please check the path.")
        return

    print(f"Found {len(label_files)} label files to process...")

    # 2. Iterate over each file in the label directory with a progress bar
    for label_filename in tqdm(label_files, desc="Processing Labels"):
        
        # 3. Define paths for corresponding files
        label_filepath = LABEL_DIR / label_filename
        skeleton_filepath = SKELETON_DIR / label_filename
        output_filepath = OUTPUT_DIR / label_filename

        # Check if the corresponding skeleton file exists
        if not skeleton_filepath.exists():
            print(f"\n[!] Warning: Skeleton file not found for '{label_filename}'. Skipping.")
            continue

        try:
            # 4. Get the total number of frames by counting lines in the skeleton file
            with open(skeleton_filepath, 'r') as f:
                num_frames = len(f.readlines())
            
            if num_frames == 0:
                print(f"\n[!] Warning: Skeleton file '{label_filename}' is empty. Skipping.")
                continue

            # 5. Initialize a NumPy vector of zeros with length equal to num_frames
            label_vector = np.zeros(num_frames, dtype=int)

            # 6. Read the original label file to get action intervals
            with open(label_filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    action_id = int(parts[0])
                    start_frame = int(parts[1])
                    end_frame = int(parts[2])

                    # Convert 1-based frame indices from the file to 0-based Python indices.
                    # The end_frame in the label file is inclusive, so a slice from
                    # [start_frame - 1] up to (but not including) [end_frame] is correct.
                    start_idx = start_frame - 1
                    end_idx = end_frame
                    
                    # Ensure indices are within the bounds of the vector
                    if start_idx < num_frames and end_idx > 0:
                        label_vector[max(0, start_idx):min(num_frames, end_idx)] = action_id

            # 7. Save the new vectorized label to the output directory
            # Each frame's label will be on a new line.
            np.savetxt(output_filepath, label_vector, fmt='%d')

        except Exception as e:
            print(f"\n[!] Error processing file {label_filename}: {e}")

    print(f"\n🎉 --- Process Complete! --- 🎉")
    print(f"All vectorized labels have been saved to the '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    create_vectorized_labels()
