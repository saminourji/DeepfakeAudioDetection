import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.datasets import TripletAudioDataset
from data.utils import preprocess_folder

input_path = "../data/processed/ASVspoof2021_LA_eval/flac"
output_path = "../data/processed/tensors"
metadata_path = "../data/processed/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
mfcc_dir = "../data/processed/tensors"

preprocess_folder(input_path, output_path)

triplet_dataset = TripletAudioDataset(metadata_path, mfcc_dir)

for i in range(len(triplet_dataset)):
    print("\n\n\n\n")
    print(f"TRIPLET {i}")
    print(triplet_dataset[i])