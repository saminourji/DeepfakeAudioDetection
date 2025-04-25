import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.datasets import TripletAudioDataset
from data.utils import preprocess_folder
from collections import defaultdict
import torchaudio

input_path = "../data/processed/ASVspoof2021_LA_eval/flac"
output_path = "../data/processed/tensors"
metadata_path = "../data/processed/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
mfcc_dir = "../data/processed/tensors"



preprocess_folder(input_path, output_path)

triplet_dataset = TripletAudioDataset(metadata_path, mfcc_dir)

total_bonafide = 0
total_spoof = 0
speaker_bonafide = defaultdict(int)
speaker_spoof = defaultdict(int)
all_speakers = set()

with open(metadata_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        speaker_id = parts[0]
        file_id = parts[1]
        label = parts[5]

        all_speakers.add(speaker_id)

        if label == "bonafide":
            total_bonafide += 1
            speaker_bonafide[speaker_id] += 1
        elif label == "spoof":
            total_spoof += 1
            speaker_spoof[speaker_id] += 1

print(f"- Total bonafide samples: {total_bonafide}")
print(f"- Total spoof samples: {total_spoof}")
print(f"- Unique speakers: {len(all_speakers)}")

for i in range(len(triplet_dataset)):
    print("\n\n\n")
    print(f"TRIPLET {i}")
    print(triplet_dataset[i])
