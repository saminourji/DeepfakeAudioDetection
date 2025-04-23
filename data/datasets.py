"""
PRE-PROCESSES DATA FOR MODEL
- loads triplets
- converts audio to tensors
- prepares batches for model
"""

from torch.utils.data import Dataset
from utils import preprocess_folder
from collections import defaultdict
import os
import random
import torch
import torchaudio

class TripletAudioDataset(Dataset):
    def __init__(self, metadata_path, mfcc_dir):
        self.triplets = []
        self.mfcc_dir = mfcc_dir
        self.speaker_to_bonafide = defaultdict(list)
        self.speaker_to_spoof = defaultdict(list)

        # reads the metadata file
        with open(metadata_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                speaker_id = parts[0]
                file_id = parts[1]
                label = parts[5]
                file_name = file_id + ".pt"

                if label == "bonafide":
                    self.speaker_to_bonafide[speaker_id].append(file_name)
                elif label == "spoof":
                    self.speaker_to_spoof[speaker_id].append(file_name)

        num_success = 0.0
        num_fails = 0.0
        for speaker in self.speaker_to_spoof:
            if speaker not in self.speaker_to_bonafide or len(self.speaker_to_bonafide[speaker]) < 2:
                continue

            
            for spoof_file in self.speaker_to_spoof[speaker]:
                anchor_file = random.choice(self.speaker_to_bonafide[speaker])
                positive_candidates = [f for f in self.speaker_to_bonafide[speaker] if f != anchor_file]
                if not positive_candidates:
                    continue
                positive_file = random.choice(positive_candidates)

                try:
                    # unsqueezes allows for a CNN to pass over it, so shape (1, 13, 400)
                    anchor = torch.load(os.path.join(self.mfcc_dir, anchor_file)).unsqueeze(0)
                    positive = torch.load(os.path.join(self.mfcc_dir, positive_file)).unsqueeze(0)
                    negative = torch.load(os.path.join(self.mfcc_dir, spoof_file)).unsqueeze(0)
                    print(f"Triplet successfully created!")
                    num_success += 1
                except Exception as e:
                    print(f"Skipping triplet due to error: {e}")
                    num_fails += 1
                    continue
                    

                self.triplets.append((anchor, positive, negative))
        
        print(f"{num_success / (num_success + num_fails)} successful conversion rate")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        return self.triplets[index]

# run datasets.py to load tensors
if __name__ == "__main__":
    input_path = "processed/ASVspoof2021_LA_eval/flac"
    output_path = "processed/tensors"
    metadata_path = "processed/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
    mfcc_dir = "processed/tensors"

    #preprocess_folder(input_path, output_path)

    triplet_dataset = TripletAudioDataset(metadata_path, mfcc_dir)
    
    #for i in range(len(triplet_dataset)):
        #print(triplet_dataset[i])