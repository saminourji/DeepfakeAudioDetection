from preprocess.datasets import TripletAudioDataset
import torch
import os
from preprocess.utils import preprocess_folder

### BEFORE RUNNING THIS, MAKE SURE YOU RUN   'interact -n 20 -t 02:00:00 -m 20g' ###


input_path = "data/ASVspoof2021_LA_eval/flac"
if os.path.isdir("data/ASVspoof2021_LA_cm_protocols/"):
    metadata_path = "data/ASVspoof2021_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
else:
    metadata_path = "data/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
mfcc_dir = "data/tensors"


# Count number of .flac files to process
flac_files = []
for root, _, files in os.walk(input_path):
    for f in files:
        if f.endswith(".flac"):
            flac_files.append(os.path.join(root, f))

# Count number of .pt files already preprocessed
pt_files = []
for root, _, files in os.walk(mfcc_dir):
    for f in files:
        if f.endswith(".pt"):
            pt_files.append(os.path.join(root, f))

# Check if preprocessing is needed
if len(pt_files) < len(flac_files):
    print(f"Preprocessing needed: found {len(pt_files)} tensors for {len(flac_files)} audio files.")
    preprocess_folder(input_path, mfcc_dir)
else:
    print(f"Preprocessing already completed: {len(pt_files)} tensors found.")

triplet_dataset = TripletAudioDataset(metadata_path, mfcc_dir)


# triplets_list = []
# for i in range(10):
#     anchor, positive, negative = triplet_dataset[i]

#     triplet = torch.stack([anchor, positive, negative], dim=0)  # Shape: (3, 1, 13, 400)
#     triplets_list.append(triplet)

# triplets_tensor = torch.stack(triplets_list, dim=0)

# print(triplets_tensor.shape)
#torch.save(triplets_tensor, "first_ten_triplets.pt")