from preprocess.datasets import TripletAudioDataset
import torch
import os

input_path = "data/ASVspoof2021_LA_eval/flac"
output_path = "data/tensors"
if os.path.isdir("data/ASVspoof2021_LA_cm_protocols/"):
    metadata_path = "data/ASVspoof2021_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
else:
    metadata_path = "data/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
mfcc_dir = "data/tensors"

triplet_dataset = TripletAudioDataset(metadata_path, mfcc_dir)

triplets_list = []
for i in range(10):
    anchor, positive, negative = triplet_dataset[i]
    triplet = torch.stack([anchor, positive, negative], dim=0)  # Shape: (3, 1, 13, 400)
    triplets_list.append(triplet)

triplets_tensor = torch.stack(triplets_list, dim=0)

print(triplets_tensor.shape)
#torch.save(triplets_tensor, "first_ten_triplets.pt")