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


preprocess_folder(input_path, mfcc_dir)

triplet_dataset = TripletAudioDataset(metadata_path, mfcc_dir)


triplets_list = []
for i in range(10):
    anchor_file, pos_file, neg_file, _ = triplet_dataset[i]

    anchor = torch.load(os.path.join(mfcc_dir, anchor_file)).unsqueeze(0)
    positive = torch.load(os.path.join(mfcc_dir, pos_file)).unsqueeze(0)
    negative = torch.load(os.path.join(mfcc_dir, neg_file)).unsqueeze(0)

    triplet = torch.stack([anchor, positive, negative], dim=0)  # Shape: (3, 1, 13, 400)
    triplets_list.append(triplet)

triplets_tensor = torch.stack(triplets_list, dim=0)

print(triplets_tensor.shape)
#torch.save(triplets_tensor, "first_ten_triplets.pt")