from preprocess.datasets import TripletAudioDataset
import torch
import os
from preprocess.utils import preprocess_folder

### BEFORE RUNNING THIS, MAKE SURE YOU RUN   'interact -n 20 -t 02:00:00 -m 20g' ###


def flac_to_tensors(input_path, metadata_path, mfcc_dir):
    if not os.path.exists(metadata_path):
        raise ValueError("THIS SHOULD NOT HAPPEN!")

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


# preprocess the training and evaluation flac files
if __name__ == "__main__":
    if not os.path.exists('data/tensors'):
        flac_to_tensors("data/LA/ASVspoof2019_LA_train/flac", "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt", "data/tensors")
    
    if not os.path.exists('data/tensors_EVAL'):
        flac_to_tensors("data/LA/ASVspoof2019_LA_eval/flac", "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt", "data/tensors_EVAL")