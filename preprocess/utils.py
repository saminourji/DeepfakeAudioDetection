"""
CONTAINS HELPER FUNCTIONS
- mfcc extraction
- normalization
- audio augmentation
"""

import os
import torchaudio
import torch
from tqdm import tqdm
from torchaudio.transforms import MFCC, Resample

### GLOBAL VARIABLES ###
SAMPLE_RATE = 16000
N_MFCC = 13

mfcc_transform = MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
)

# takes a filepath of a .flac file and returns a processed tensor of shape (13, T)
def preprocess_audio(filepath, max_frames=400):
    waveform, sample_rate = torchaudio.load(filepath)

    if waveform.shape[0] > 1:  # if waveform is NOT mono
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        resampler = Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    mfcc = mfcc_transform(waveform)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)  # normalize sample
    mfcc = mfcc.squeeze(0)

    T = mfcc.shape[1]

    if T < max_frames:
        pad_amount = max_frames - T
        mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount))
    else:
        mfcc = mfcc[:, :max_frames]

    return mfcc  # (13, max_frames)

# applies preprocess_audio to a single .flac file
def process_file(input_path, input_dir, output_dir):
    relative_path = os.path.relpath(input_path, input_dir)
    output_path = os.path.join(output_dir, relative_path.replace(".flac", ".pt"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        mfcc = preprocess_audio(input_path)
        torch.save(mfcc, output_path)
        return True
    except Exception as e:
        print(f"Skipping {input_path}: {e}")
        return False

def preprocess_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    flac_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.flac'):
                flac_files.append(os.path.join(root, f))

    num_processed = 0

    for f in tqdm(flac_files, desc="Processing files"):
        if process_file(f, input_dir, output_dir):
            num_processed += 1

    success_rate = num_processed / (len(flac_files) + 1e-6)
    print(f"\nCompleted with success rate: {success_rate:.2%}")

def extract_file_paths_from_metadata(metadata_path, input_base_dir):
    file_paths = []
    with open(metadata_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            file_id = parts[1]
            file_path = os.path.join(input_base_dir, file_id + ".flac")
            file_paths.append(file_path)
    return file_paths

if __name__ == "__main__":
    input_path = "data/ASVspoof2021_LA_eval/flac"
    output_path = "data/tensors"
    if os.path.isdir("data/ASVspoof2021_LA_cm_protocols/"):
        metadata_path = "data/ASVspoof2021_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    else:
        metadata_path = "data/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
    mfcc_dir = "data/tensors"

    flac_files = extract_file_paths_from_metadata(metadata_path, input_path)

    with open("flac_files.txt", "w") as f:
        for file in flac_files:
            f.write(file + "\n")