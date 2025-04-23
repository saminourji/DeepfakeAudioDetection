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
from collections import defaultdict

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

    if waveform.shape[0] > 1: # "if waveform is NOT mono"
        waveform = waveform.mean(dim=0, keepdim=True) # convert audio to mono
    
    if sample_rate != SAMPLE_RATE:
        resampler = Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)
    
    mfcc = mfcc_transform(waveform)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6) # normalize sample
    mfcc =  mfcc.squeeze(0)

    T = mfcc.shape[1]

    if T < max_frames:
        pad_amount = max_frames - T
        mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount)) # add padding
    else:
        mfcc = mfcc[:, :max_frames]
    
    return mfcc # (13, max_frames) -> defaults to (13, 400)


# applies preprocess_audio to a collection of .flac audio files by looping over them
# saves them to a folder
def preprocess_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True) # ensures that the directory exists
    flac_files = [f for f in os.listdir(input_dir) if f.endswith('.flac')]

    for fname in tqdm(flac_files, desc=f"Processing {input_dir}"):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname.replace(".flac", ".pt"))

        try:
            mfcc = preprocess_audio(input_path)
            torch.save(mfcc, output_path)
            print(f"Created tensor for {fname}")
        except Exception as e:
            print(f"Skipping {fname}: {e}")