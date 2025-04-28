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
import concurrent.futures


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
    torchaudio.set_audio_backend("soundfile") # added recently
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
def preprocess_folder(input_dir, output_dir, max_workers=20):
    os.makedirs(output_dir, exist_ok=True)
    flac_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.flac'):
                flac_files.append(os.path.join(root, f))

    def process_file(input_path):
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

    num_processed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for success in tqdm(executor.map(process_file, flac_files), total=len(flac_files)):
            if success:
                num_processed += 1

    success_rate = num_processed / (len(flac_files) + 1e-6)
    print(f"\nCompleted with success rate: {success_rate:.2%}")