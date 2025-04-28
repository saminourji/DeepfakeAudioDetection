import sys
import os
import torch
from preprocess.utils import preprocess_audio

input_path = sys.argv[1]
output_dir = "data/tensors"  # Where you want .pt files saved

# Create output path
relative = os.path.basename(input_path).replace(".flac", ".pt")
output_path = os.path.join(output_dir, relative)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Process and save
mfcc = preprocess_audio(input_path)
torch.save(mfcc, output_path)