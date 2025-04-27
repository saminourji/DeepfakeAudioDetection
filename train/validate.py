import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from model.siamese_cnn import SiameseCNN
from train.datasets import TripletAudioDataset

"""
Eval Metrics
"""

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
MODEL_PATH = 'checkpoint_best.pth'
VAL_TENSOR = 'data/val_tensor.pt'  

def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    "Euclidean distance between two batches of embeddings."
    return (a-b).pow(2).sum(dim=1).sqrt()

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    "Compute Equal Error Rate. Invert scores so higher scores imply positive class."
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1-tpr
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer


def main():
    # Loading model
    model = SiameseCNN().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Val triplets
    val_ds = TripletAudioDataset(VAL_TENSOR)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE)