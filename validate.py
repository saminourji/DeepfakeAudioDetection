import torch
import numpy as np
import os
import itertools
from typing import List, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from model.siamese_cnn import SiameseCNN
from preprocess.datasets import TripletAudioDataset, BalancedBatchSampler

"""
Eval Metrics
"""

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
MODEL_DIR  = "train/saved_models"
META_PATH  = "data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
MFCC_DIR   = "data/tensors_EVAL"

def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    "Euclidean distance between two batches of embeddings."
    return (a-b).pow(2).sum(dim=1).sqrt()

def compute_eer_thr(labels: np.ndarray, scores: np.ndarray) -> float:
    "Compute Equal Error Rate. Invert scores so higher scores imply positive class."
    fpr, tpr, thresh = roc_curve(labels, scores, pos_label=1)
    fnr = 1-tpr
    index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[index] + fnr[index]) * 0.5
    return eer * 100.0, thresh[index]

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return accuracy_score(y_true, y_pred) * 100.0

@torch.inference_mode()
def evaluate_model(file_path: str) -> dict:
    """
    Goal: load "model_k.pth" and build an evaluation set with ablate_index = k
    Return a metrics dict
    """
    # Extract MFCC index "k" from file name
    basename = os.path.basename(file_path) # e.g. model_7.pth
    k = int(basename.split("_")[1].split(".")[0]) # k=7

    # Prepare dataset and dataloader
    val_ds = TripletAudioDataset(
        metadata_path=META_PATH,
        mfcc_dir=MFCC_DIR,
        ablate_idx=k
    )
    val_loader = DataLoader(val_ds, batch_sampler=BalancedBatchSampler(val_ds), pin_memory=True)

    # Prepare the model
    model = SiameseCNN().to(DEVICE)
    state = torch.load(file_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Grab the scores
    all_labels = []
    all_scores = []
    for x, y, _ in val_loader:
        x = x.to(DEVICE)
        embedding = model.encode(x).cpu()
        y = y.numpy()
        dists = torch.cdist(embedding, embedding, p=2).numpy()
        distance_matrix_size = len(y)
        for i, j in itertools.combinations(range(distance_matrix_size), 2):
            label_pair = 1 if (y[i] == 1 and y[j] == 1) else 0 # 1 if real-real, 0 otherwise
            score = -dists[i, j] # Larger values => more bonafide
            all_labels.append(label_pair)
            all_scores.append(score)
    all_labels = np.array(all_labels, dtype=float)
    all_scores = np.array(all_scores, dtype=float)
    eer, thr = compute_eer_thr(all_labels, all_scores)
    y_pred = (all_scores > thr).astype(int)
    acc = accuracy(all_labels, y_pred)
    return {"Accuracy (%)": acc, "EER (%)": eer, "Thr@EER": thr}


def main():
    # Sort the model files in numeric order; e.g. model_0.pth comes before model_1.pth
    model_files = []
    for fname in os.listdir(MODEL_DIR):
        if fname.startswith("model_") and fname.endswith(".pth"):
            model_files.append(os.path.join(MODEL_DIR, fname))
    model_files.sort(key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    print(f"{'Model':<12}  {'Acc(%)':>7}  {'EER(%)':>7}  Thr")
    print("-" * 38)
    for mp in model_files:
        m = evaluate_model(mp)
        print(f"{os.path.basename(mp):<12}  "
              f"{m['Accuracy (%)']:>7.2f}  "
              f"{m['EER (%)']:>7.2f}  {m['Thr@EER']:.3f}")

if __name__ == "__main__":
    main()