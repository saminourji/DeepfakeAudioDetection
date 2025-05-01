import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from model.siamese_cnn import SiameseCNN
from model.stac_loss import StacLoss
from preprocess.datasets import TripletAudioDataset, BalancedBatchSampler, batch_hard_triplet_loss

"""
RUNS TRAINING LOOP 
"""

# hyperparameters
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 500
MARGIN = 1.0
ABLATE_IDX = os.environ.get('ABLATE_IDX')
ABLATE_IDX = int(ABLATE_IDX) if ABLATE_IDX is not None else None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    "compute euclidean distance between two batches of embeddings"
    return (a-b).pow(2).sum(dim=1).sqrt()

def run_epoch(model, loader, loss_fn, optimizer=None):
    """pass through data once (1 epoch) for training if optimizer is provided, 
    otherwise eval. returns avg loss."""
    
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0

    for x, y, speaker in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        speaker = speaker.to(DEVICE)

        embedding = model.encode(x) # shape is (B, dimensionality of embedding)
        loss = batch_hard_triplet_loss(embedding, y, speaker, margin=MARGIN)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    
    return total_loss/len(loader)


def main():
    # expects .pt files of shape [N,1,T,F]
    train_ds = TripletAudioDataset(metadata_path='../data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt',
                                   mfcc_dir='../data/tensors', ablate_idx=ABLATE_IDX)


    train_loader = DataLoader(
        train_ds,
        batch_sampler=BalancedBatchSampler(train_ds, speakers_per_batch=4),
        num_workers=4
    )

    # build model + loss + optimizer
    model = SiameseCNN().to(DEVICE)
    loss_fn = StacLoss(MARGIN)
    optimizer = Adam(model.parameters(),lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)

    # training loop (checkpointing included)
    os.makedirs('saved_models', exist_ok=True)

    total_tr_loss = 0
    for epoch in range(1, EPOCHS+1):
        tr_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        total_tr_loss += tr_loss
        print(f'Epoch {epoch:03d} | train_loss: {tr_loss:.4f}')

    print(f"Model {ABLATE_IDX} | Avg. Loss: {total_tr_loss / EPOCHS}")

    save_path = f'saved_models/model_{ABLATE_IDX}.pth'
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
