
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.siamese_cnn import SiameseCNN
from models.stac_loss import StacLoss
from train.datasets import TripletAudioDataset

"""
RUNS TRAINING LOOP 
"""

# hyperparameters
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10
MARGIN = 1.0
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

    for A,P,N in loader:
        A,P,N = A.to(DEVICE), P.to(DEVICE), N.to(DEVICE)
        d_pos = model.forward(A, P)
        d_neg = model.forward(A, N)
        loss = loss_fn(d_pos, d_neg)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss+=loss.item()
    return total_loss/len(loader)


def main():
    # expects .pt files of shape [N,1,T,F]
    train_ds = TripletAudioDataset('data/train_tensor.pt')
    val_ds = TripletAudioDataset('data/val_tensor.pt')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # build model + loss + optimizer
    model = SiameseCNN().to(DEVICE)
    loss_fn = StacLoss(MARGIN)
    optimizer = Adam(model.parameters(),lr=LR, betas=(0.9, 0.999), weight_decay=1e-4)

    # training loop (checkpointing included)
    best_val, wait = float('inf'),0
    for epoch in range(1, EPOCHS+1):
        tr_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        va_loss = run_epoch(model, val_loader, loss_fn, None)
        print(f'Epoch {epoch:03d} | train_loss: {tr_loss:.4f} | validation_loss: {va_loss:.4f}')

        if va_loss < best_val:
            best_val, wait = va_loss, 0
            torch.save(model.state_dict(), 'checkpoint_best.pth')
            print("Saved new best checkpoint")
        else:
            wait +=1
            if wait >= PATIENCE:
                print("Early stop because no val improvement in " # early stoppage if we don't see an improvement
                      f"{PATIENCE} epochs")                      # will be useful considering size of data set.
                break


if __name__ == '__main__':
    main()
