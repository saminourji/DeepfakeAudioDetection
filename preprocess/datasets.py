"""
PRE-PROCESSES DATA FOR MODEL
- loads triplets
- converts audio to tensors
- prepares batches for model
"""

# sets root path as the root folder DeepFakeAudioDetection

from torch.utils.data import Dataset
from torch.utils.data import Sampler
import torch, torch.nn.functional as F
from .utils import preprocess_folder
from collections import defaultdict
import os
import random
import torch
# import torchaudio

class TripletAudioDataset(Dataset):
    def __init__(self, metadata_path, mfcc_dir, ablate_idx: int | None = None):
        super().__init__()
        self.mfcc_dir = mfcc_dir
        self.ablate_idx = ablate_idx
        self.items = [] # list of (file_name, label, speaker_id)
        with open(metadata_path) as f:
            for line in f:
                parts = line.strip().split()
                speaker_id = parts[0]
                file_id = parts[1]
                label_text = parts[4]
                file_name = file_id + ".pt"
                label = 1 if label_text == "bonafide" else 0
                self.items.append((file_name, label, speaker_id))
        print(f"Loaded {len(self.items):,} utterances")


        ############ ALL BELOW INCLUDED IN PREV AND RECENT ############
        # self.triplets = [] # Triplets in the form: (speaker, anchor_file, pos_file, neg_file, anchor_label)
        # self.mfcc_dir = mfcc_dir
        # self.speaker_to_bonafide = defaultdict(list)
        # self.speaker_to_spoof = defaultdict(list)

        # # Reads the metadata file
        # with open(metadata_path, "r") as f:
        #     for line in f:
        #         parts = line.strip().split()
        #         speaker_id = parts[0]
        #         file_id = parts[1]
        #         label = parts[5]
        #         file_name = file_id + ".pt"

        #         if label == "bonafide":
        #             self.speaker_to_bonafide[speaker_id].append(file_name)
        #         elif label == "spoof":
        #             self.speaker_to_spoof[speaker_id].append(file_name)
        ############ ALL ABOVE INCLUDED IN PREV AND RECENT ############

        ############# RECENT #############
        # Build triplets
        # for speaker in self.speaker_to_spoof.keys() | self.speaker_to_bonafide.keys():
        #     spoofed = self.speaker_to_spoof.get(speaker, [])
        #     bonafide = self.speaker_to_bonafide.get(speaker, [])
        #     if len(bonafide) >= 2 and spoofed: # Anchor / Positive => BONAFIDE
        #         self.build_triplets(spk=speaker, same=bonafide, opp=spoofed, anchor_label=1)
        #     if len(spoofed) >= 2 and bonafide: # Anchor / Positive => SPOOFED
        #         self.build_triplets(spk=speaker, same=spoofed, opp=bonafide, anchor_label=0)
        # print(f"Triplets: {len(self.triplets):,}")

        ############ PREVIOUS ############
        # num_success = 0.0
        # num_fails = 0.0
        # for speaker in self.speaker_to_spoof:
        #     if speaker not in self.speaker_to_bonafide or len(self.speaker_to_bonafide[speaker]) < 2:
        #         continue
        #     for spoof_file in self.speaker_to_spoof[speaker]:
        #         anchor_file = random.choice(self.speaker_to_bonafide[speaker])
        #         positive_candidates = [f for f in self.speaker_to_bonafide[speaker] if f != anchor_file]
        #         if not positive_candidates:
        #             continue
        #         positive_file = random.choice(positive_candidates)
        #         try:
        #             # unsqueezes allows for a CNN to pass over it, so shape (1, 13, 400)
        #             anchor = torch.load(os.path.join(self.mfcc_dir, anchor_file)).unsqueeze(0)
        #             positive = torch.load(os.path.join(self.mfcc_dir, positive_file)).unsqueeze(0)
        #             negative = torch.load(os.path.join(self.mfcc_dir, spoof_file)).unsqueeze(0)
        #             print(f"Triplet successfully created!")
        #             num_success += 1
        #         except Exception as e:
        #             print(f"Skipping triplet due to error: {e}")
        #             num_fails += 1
        #             continue
        #         self.triplets.append((anchor, positive, negative))
        # print(f"{num_success / (num_success + num_fails)} successful conversion rate")
    
    # def build_triplets(self, spk, same, opp, anchor_label):
    #     for anchor_file in same:
    #         pos_file = random.choice([f for f in same if f != anchor_file])
    #         neg_file = random.choice(opp)
    #         self.triplets.append((spk, anchor_file, pos_file, neg_file, anchor_label))

    def __len__(self):
        # return len(self.triplets)
        return len(self.items)

    def __getitem__(self, index):
        # return self.triplets[index]
        file_name, label, speaker = self.items[index]
        tensor = torch.load(os.path.join(self.mfcc_dir, file_name)).unsqueeze(0)
        if self.ablate_idx is not None:
            k = self.ablate_idx
            tensor = torch.cat([tensor[:, :k, :],
             tensor[:, k+1:, :]], dim=1) # size now (1,12,T)

        return tensor, torch.tensor(label), torch.tensor(int(speaker[3:]))

class BalancedBatchSampler(Sampler):
    """
    Class that will sample INDICES such that every batch must contain:
        speakers_per_batch; the number of unique speakers per batch
        min_bonafide; minimum bonafide speech clips per speaker (2)
        min_spoof; minimum spoofed speech clips per speaker (2)
    """
    def __init__(self, dataset: TripletAudioDataset, speakers_per_batch=4, min_bona=2, min_spoof=2):
        self.speaker_to_indices = defaultdict(lambda: {0: [], 1: []}) # {"Trump": {0: [dataset_index_X, dataset_index_Y],
                                                                        # 1: [dataset_index_A, dataset_index_B, dataset_index_C]}}
        for index, (_, label, speaker) in enumerate(dataset.items):
            self.speaker_to_indices[speaker.item()][label.item()].append(index)

        # keep only speakers with sufficient examples
        self.speakers = [s for s, d in self.speaker_to_indices.items()
                         if len(d[0]) >= min_spoof and len(d[1]) >= min_bona]

        self.speakers_per_batch = speakers_per_batch
        self.min_bonafide  = min_bona
        self.min_spoofed = min_spoof

    def __iter__(self):
        speaker_pool = self.speakers.copy()
        random.shuffle(speaker_pool) # Shuffle speakers

        # Pop off self.speakers_per_batch (int) per iteration
        while len(speaker_pool) >= self.speakers_per_batch:
            batch_indices = []
            selected_speakers = [speaker_pool.pop() for _ in range(self.speakers_per_batch)] # length is self.speakers_per_batch
            for speaker in selected_speakers:
                dictionary = self.speaker_to_indices[speaker]
                batch_indices += random.sample(dictionary[1], self.min_bonafide)
                batch_indices += random.sample(dictionary[0], self.min_spoofed)
            yield batch_indices # returns batch_indices to the DataLoader, pauses the function, and continues on the next iteration
            # Each yield therefore produces one batch

    def __len__(self):
        return len(self.speakers)
    
def batch_hard_triplet_loss(embedding: torch.Tensor, labels: torch.Tensor, speakers: torch.Tensor, margin: float=1.0) -> torch.Tensor:
    """
    Function that computes batch hard triplet loss, AFTER encoding from siamese_cnn.py

    Input:
        embedding; L_2 normalized embeddings; shape is (batch_size, embedding_dimensionality)
        labels; which utterances are bonafide (1) and which are spoof (0); shape is (batch_size,)
            Allows us to form "same-label" and "different-label" masks for positive/negative triplet selection
        speakers; identifies the speaker for each utterance; shape is (batch_size,)
            Allows us to enforce the "same-speaker" constraint for triplets
    
    Output:
        Scalar loss value as defined by the paper; torch.Tensor with shape of ()
    """
    # Compute distance matrix between our feature embeddings
    dist = torch.cdist(embedding, embedding)

    # Unsqueeze adds a dimension to the 1st index of shape => COLUMN VECTOR
    lbl_col = labels.unsqueeze(1) # shape becomes (B, 1)
    spk_col = speakers.unsqueeze(1) # shape becomes (B, 1)

    same_speaker = spk_col == spk_col.t() # (B, 1) vs (1, B) => (B, B) boolean matrix; (i,j) is True IFF utterances i and j same speaker
    same_label = lbl_col == lbl_col.t() # (B, B) boolean matrix also
    diff_label = ~same_label # Negate

    pos_mask = same_label & same_speaker # same speaker + SAME label
    neg_mask = diff_label & same_speaker # same speaker + DIFF label

    eye = torch.eye(dist.size(0), dtype=torch.bool, device=dist.device) # (B,B) identity-matrix mask where diagonals are True
    pos_mask = pos_mask & (~eye) # keep only off-diagonal positives; diagonals are self-distance

    # Hardest positive (max) / negative (min)
    d_pos = dist.clone()
    d_pos[~pos_mask] = -float('inf')
    d_neg = dist.clone()
    d_neg[~neg_mask] = float('inf')
    hardest_pos = d_pos.max(dim=1).values
    hardest_neg = d_neg.min(dim=1).values

    # Handle anchors lacking valid pos/neg
    hardest_pos[hardest_pos == -float('inf')] = 0
    hardest_neg[hardest_neg == float('inf')] = margin

    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss.mean()

####   BEFORE RUNNING THIS, REFER TO THE README FOR INSTRUCTIONS ON HOW TO DOWNLOAD THE DATA   ####
if __name__ == "__main__":

    # in case downloaded ASVspoof 2019
    folder = "LA"
    for root, dirs, files in os.walk(folder):
        for name in files + dirs:
            if "ASVspoof2019" in name:
                old_path = os.path.join(root, name)
                new_name = name.replace("ASVspoof2019", "ASVspoof2021")
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)

    #preprocess
    input_path = "data/ASVspoof2021_LA_eval/flac"
    output_path = "data/tensors"
    if os.path.isdir("data/ASVspoof2021_LA_cm_protocols/"):
        metadata_path = "data/ASVspoof2021_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
    else:
        metadata_path = "data/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt"
    mfcc_dir = "data/tensors"

    triplet_dataset = TripletAudioDataset(metadata_path, mfcc_dir)
    
    #for i in range(len(triplet_dataset)):
        #print(triplet_dataset[i])