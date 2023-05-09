import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd



class ContrastiveDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with paths.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        annotations = pd.read_csv(csv_file)
        self.paths = annotations["path"].values
        self.labels = annotations["label"].values
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(set(self.labels))}

    def class_to_idx(self):
        return self.class_to_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        anchor_file = self.paths[idx]
        anchor_label = self.labels[idx]

        positive_idx = np.argwhere((self.labels == anchor_label) & (self.paths != anchor_file))
        positives = self.paths[positive_idx].flatten()
        positive = np.random.choice(positives)

        negatives_idx = np.argwhere(self.labels != anchor_label)
        negatives = self.paths[negatives_idx].flatten()
        negative = np.random.choice(negatives)
        
        anc_pos_neg_tensors = []
        for file in [anchor_file, positive, negative]:
            img = Image.open(file)
            img_arr = np.array(img)
            img_arr = np.transpose(img_arr, (2, 0, 1)) / 255.0
            img_tensor = torch.tensor(img_arr, dtype=torch.float)
            anc_pos_neg_tensors.append(img_tensor)
        
        if np.random.random() > 0.5:
            # Return positive pairs with label 0
            return anc_pos_neg_tensors[0], anc_pos_neg_tensors[1], 0
        else:
            # Return negative pairs with label 1
            return anc_pos_neg_tensors[0], anc_pos_neg_tensors[2], 1


class ContrastiveLoss(torch.nn.Module):
    """
    Reference: 
    https://jamesmccaffrey.wordpress.com/2022/03/04/contrastive-loss-function-in-pytorch/
    """
    def __init__(self, m=2.0):
        super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
        self.m = m  # margin 

    def forward_old(self, y1, y2, d=0):
        # d = 0 means y1 and y2 are supposed to be same
        # d = 1 means y1 and y2 are supposed to be different
        
        euc_dist = F.pairwise_distance(y1, y2)

        if d == 0:
            return torch.mean(torch.pow(euc_dist, 2))  # distance squared
        else:  # d == 1
            delta = self.m - euc_dist  # sort of reverse distance
            delta = torch.clamp(delta, min=0.0, max=None)
            return torch.mean(torch.pow(delta, 2))  # mean over all rows
        
    def forward(self, y1, y2, d=0):
        euc_dist = F.pairwise_distance(y1, y2)
        d_zero_mask = torch.eq(d, 0)
        d_one_mask = torch.eq(d, 1)

        d_zero_mean = torch.mean(torch.pow(euc_dist[d_zero_mask], 2))
        delta = self.m - euc_dist[d_one_mask]
        delta = torch.clamp(delta, min=0.0, max=None)
        d_one_mean = torch.mean(torch.pow(delta, 2))

        losses_batch = torch.where(d_zero_mask, d_zero_mean, d_one_mean)
        return torch.mean(losses_batch)  
        

