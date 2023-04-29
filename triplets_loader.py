import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.nn.modules.distance import PairwiseDistance

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import config


# Reference: https://github.com/kvsnoufal/Pytorch-FaceNet-DogDataset/blob/master/src/dataset.py
class TripletsDataset(Dataset):
    """
    The goal of the custom triplet_dataset is to create a triplet_dataset that returns a triplet of images
    composed of an anchor image, a positive image and a negative image, to be used with a
    face recognition model and triplet model.
    """

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

    def __len__(self):
        return len(self.paths)

    def class_to_idx(self):
        return self.class_to_idx

    def __getitem__(self, idx):
        anchor_file = self.paths[idx]
        anchor_label = self.labels[idx]

        positive_idx = np.argwhere((self.labels == anchor_label) & (self.paths != anchor_file))
        positives = self.paths[positive_idx].flatten()
        positive = np.random.choice(positives)

        negatives_idx = np.argwhere(self.labels != anchor_label)
        negatives = self.paths[negatives_idx].flatten()
        negative = np.random.choice(negatives)
        anchors = np.array(Image.open(anchor_file))
        positives = np.array(Image.open(positive))
        negatives = np.array(Image.open(negative))

        anchors = np.transpose(anchors, (2, 0, 1)) / 255.0
        positives = np.transpose(positives, (2, 0, 1)) / 255.0
        negatives = np.transpose(negatives, (2, 0, 1)) / 255.0

        return {"anchor": torch.tensor(anchors,
                                       dtype=torch.float),
                "positive": torch.tensor(positives,
                                         dtype=torch.float),
                "negative": torch.tensor(negatives,
                                         dtype=torch.float)}


class TripletLoss(torch.nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)
        return loss
