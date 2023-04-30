import os
import glob
import random
import time
import numpy as np
import wandb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn

import utils
from data_loaders import get_data_loaders
from utils_torch_model import copmute_accuracy, validation_step, \
    validation_epoch_end, epoch_end, evaluate_torch_model

# Parameters
data_dir = './lfw'
IMG_SIZE = 64
BATCH_SIZE = 96
l_rate = 1e-3


train_loader, val_loader, test_loader, num_classes = get_data_loaders(data_dir, IMG_SIZE, BATCH_SIZE)
device = utils.get_default_device()


def fit_torch_model(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            with torch.cuda.amp.autocast():
                images, labels = batch
                out = model(images)  # Generate predictions
                loss = F.cross_entropy(out, labels)  # Calculate loss

            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # log metrics to wandb
        wandb.log({"loss": loss, "epoch": epoch})

        # Validation phase
        result = evaluate_torch_model(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(epoch, result)
        history.append(result)

    return history


# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="PRinAI",
#
#     # track hyper-parameters and run metadata
#     config={
#         "learning_rate": l_rate,
#         "architecture": "ResNet-50",
#         "triplet_dataset": "LFWDataset",
#         "epochs": 20,
#         "optimizer": "Adam",
#         "batch_size": BATCH_SIZE
#     }
# )


model_torch = resnet50(weights=ResNet50_Weights.DEFAULT)
model_torch.fc = nn.Linear(model_torch.fc.in_features, num_classes)
model_torch = utils.to_device(model_torch, device)

start_time = time.perf_counter()
history = fit_torch_model(epochs=10, lr=l_rate,
                          model=model_torch,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          opt_func=torch.optim.Adam)

# wandb.finish()

print('\nTraining time: ', time.perf_counter() - start_time)
