import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from facenet_pytorch import MTCNN


##################
# Data Utilities #
##################
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break


def get_class_distribution(dataset):
    """Get the distribution of classes in the triplet_dataset"""
    label_counts = pd.Series(dataset.targets).value_counts().sort_index()
    dist_class = {label: count for label, count in zip(dataset.classes, label_counts)}

    return dist_class


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Utility function
def create_csv_file(data_dir):
    """
    Load all the paths in the lfw_cropped folder and create a dataframe with the paths and labels

    Labels: names of the folders in the lfw_cropped folder
    Paths: paths to the images in the lfw_cropped folder
    Dataframe: saved as a csv file
    """
    if not os.path.exists("lfw_cropped_annots.csv"):
        paths = []
        labels = []
        for folder in os.listdir(data_dir):
            for file in os.listdir(os.path.join(data_dir, folder)):
                paths.append(os.path.join(data_dir, folder, file))
                labels.append(folder)

        df = pd.DataFrame({"path": paths,
                           "label": labels})

        # Drop all elements with only one occurrence
        df = df.groupby("label").filter(lambda x: len(x) > 1)

        df.to_csv("lfw_cropped_annots.csv", index=False)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for batch in self.dataloader:
            yield to_device(batch, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)


#################
# Face Detector #
#################
# The following cell are similar to the facenet-pytorch examples [here](https://github.com/timesler/facenet-pytorch)
def create_detected_faces_folder(data_dir, batch_size, device):
    # Define the data loader for the input set of images
    orig_img_ds = ImageFolder(data_dir, transform=None)

    # overwrites class labels in triplet_dataset with path so path can be used for saving output in mtcnn batches
    orig_img_ds.samples = [
        (p, p)
        for p, _ in orig_img_ds.samples
    ]

    def collate_pil(x):
        out_x, out_y = [], []
        for xx, yy in x:
            out_x.append(xx)
            out_y.append(yy)
        return out_x, out_y

    loader = DataLoader(
        orig_img_ds,
        num_workers=4,
        batch_size=batch_size,
        collate_fn=collate_pil
    )

    # Add the face detector model as a transform to the triplet_dataset
    detector = MTCNN(
        image_size=160,
        margin=14,
        device=device,
        selection_method='center_weighted_size'
    )

    crop_paths = []
    box_probs = []
    # Create a directory to save the cropped images
    for i, (x, b_paths) in enumerate(loader):
        crops = [p.replace(data_dir, data_dir + '_cropped') for p in b_paths]
        detector(x, save_path=crops)
        crop_paths.extend(crops)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

    # We don't need the detector anymore, so we can delete it to free up memory
    del detector
    torch.cuda.empty_cache()


###################
# Model Utilities #
###################
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        """Calculate loss for a batch of training data"""
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        """Calculate loss and accuracy for a batch of validation data"""
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(preds, labels):
    """Calculate accuracy"""
    _, preds = torch.max(preds, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    

def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
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
                loss = model.training_step(batch)
                
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


####################
# Plotting Utility #
####################
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
