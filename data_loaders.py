from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from utils import get_default_device, DeviceDataLoader

# Loading the data from the folder
data_dir = './lfw'
IMG_SIZE = 64
BATCH_SIZE = 96


def get_data_loaders(data_directory_path, img_size, batch_size):
    dataset = ImageFolder(data_directory_path, transforms.Compose(
        {
            transforms.Resize(img_size),
            transforms.ToTensor()
        }))

    # Splitting the data into train, validation and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size  # 0.05 * len(triplet_dataset)

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    # Creating data loaders
    train_loader = DataLoader(train_ds,
                              batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    val_loader = DataLoader(val_ds,
                            batch_size,
                            num_workers=4,
                            pin_memory=True)

    test_loader = DataLoader(test_ds,
                             batch_size,
                             num_workers=4,
                             pin_memory=True)

    # Moving the data to the GPU
    device = get_default_device()
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    number_of_classes = len(dataset.classes)

    return train_loader, val_loader, test_loader, number_of_classes
