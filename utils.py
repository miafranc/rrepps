import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
import codecs
import numpy as np


def one_hot_np(v, num_classes):
    return np.eye(num_classes)[v] * 2 - 1


def one_hot_torch(v, num_classes):
    return torch.eye(num_classes)[v] * 2 - 1


def dump(obj, filename):
    f = codecs.open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()


def load(filename):
    f = codecs.open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def get_train_validation(dataset, use_percent=1, val_split=0.20, stratify=True):
    """
    Split data into train and validation sets.
    Added use_percent parameter.
    """
    labels = None
    if stratify:
        if isinstance(dataset, torch.utils.data.Subset):
            labels = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            labels = np.array(dataset.targets)

    if use_percent == 1:
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, stratify=labels)
    else:
        idx, _              = train_test_split(list(range(len(dataset))), test_size=1-use_percent, stratify=labels)
        train_idx, val_idx  = train_test_split(idx, test_size=val_split, stratify=labels[idx])
    
    return (torch.utils.data.Subset(dataset, train_idx), 
            torch.utils.data.Subset(dataset, val_idx))


def calculate_mean_and_std(path, img_size):
    dataset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((img_size[0], img_size[1])),
                                                    transforms.ToTensor()
                                                ]))

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=1
    )

    psum = 0
    psum_sq = 0

    with tqdm(dataloader, unit='batch') as t:
        for data, label in t:
            psum += data.sum(axis=(0, 2, 3))
            psum_sq += (data**2).sum(axis=(0, 2, 3))

    norm = len(dataset) * img_size[0] * img_size[1]

    m = psum/norm
    s = torch.sqrt((psum_sq/norm) - (m**2))
    print(f'Mean: {m}')
    print(f' Std: {s}')
        

if __name__ == '__main__':
    calculate_mean_and_std('path_to_images', (224, 224))
