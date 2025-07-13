import os
from PIL import Image
from torchvision.io import read_image

from torch.utils.data import Dataset


class ImageDatasetWithFilenames(Dataset):
    
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        subfolders = sorted(os.listdir(self.img_dir))
        self.map = {subfolders[i]:i for i in range(len(subfolders))}
        self.imap = {i:subfolders[i] for i in range(len(subfolders))}
        self.images = []
        self.targets = []
        for sf in subfolders:
            for f in sorted(os.listdir(os.path.join(self.img_dir, sf))):
                self.images.append((f, self.map[sf]))
                self.targets.append(self.map[sf])

    
    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imap[self.images[idx][1]], self.images[idx][0])
        image = Image.open(img_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = self.images[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label, self.images[idx][0]
