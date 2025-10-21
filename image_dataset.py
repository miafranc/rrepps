import os
from PIL import Image
from PIL import ImageFilter, ImageEnhance
from torchvision.io import read_image
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from settings import *


def corrupt_image(image, corruption):
    if corruption == 'gaussian_blur':
        image = image.filter(ImageFilter.GaussianBlur(radius=CORRUPTION_BLUR_RADIUS))
    elif corruption == 'gaussian_noise':
        noise = Image.effect_noise((image.size[0], image.size[1]), CORRUPTION_NOISE_STD).convert('RGB') # (mean = 128)
        image = Image.blend(image.convert('RGB'), noise, CORRUPTION_NOISE_BLEND)
    elif corruption == 'contrast':
        e = ImageEnhance.Contrast(image)
        image = e.enhance(CORRUPTION_CONTRAST) # 1.5 = 50% more contrast
    elif corruption == 'brightness':
        e = ImageEnhance.Brightness(image)
        image = e.enhance(CORRUPTION_BRIGHTNESS) # 1.2 = 20% brighter
    elif corruption == 'pixelate':
        orig_size = image.size
        image = image.resize((orig_size[0] // CORRUPTION_PIXELATE_LEVEL, orig_size[1] // CORRUPTION_PIXELATE_LEVEL), resample=0)
        image = image.resize(orig_size, resample=0)

    return image


class ImageDatasetWithFilenames(Dataset):
    
    def __init__(self, img_dir, transform=None, target_transform=None, corruption=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        subfolders = sorted(os.listdir(self.img_dir))
        self.map = {subfolders[i]:i for i in range(len(subfolders))}
        self.imap = {i:subfolders[i] for i in range(len(subfolders))}
        self.images = []
        self.targets = []
        self.corruption = corruption
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

        image = corrupt_image(image, self.corruption)

        label = self.images[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label, self.images[idx][0]
