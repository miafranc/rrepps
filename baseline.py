import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pprint import pprint
import numpy as np
from torchvision.transforms import v2
from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
)
import os
import argparse

from image_dataset import ImageDatasetWithFilenames

from utils import calculate_mean_and_std, get_train_validation, load, dump
from settings import *


name2model = {
    'resnet18': (resnet18,  
                 ResNet18_Weights.DEFAULT,
                 'layer4'),
    'resnet34': (resnet34,  
                 ResNet34_Weights.DEFAULT,
                 'layer4'),
    'resnet50': (resnet50,  
                 ResNet50_Weights.DEFAULT,
                 'layer4'),
 }

DEVICE = f'cuda:{CUDA_ID}' if torch.cuda.is_available() else 'cpu'


def build_model(base_model_name, num_classes):
    base_model_name_lowered = base_model_name.lower()
    model_fn, weight_enum, _ = name2model[base_model_name_lowered]
    model = model_fn(weights=weight_enum)
    
    if base_model_name_lowered.startswith(('resnet',)):
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=False)
    if base_model_name_lowered.startswith(('mnasnet', 'vgg')):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes, bias=False)

    return model


def train_or_test(model, dataloader, criterion, optimizer=None):
    is_train = optimizer is not None
    
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_loss = 0

    predictions = []

    loss = 0
    total_loss = 0

    for i, data in enumerate(tqdm(dataloader, unit='batch')):
        (image, label, filename) = data

        input = image.to(DEVICE)
        target = label.long().to(DEVICE)

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        
        with grad_req:
            output = model(input)

            loss = criterion(output, target)

            predicted = torch.argmax(output.data, dim=1)
            if not is_train:
                predictions.extend(predicted.cpu().numpy())
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_loss += loss.item()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted

    acc = n_correct / n_examples
    loss = total_loss / n_batches

    print(f'\tacc: \t\t{acc}')
    print(f'\tloss: \t\t{loss}')

    return acc, loss, predictions


def test():
    print('Testing the model...')
    model = build_model(MODEL_NAME, NUM_CLASSES)
    model.load_state_dict(torch.load(f'models/model_best_{BEST_MODEL}_{MODEL_NAME}.pth', weights_only=True))
    model = model.to(DEVICE)
    model_multi = torch.nn.DataParallel(model)

    testset = ImageDatasetWithFilenames(DATA_PATH_TEST,
                                        transform=v2.Compose([
                                            v2.Resize(256),
                                            v2.CenterCrop(224),
                                            v2.ToImage(), 
                                            v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
                                        ]))

    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=1
    )

    criterion = torch.nn.CrossEntropyLoss()

    model_multi.eval()
    test_acc, test_loss, predictions = train_or_test(model_multi, testloader, criterion, None)


def train():
    model = build_model(MODEL_NAME, NUM_CLASSES)
    model = model.to(DEVICE)
    model_multi = torch.nn.DataParallel(model)

    criterion = torch.nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(model_multi.parameters(), lr=LR)
    optimizer = torch.optim.SGD(model_multi.parameters(), lr=LR, momentum=MOMENTUM)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if os.path.exists(TRAINSET):
        trainset = load(TRAINSET)
        validationset = load(VALSET)
    else:
        augment = v2.Compose([
            v2.Resize(256),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomRotation(degrees=(-15, 15)),
            v2.RandomPerspective(distortion_scale=0.3),
            v2.RandomCrop(224),
        ])
        trainset = ImageDatasetWithFilenames(DATA_PATH_TRAIN,
                                             transform=v2.Compose([
                                                augment,
                                                v2.ToImage(), 
                                                v2.ToDtype(torch.float32, scale=True),
                                                v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
                                             ]))
        
        trainset, validationset = get_train_validation(trainset, use_percent=DATA_PERCENTAGE, val_split=0.2, stratify=True)

        dump(trainset, TRAINSET)
        dump(validationset, VALSET)

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=1
    )

    validationloader = torch.utils.data.DataLoader(
        validationset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=1
    )

    # Early Stopping initialization:
    best_acc = 0.0
    best_loss = float('inf')
    epochs_since_improvement = 0

    # Training:
    print('Training the model...')
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print(f'LR={scheduler.get_last_lr()}')
        
        # Train:
        print('Training:')
        model_multi.train()
        train_or_test(model_multi, trainloader, criterion, optimizer)

        # Validation:
        print('Validation:')
        model_multi.eval() 
        val_acc, val_loss, predictions = train_or_test(model_multi, validationloader, criterion, None)

        if val_acc <= best_acc and val_loss >= best_loss:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epochs.")
        else:
            epochs_since_improvement = 0

        if epochs_since_improvement >= PATIENCE:
            print(f"Early stopping after {epoch+1} epochs.")
            break

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_multi.module.state_dict(), f'models/model_best_acc_{MODEL_NAME}.pth')
            print(f"New best model (Accuracy) saved! Accuracy: {best_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model_multi.module.state_dict(), f'models/model_best_loss_{MODEL_NAME}.pth')
            print(f"New best model (Loss) saved! Loss: {best_loss:.6f}")

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=False, action='store_true')
    parser.add_argument('--test', required=False, action='store_true')

    args = parser.parse_args()

    if args.train:
        train()
    elif args.test:
        test()
