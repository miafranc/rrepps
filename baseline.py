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
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    densenet121, densenet161,
    DenseNet121_Weights, DenseNet161_Weights,
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)
from torchvision.models.feature_extraction import create_feature_extractor
import os
import argparse
import time
from PIL import Image
import matplotlib.pyplot as plt
from functools import reduce
from torchmetrics.classification import MulticlassHingeLoss

from image_dataset import ImageDatasetWithFilenames, corrupt_image

from utils import calculate_mean_and_std, get_train_validation, load, dump
import settings
from settings import *

from utils import set_seed, tb_writer, dump_parameters


name2model = {
    'resnet18':         (resnet18,      ResNet18_Weights.IMAGENET1K_V1,         'avgpool',      'fc',           'fc'),
    'resnet34':         (resnet34,      ResNet34_Weights.IMAGENET1K_V1,         'avgpool',      'fc',           'fc'),
    'resnet50':         (resnet50,      ResNet50_Weights.IMAGENET1K_V1,         'avgpool',      'fc',           'fc'),
    'densenet121':      (densenet121,   DenseNet121_Weights.IMAGENET1K_V1,      'features',     'classifier',   'classifier'),
    'densenet161':      (densenet161,   DenseNet161_Weights.IMAGENET1K_V1,      'features',     'classifier',   'classifier'),
    'convnext_tiny':    (convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1,    'classifier.1', 'classifier',   'classifier.2'),
}

DEVICE = f'cuda:{CUDA_ID}' if torch.cuda.is_available() else 'cpu'


def hinge_loss(input, target, p=2):
    """ 
    Hinge loss (squared by default).
    """
    y = F.one_hot(target, num_classes=input.shape[1]) * 2 - 1
    return (1/input.shape[0]) * torch.sum(torch.max(torch.zeros(input.shape).to(DEVICE), 1 - input * y) ** p)


def build_model(base_model_name, num_classes, bias=True):
    base_model_name_lowered = base_model_name.lower()
    name2model_values = name2model[base_model_name_lowered]
    model_fn, weight_enum = name2model_values[0], name2model_values[1]
    model = model_fn(weights=weight_enum)
    
    if base_model_name_lowered.startswith(('resnet',)):
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=bias)
    elif base_model_name_lowered.startswith(('mnasnet', 'vgg', 'convnext')):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes, bias=bias)
    elif base_model_name_lowered.startswith(('densenet',)):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes, bias=bias)
    else:
        raise Exception('Model not implemented!')

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
        target = label.to(DEVICE)

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        
        with grad_req:
            output = model(input)

            loss = criterion(output, target)

            if LOSS == 'hinge':
                loss *= LOSS_HINGE_C
                layer = reduce(getattr, name2model[MODEL_NAME][4].split('.'), model.module)
                weight_norm = layer.weight.norm(2)
                loss += 0.5 * weight_norm

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


def test(svm=False, finetuned=False):
    print('Testing the model...')
    model = build_model(MODEL_NAME, NUM_CLASSES, BASE_MODEL_BIAS)
    if not svm:
        model.load_state_dict(torch.load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_{BEST_MODEL}_{MODEL_NAME}.pth'), weights_only=True))
    else:
        if not finetuned:
            model.load_state_dict(torch.load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_{BEST_MODEL}_{MODEL_NAME}_SVM.pth'), weights_only=True))
        else:
            model.load_state_dict(torch.load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_{BEST_MODEL}_{MODEL_NAME}_SVM_finetuned.pth'), weights_only=True))
    model = model.to(DEVICE)
    model_multi = torch.nn.DataParallel(model)

    testset = ImageDatasetWithFilenames(os.path.join(DATA_PATH, DATASET_NAME, 'test'),
                                        transform=v2.Compose([
                                            v2.Resize((IMG_SIZE, IMG_SIZE)),
                                            v2.ToImage(), 
                                            v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
                                        ]),
                                        corruption=CORRUPTION)

    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=8
    )

    if LOSS == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif LOSS == 'hinge':
        criterion = lambda x, z: hinge_loss(x, z, p=2)
        # criterion = MulticlassHingeLoss(NUM_CLASSES, squared=True).to(DEVICE)

    model_multi.eval()
    test_acc, test_loss, predictions = train_or_test(model_multi, testloader, criterion, None)
    return test_acc, test_loss, predictions


def train():
    tb = tb_writer(TENSORBOARD_LOGDIR_PREFIX, DATASET_NAME + '_' + MODEL_NAME)
    dump_parameters(tb.get_logdir(), 'settings.json')

    model = build_model(MODEL_NAME, NUM_CLASSES, BASE_MODEL_BIAS)
    model = model.to(DEVICE)
    model_multi = torch.nn.DataParallel(model)

    if LOSS == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif LOSS == 'hinge':
        criterion = lambda x, z: hinge_loss(x, z, p=2)
        # criterion = MulticlassHingeLoss(NUM_CLASSES, squared=True).to(DEVICE)

    if OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(model_multi.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model_multi.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(model_multi.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    if SCHEDULER.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA)
    elif SCHEDULER.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif SCHEDULER.lower() == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=STEP_MULTI, gamma=STEP_GAMMA)

    augment = v2.Compose([
        v2.Resize(256),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomRotation(degrees=(-15, 15)),
        v2.RandomPerspective(distortion_scale=0.3),
        v2.RandomCrop(IMG_SIZE),
    ])

    trainset = ImageDatasetWithFilenames(os.path.join(DATA_PATH, DATASET_NAME, 'train'),
                                         transform=v2.Compose([
                                            augment,
                                            v2.ToImage(), 
                                            v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
                                         ]))
    
    trainset, validationset = get_train_validation(trainset, use_percent=DATA_PERCENTAGE, val_split=VAL_SPLIT, stratify=True)

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    validationloader = torch.utils.data.DataLoader(
        validationset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
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
        train_acc, train_loss, _ = train_or_test(model_multi, trainloader, criterion, optimizer)

        tb.add_scalar('Accuracy/train', train_acc, epoch)
        tb.add_scalar('Loss/train', train_loss, epoch)

        # Validation:
        print('Validation:')
        model_multi.eval()
        val_acc, val_loss, _ = train_or_test(model_multi, validationloader, criterion, None)

        tb.add_scalar('Accuracy/val', val_acc, epoch)
        tb.add_scalar('Loss/val', val_loss, epoch)
        tb.flush()

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
            if SAVE_BEST_MODEL and val_acc > SAVE_MIN_ACC:
                torch.save(model_multi.module.state_dict(), os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_acc_{MODEL_NAME}.pth'))
                print(f"New best model (Accuracy) saved! Accuracy: {best_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            if SAVE_BEST_MODEL and val_acc > SAVE_MIN_ACC:
                torch.save(model_multi.module.state_dict(), os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_loss_{MODEL_NAME}.pth'))
                print(f"New best model (Loss) saved! Loss: {best_loss:.6f}")

        scheduler.step()
    
    tb.close()


def train_last_layer_only(model):
    for p in model.parameters():
        p.requires_grad = False
    last_layer = reduce(getattr, name2model[MODEL_NAME][4].split('.'), model)
    for p in last_layer.parameters():
        p.requires_grad = True


def train_all_layers(model):
    for p in model.parameters():
        p.requires_grad = True


def train_except_last_layer(model):
    for p in model.parameters():
        p.requires_grad = True
    last_layer = reduce(getattr, name2model[MODEL_NAME][4].split('.'), model)
    for p in last_layer.parameters():
        p.requires_grad = False


def fine_tune():
    model = build_model(MODEL_NAME, NUM_CLASSES, BASE_MODEL_BIAS)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_{BEST_MODEL}_{MODEL_NAME}_SVM.pth'), weights_only=True))
    model = model.to(DEVICE)

    train_except_last_layer(model)

    model_multi = torch.nn.DataParallel(model)

    if LOSS == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif LOSS == 'hinge':
        criterion = lambda x, z: hinge_loss(x, z, p=2)
        # criterion = MulticlassHingeLoss(NUM_CLASSES, squared=True).to(DEVICE)

    if OPTIMIZER.lower() == 'adam':
        optimizer = torch.optim.Adam(model_multi.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model_multi.parameters(), lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(model_multi.parameters(), lr=LR_FINETUNE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    augment = v2.Compose([
        v2.Resize(256),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomRotation(degrees=(-15, 15)),
        v2.RandomPerspective(distortion_scale=0.3),
        v2.RandomCrop(IMG_SIZE),
    ])

    trainset = ImageDatasetWithFilenames(os.path.join(DATA_PATH, DATASET_NAME, 'train'),
                                         transform=v2.Compose([
                                            augment,
                                            v2.ToImage(), 
                                            v2.ToDtype(torch.float32, scale=True),
                                            v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
                                         ]))
    
    trainset, validationset = get_train_validation(trainset, use_percent=DATA_PERCENTAGE, val_split=VAL_SPLIT, stratify=True)

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )

    validationloader = torch.utils.data.DataLoader(
        validationset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # Early Stopping initialization:
    best_acc = 0.0
    best_loss = float('inf')
    epochs_since_improvement = 0

    # Training:
    print('Training the model...')
    for epoch in range(NUM_EPOCHS_FINETUNE):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')

        # Train:
        print('Training:')
        model_multi.train()
        train_acc, train_loss, _ = train_or_test(model_multi, trainloader, criterion, optimizer)

        # Validation:
        print('Validation:')
        model_multi.eval()
        val_acc, val_loss, _ = train_or_test(model_multi, validationloader, criterion, None)

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
            if SAVE_BEST_MODEL and val_acc > SAVE_MIN_ACC:
                torch.save(model_multi.module.state_dict(), os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_acc_{MODEL_NAME}_SVM_finetuned.pth'))
                print(f"New best model (Accuracy) saved! Accuracy: {best_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            if SAVE_BEST_MODEL and val_acc > SAVE_MIN_ACC:
                torch.save(model_multi.module.state_dict(), os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_loss_{MODEL_NAME}_SVM_finetuned.pth'))
                print(f"New best model (Loss) saved! Loss: {best_loss:.6f}")


if __name__ == '__main__':
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=False, action='store_true')
    parser.add_argument('--test', required=False, action='store_true')
    parser.add_argument('--svm', required=False, action='store_true')
    parser.add_argument('--finetuned', required=False, action='store_true')

    args = parser.parse_args()

    if args.train:
        if args.finetuned:
            fine_tune()
        else:
            train()
    elif args.test:
        if args.svm:
            if args.finetuned:
                test(svm=True, finetuned=True)
            else:
                test(svm=True, finetuned=False)
        else:
            test()
