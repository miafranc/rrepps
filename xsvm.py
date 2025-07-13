import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import v2
import os
from tqdm import tqdm
from scipy.sparse import lil_matrix
import numpy as np
import time
from pprint import pprint
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import argparse

from settings import *
from utils import load, dump, get_train_validation, one_hot_np, one_hot_torch
from baseline import build_model, name2model
from image_dataset import ImageDatasetWithFilenames


DEVICE = f'cuda:{CUDA_ID}' if torch.cuda.is_available() else 'cpu'


def get_features(image, feature_extractor, batch_size=32):
    ds = torch.utils.data.TensorDataset(image)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    X = []
    for data in dl:
        x = feature_extractor(data[0])[name2model[MODEL_NAME][2]].detach()
        X.append(torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1))

    return torch.vstack(X)


def train_np(dataloader, feature_extractor, lamb):
    """
    Kernelized Pegasos algorithm.
    (Shalev-Shwartz, Shai, Yoram Singer, and Nathan Srebro. Pegasos: Primal estimated sub-gradient solver for SVM. Proceedings of the 24th ICML. 2007. https://home.ttic.edu/~nati/Publications/PegasosMPB.pdf;
    Some useful notes on Pegasos: https://karlstratos.com/notes/pegasos.pdf)
    """
    alpha = lil_matrix((len(dataloader.dataset), NUM_CLASSES), dtype=np.int32)
    print('Creating index...')
    index_map = {dataloader.dataset[i][2]:i for i in range(len(dataloader.dataset))} # filename -> index
    print('Done.')

    torch.no_grad()

    for i, data in enumerate(tqdm(dataloader, unit='batch')):
        start = time.time()

        (image, label, filename) = data
        image = image.to(DEVICE)

        X = get_features(image, feature_extractor)
        y = one_hot_np(label.cpu().numpy(), num_classes=NUM_CLASSES)

        X = X.cpu().numpy()
        idx = np.array([index_map[ind] for ind in filename])

        X = np.insert(X, X.shape[1], 1, axis=1)
        eta = 1. / (lamb*(i+1)*image.shape[0])

        for c in range(NUM_CLASSES):
            nz_ind = alpha[:, c].nonzero()[0]
            if len(nz_ind) == 0:
                v = np.zeros((1, BATCH_SIZE_XSVM))
            else:
                Xi = torch.vstack([dataloader.dataset[nzi][0].to(DEVICE).unsqueeze(0) for nzi in nz_ind])
                X_nz = get_features(Xi, feature_extractor).cpu().numpy()
                X_nz = np.insert(X_nz, X_nz.shape[1], 1, axis=1)
                y_nz = one_hot_np(np.array([dataloader.dataset[nzi][1] for nzi in nz_ind]), num_classes=NUM_CLASSES)
                v = ((alpha[nz_ind, c].multiply(np.expand_dims(y_nz[:, c], axis=1))).T @ X_nz) @ X.T
            if len(idx) == 1:
                alpha[idx[0], c] = alpha[idx[0], c] + (((eta * np.multiply(np.expand_dims(y[:, c], axis=1), v.T)) < 1)*1)[0,0]
            else:
                alpha[idx, c] = alpha[idx, c] + ((eta * np.multiply(np.expand_dims(y[:, c], axis=1), v.T)) < 1)*1

        end = time.time()
        print(f'Time: {end-start}')

    W = []
    for c in range(NUM_CLASSES):
        nz_ind = alpha[:, c].nonzero()[0]
        Xi = torch.vstack([dataloader.dataset[nzi][0].to(DEVICE).unsqueeze(0) for nzi in nz_ind])
        X_nz = get_features(Xi, feature_extractor).cpu().numpy()
        X_nz = np.insert(X_nz, X_nz.shape[1], 1, axis=1)
        y_nz = one_hot_np(np.array([dataloader.dataset[nzi][1] for nzi in nz_ind]), num_classes=NUM_CLASSES)

        W.append((1. / (lamb * (i+1) * BATCH_SIZE_XSVM)) * (X_nz.T @ alpha[nz_ind, c].multiply(np.expand_dims(y_nz[:, c], axis=1))))

    return np.hstack(W), alpha, index_map


def train_torch(dataloader, feature_extractor, lamb):
    """
    Kernelized Pegasos algorithm.
    """
    alpha = lil_matrix((len(dataloader.dataset), NUM_CLASSES), dtype=np.int32)
    print('Creating index...')
    index_map = {dataloader.dataset[i][2]:i for i in range(len(dataloader.dataset))}
    print('Done.')

    torch.no_grad()

    for i, data in enumerate(tqdm(dataloader, unit='batch')):
        start = time.time()

        (image, label, filename) = data
        image = image.to(DEVICE)

        X = get_features(image, feature_extractor) # b x d
        y = one_hot_torch(label, num_classes=NUM_CLASSES).to(DEVICE)

        idx = torch.tensor([index_map[ind] for ind in filename])

        X = torch.cat((X, torch.ones((X.shape[0], 1)).to(DEVICE)), dim=1)
        eta = 1. / (lamb*(i+1)*image.shape[0])

        for c in range(NUM_CLASSES):
            nz_ind = alpha[:, c].nonzero()[0]
            if len(nz_ind) == 0:
                v = torch.zeros((1, BATCH_SIZE_XSVM)).to(DEVICE)
            else:
                Xi = torch.vstack([dataloader.dataset[nzi][0].to(DEVICE).unsqueeze(0) for nzi in nz_ind])
                X_nz = get_features(Xi, feature_extractor)
                X_nz = torch.cat((X_nz, torch.ones((X_nz.shape[0], 1)).to(DEVICE)), dim=1)
                y_nz = one_hot_torch(torch.tensor([dataloader.dataset[nzi][1] for nzi in nz_ind]), num_classes=NUM_CLASSES).to(DEVICE)
                v = torch.matmul(torch.matmul((torch.tensor(alpha[nz_ind, c].todense()).to(DEVICE) * y_nz[:, c].unsqueeze(dim=1)).T, X_nz), X.T)
            if len(idx) == 1:
                alpha[idx[0], c] = alpha[idx[0], c] + (((eta * (y[:, c].unsqueeze(dim=1) * v.T)) < 1)*1).cpu().numpy()[0,0]
            else:
                alpha[idx, c] = alpha[idx, c] + (((eta * (y[:, c].unsqueeze(dim=1) * v.T)) < 1)*1).cpu().numpy()

        end = time.time()
        print(f'Time: {end-start}')

    W = []
    for c in range(NUM_CLASSES):
        nz_ind = alpha[:, c].nonzero()[0]
        Xi = torch.vstack([dataloader.dataset[nzi][0].to(DEVICE).unsqueeze(0) for nzi in nz_ind])
        X_nz = get_features(Xi, feature_extractor)
        X_nz = torch.cat((X_nz, torch.ones((X_nz.shape[0], 1)).to(DEVICE)), dim=1)
        y_nz = one_hot_torch(np.array([dataloader.dataset[nzi][1] for nzi in nz_ind]), num_classes=NUM_CLASSES).to(DEVICE)

        W.append((1. / (lamb * (i+1) * BATCH_SIZE_XSVM)) * torch.matmul(X_nz.T, torch.tensor(alpha[nz_ind, c].todense()).to(DEVICE) * y_nz[:, c].unsqueeze(dim=1)))

    return torch.hstack(W), alpha, index_map


def train_per_class(dataloader, feature_extractor, lamb, start_class=0):
    """
    Kernelized Pegasos algorithm.
    """
    print('Creating index...')
    if not os.path.exists(f'models/xsvm_indexmap_{MODEL_NAME}'):
        index_map = {dataloader.dataset[i][2]:i for i in range(len(dataloader.dataset))}
        dump(index_map, f'models/xsvm_indexmap_{MODEL_NAME}')
    else:
        index_map = load(f'models/xsvm_indexmap_{MODEL_NAME}')
    print('Done.')

    torch.no_grad()

    W = []
    alpha = lil_matrix((len(dataloader.dataset), NUM_CLASSES), dtype=np.int32)

    for c in range(start_class, NUM_CLASSES):
        print(f'Class {c}:')
        start = time.time()

        for i, data in enumerate(tqdm(dataloader, unit='batch')):

            (image, label, filename) = data
            image = image.to(DEVICE)

            X = get_features(image, feature_extractor) # b x d
            y = one_hot_torch(label, num_classes=NUM_CLASSES).to(DEVICE)

            idx = torch.tensor([index_map[ind] for ind in filename])

            X = torch.cat((X, torch.ones((X.shape[0], 1)).to(DEVICE)), dim=1)
            eta = 1. / (lamb*(i+1)*image.shape[0])

            nz_ind = alpha[:, c].nonzero()[0]
            if len(nz_ind) == 0:
                v = torch.zeros((1, BATCH_SIZE_XSVM)).to(DEVICE)
            else:
                Xi = torch.vstack([dataloader.dataset[nzi][0].to(DEVICE).unsqueeze(0) for nzi in nz_ind])
                X_nz = get_features(Xi, feature_extractor)
                X_nz = torch.cat((X_nz, torch.ones((X_nz.shape[0], 1)).to(DEVICE)), dim=1)
                y_nz = one_hot_torch(torch.tensor([dataloader.dataset[nzi][1] for nzi in nz_ind]), num_classes=NUM_CLASSES).to(DEVICE)
                v = torch.matmul(torch.matmul((torch.tensor(alpha[nz_ind, c].todense()).to(DEVICE) * y_nz[:, c].unsqueeze(dim=1)).T, X_nz), X.T)
            if len(idx) == 1:
                alpha[idx[0], c] = alpha[idx[0], c] + (((eta * (y[:, c].unsqueeze(dim=1) * v.T)) < 1)*1).cpu().numpy()[0,0]
            else:
                alpha[idx, c] = alpha[idx, c] + (((eta * (y[:, c].unsqueeze(dim=1) * v.T)) < 1)*1).cpu().numpy()

        end = time.time()
        print(f'Time: {end-start}')
        
        nz_ind = alpha[:, c].nonzero()[0]
        Xi = torch.vstack([dataloader.dataset[nzi][0].to(DEVICE).unsqueeze(0) for nzi in nz_ind])
        X_nz = get_features(Xi, feature_extractor)
        X_nz = torch.cat((X_nz, torch.ones((X_nz.shape[0], 1)).to(DEVICE)), dim=1)
        y_nz = one_hot_torch(np.array([dataloader.dataset[nzi][1] for nzi in nz_ind]), num_classes=NUM_CLASSES).to(DEVICE)

        W.append((1. / (lamb * (i+1) * BATCH_SIZE_XSVM)) * torch.matmul(X_nz.T, torch.tensor(alpha[nz_ind, c].todense()).to(DEVICE) * y_nz[:, c].unsqueeze(dim=1)))

        dump([W[-1], alpha[:, c]], f'models/xsvm_w_{c}_{MODEL_NAME}')

    return torch.hstack(W), alpha, index_map


def test_np(dataloader, feature_extractor, W):
    n_examples = 0
    n_correct = 0

    for i, data in enumerate(tqdm(dataloader, unit='batch')):
        (image, label, filename) = data
        image = image.to(DEVICE)

        X = get_features(image, feature_extractor).cpu().numpy()
        X = np.insert(X, X.shape[1], 1, axis=1)

        pred = np.argmax(X @ W, axis=1)
        n_examples += label.size(0)
        n_correct += (pred == label.cpu().numpy()).sum()

    acc = n_correct / n_examples

    print('\tacc:\t{0}%'.format(acc))
    return acc


def test_torch(dataloader, feature_extractor, W):
    n_examples = 0
    n_correct = 0

    for i, data in enumerate(tqdm(dataloader, unit='batch')):
        (image, label, filename) = data
        image = image.to(DEVICE)
        label = label.to(DEVICE)

        X = get_features(image, feature_extractor) # b x d
        X = torch.cat((X, torch.ones((X.shape[0], 1)).to(DEVICE)), dim=1)

        pred = torch.argmax(torch.matmul(X, W), axis=1)
        n_examples += label.size(0)
        n_correct += (pred == label).sum()

    acc = n_correct / n_examples

    print('\tacc:\t{0}%'.format(acc))
    return acc.cpu().numpy()


def train_test_xsvm(is_train=True, id=0):
    model = build_model(MODEL_NAME, NUM_CLASSES)
    model.load_state_dict(torch.load(f'models/model_best_{BEST_MODEL}_{MODEL_NAME}.pth', weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    if os.path.exists(TRAINSET):
        trainset = load(TRAINSET)
        validationset = load(VALSET)
    else:
        if AUGMENT_XSVM:
            augment = v2.Compose([
                v2.Resize(256),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomRotation(degrees=(-15, 15)),
                v2.RandomPerspective(distortion_scale=0.3),
                v2.RandomCrop(224),
            ])
        else:
            augment = v2.Compose([
                v2.Resize((224, 224)),
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
    
    sampler = torch.utils.data.RandomSampler(
        trainset, 
        replacement=True,
        num_samples=int(NUM_SAMPLES_COEF*len(trainset)),
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=BATCH_SIZE_XSVM, 
        # shuffle=True, 
        num_workers=1,
        sampler=sampler
    )

    validationloader = torch.utils.data.DataLoader(
        validationset, 
        batch_size=BATCH_SIZE_XSVM, 
        shuffle=False, 
        num_workers=1
    )

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
        batch_size=BATCH_SIZE_XSVM, 
        shuffle=False, 
        num_workers=1
    )

    feature_extractor = create_feature_extractor(model, return_nodes=[name2model[MODEL_NAME][2]])

    vacc = 0
    acc = 0
    avg_sv_num = 0
    nsv_pos_neg = 0

    if is_train:
        W, alpha, index_map = train_torch(trainloader, feature_extractor, lamb=1)
        
        dump([W, alpha, index_map], f'models/xsvm_{MODEL_NAME}_{id}.pickle')

        test_torch(validationloader, feature_extractor, W)
        acc = test_torch(testloader, feature_extractor, W)
    else:
        print(f'ID={id}:')
        W, alpha, index_map = load(f'models/xsvm_{MODEL_NAME}_{id}.pickle')
        vacc = test_torch(validationloader, feature_extractor, W)
        acc = test_torch(testloader, feature_extractor, W)
        avg_sv_num = alpha.count_nonzero() / NUM_CLASSES
        for c in range(NUM_CLASSES):
            nz_ind = alpha[:, c].nonzero()[0]
            y_nz = one_hot_torch(torch.tensor([trainloader.dataset[nzi][1] for nzi in nz_ind]), num_classes=NUM_CLASSES)
            a_y = torch.tensor(alpha[nz_ind, c].todense()) * y_nz[:, c].unsqueeze(dim=1)
            nsv_pos_neg += ((a_y > 0) * 1).sum() / alpha[:, c].count_nonzero()
            print(f'{c}: {((a_y > 0) * 1).sum(), alpha[:, c].count_nonzero()}')

    nsv_pos_neg /= NUM_CLASSES

    return acc, vacc, avg_sv_num, nsv_pos_neg


def test_and_visualize(image_path, id=0):
    model = build_model(MODEL_NAME, NUM_CLASSES)
    model.load_state_dict(torch.load(f'models/model_best_{BEST_MODEL}_{MODEL_NAME}.pth', weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    trainset = load(TRAINSET)

    feature_extractor = create_feature_extractor(model, return_nodes=[name2model[MODEL_NAME][2]])

    image = Image.open(image_path)
    image = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)
    image = transforms.Resize((224, 224))(image)
    image = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)(image)
    
    X = get_features(image, feature_extractor)
    X = torch.cat((X, torch.ones((X.shape[0], 1)).to(DEVICE)), dim=1)

    W, alpha, index_map = load(f'models/xsvm_{MODEL_NAME}_{id}.pickle')

    pred = torch.argmax(torch.matmul(X, W), axis=1).cpu().numpy()

    inv_index_map = {ind:filename for (filename, ind) in index_map.items()}
    label_map = dict(trainset.dataset.images)
    folder_class_map = trainset.dataset.map
    folder_class_map_inv = {v:k for k, v in folder_class_map.items()}

    nonzeros = []
    data_info = []
    for i in alpha[:, pred].nonzero()[0]:
        filename = inv_index_map[i]
        label = label_map[filename]
        class_folder = folder_class_map_inv[label]
        img_path = os.path.join(trainset.dataset.img_dir, class_folder, filename)
        img = Image.open(img_path)
        img = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)
        img = transforms.Resize((224, 224))(img)
        img = transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)(img)
        nonzeros.append(img)
        data_info.append([filename, label, class_folder, img_path, alpha[i, pred]])

    X2 = get_features(torch.vstack(nonzeros), feature_extractor)
    X2 = torch.cat((X2, torch.ones((X2.shape[0], 1)).to(DEVICE)), dim=1)

    a = alpha[alpha[:, pred].nonzero()[0], pred].todense()
    sim = X.matmul(X2.T).cpu().numpy()
    a_sim = np.squeeze(np.asarray(np.multiply(sim, a)))
    labels = np.array([d[1] for d in data_info])
    pos_ind = np.argwhere(labels == pred).squeeze(1)
    neg_ind = np.argwhere(labels != pred).squeeze(1)
    largest_pos_ind = pos_ind[np.flip(np.argsort(a_sim[pos_ind]))]
    largest_neg_ind = neg_ind[np.flip(np.argsort(a_sim[neg_ind]))]
    
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        img = np.array(Image.open(data_info[largest_pos_ind[i]][3]).convert('RGB'))
        figure.add_subplot(rows, cols, i)
        plt.axis('off')
        plt.title(f'{data_info[largest_pos_ind[i]][2]}\n{a_sim[largest_pos_ind[i]]:.4f}')
        plt.imshow(img.squeeze())
    
    plt.axis('off')
    plt.show()

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        img = np.array(Image.open(data_info[largest_neg_ind[i]][3]).convert('RGB'))
        figure.add_subplot(rows, cols, i)
        plt.axis('off')
        plt.title(f'{data_info[largest_neg_ind[i]][2]}\n{a_sim[largest_neg_ind[i]]:.4f}')
        plt.imshow(img.squeeze())
    
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    print(f'Device = {DEVICE}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=False, action='store_true')
    parser.add_argument('--test', required=False, action='store_true')
    parser.add_argument('--vis', required=False)

    args = parser.parse_args()

    if args.train:
        train_test_xsvm(True, 0)
    elif args.test:
        a, va, n, pn = train_test_xsvm(False, 0)
        print(f'Accuracy: {a}')
        print(f'Validation accuracy: {va}')
        print(f'Number of support vectors: {n}')
        print(f'Ratio of pos. and neg. SVs: {pn}')
    elif args.vis:
        test_and_visualize(args.vis)
