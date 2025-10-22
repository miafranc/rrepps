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
import codecs
import pickle
import cv2
from functools import reduce

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from scipy.linalg import norm

from settings import *
from utils import load, dump, dump_json, load_json, get_train_validation, one_hot_np, one_hot_torch
from baseline import build_model, name2model
from image_dataset import ImageDatasetWithFilenames


DEVICE = f'cuda:{CUDA_ID}' if torch.cuda.is_available() else 'cpu'


def extract_features(image, feature_extractor, batch_size=32):
    '''
    Extracts features from the neural network and returns as tensor,
    in which every row is a feature vector.
    '''
    ds = torch.utils.data.TensorDataset(image)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    X = []
    for data in dl:
        x = feature_extractor(data[0])[name2model[MODEL_NAME][2]].detach()
        # print(x.shape)
        if MODEL_NAME.startswith('densenet'):
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        else:
            x = torch.flatten(x, 1)
        X.append(x)

    return torch.vstack(X)


def generate_features(dataset_type='train'):
    '''
    Extracts and saves the features for the given dataset.
    '''
    model = build_model(MODEL_NAME, NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_{BEST_MODEL}_{MODEL_NAME}.pth'), weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    feature_extractor = create_feature_extractor(model, return_nodes=[name2model[MODEL_NAME][2]])

    transform = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])

    dataset = ImageDatasetWithFilenames(os.path.join(DATA_PATH, DATASET_NAME, dataset_type),
                                        transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=1
    )

    X = []
    y = []
    filenames = [] # [subfolder, filename]
    imap = dataset.imap

    for i, data in enumerate(tqdm(dataloader, unit='batch')):
        (image, label, filename) = data
        features = extract_features(image.to(DEVICE), feature_extractor, BATCH_SIZE)
        # features = feature_extractor(image.to(DEVICE))[name2model[MODEL_NAME][2]].detach()
        X.append(features)
        y.append(label)
        filenames.extend([[imap[int(label[j].cpu().numpy())], filename[j]] for j in range(len(filename))])
    
    dump(torch.vstack(X), os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'X_{MODEL_NAME}_{dataset_type}.pickle'))
    dump(torch.hstack(y), os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'y_{MODEL_NAME}_{dataset_type}.pickle'))
    dump(filenames, os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'f_{MODEL_NAME}_{dataset_type}.pickle'))


def train_svm(verbose=False, kernel='linear', kernel_params={}):
    '''
    Trains and saves the linear SVM model.
    '''
    X_train = load(os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'X_{MODEL_NAME}_train.pickle')).cpu().numpy()
    y_train = load(os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'y_{MODEL_NAME}_train.pickle')).cpu().numpy()

    clf = OneVsRestClassifier(SVC(kernel=kernel, verbose=verbose, **kernel_params))

    clf.fit(X_train, y_train)

    dump(clf, os.path.join(MODEL_PATH, DATASET_NAME, f'model_svm_{MODEL_NAME}.pickle'))


def test_svm():
    '''
    Test the trained model on the test set.
    '''
    X_test = load(os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'X_{MODEL_NAME}_test.pickle')).cpu().numpy()
    y_test = load(os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'y_{MODEL_NAME}_test.pickle')).cpu().numpy()

    clf = load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_svm_{MODEL_NAME}.pickle'))

    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')


def set_nn_weights():
    model = build_model(MODEL_NAME, NUM_CLASSES, BASE_MODEL_BIAS)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_{BEST_MODEL}_{MODEL_NAME}.pth'), weights_only=True))
    model = model.to(DEVICE)

    clf = load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_svm_{MODEL_NAME}.pickle'))

    class_layer_name = name2model[MODEL_NAME][3]

    for i in range(len(clf.estimators_)):
        print(f'Class {i}...')
        with torch.no_grad():
            w = clf.estimators_[i].coef_
            w.setflags(write=1)
            b = clf.estimators_[i].intercept_
            b.setflags(write=1)
            if MODEL_NAME.startswith('convnext'):
                getattr(model, class_layer_name)[2].weight[i,:] = torch.nn.Parameter(torch.from_numpy(w))
                getattr(model, class_layer_name)[2].bias[i] = torch.nn.Parameter(torch.from_numpy(b))
            else:
                getattr(model, class_layer_name).weight[i,:] = torch.nn.Parameter(torch.from_numpy(w))
                getattr(model, class_layer_name).bias[i] = torch.nn.Parameter(torch.from_numpy(b))

    torch.save(model.state_dict(), os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_{BEST_MODEL}_{MODEL_NAME}_SVM.pth'))


def visualize(img_test_path, positive=True, title=True):
    img_test = Image.open(img_test_path)
    if img_test.mode != 'RGB':
        img_test = img_test.convert('RGB')

    transform = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    img_test = transform(img_test)

    model = build_model(MODEL_NAME, NUM_CLASSES, BASE_MODEL_BIAS)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_{BEST_MODEL}_{MODEL_NAME}.pth'), weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    feature_vector_extractor = create_feature_extractor(model, return_nodes=[name2model[MODEL_NAME][2]]) # feature vector (before classification layer) extractor / HARD CODED 'avgpool'!!!
    x_test = feature_vector_extractor(img_test.unsqueeze(0).to(DEVICE))[name2model[MODEL_NAME][2]]
    if MODEL_NAME.startswith('densenet'):
        x_test = F.relu(x_test, inplace=True)
        x_test = F.adaptive_avg_pool2d(x_test, (1, 1))
        x_test = torch.flatten(x_test, 1)
    x_test = x_test.detach().squeeze().cpu().numpy()

    X_train = load(os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'X_{MODEL_NAME}_train.pickle')).cpu().numpy()
    y_train = load(os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'y_{MODEL_NAME}_train.pickle')).cpu().numpy()
    
    filenames_train = load(os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'f_{MODEL_NAME}_train.pickle'))

    clf = load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_svm_{MODEL_NAME}.pickle'))

    decision_scores = []
    for i in clf.classes_:
        score = clf.estimators_[i].decision_function([x_test])
        decision_scores.append(score[0])

    if max(decision_scores) < 0:
        print('EVERY PREDICTION IS NEGATIVE IN OVR!')
        # exit(0)

    pred_y = np.argmax(decision_scores)
    
    print(f'Prediction: {pred_y}')
    print(f'Decision score: {decision_scores[pred_y]}')

    alpha_y = csr_matrix(
        (
            clf.estimators_[pred_y].dual_coef_[0],
            (
                [0] * len(clf.estimators_[pred_y].dual_coef_[0]), 
                clf.estimators_[pred_y].support_
             )
        ),
        shape=(1, X_train.shape[0])
    )
    score2 = alpha_y @ (X_train @ x_test.T) + clf.estimators_[pred_y].intercept_
    print(f'Decision score (verif.): {score2[0]}')

    simi = alpha_y.multiply(X_train @ x_test.T)
    simi_only = X_train @ x_test.T
    coefs = list(zip(simi.data, simi.nonzero()[1]))

    simi = np.array(simi.todense()).squeeze()

    positive_coefs = filter(lambda c: c[0] > 0, coefs)
    negative_coefs = filter(lambda c: c[0] < 0, coefs)
    pos_first = sorted(list(positive_coefs), key=lambda c: c[0], reverse=True)[:10] # 10 most similar
    neg_first = sorted(list(negative_coefs), key=lambda c: c[0], reverse=False)[:10] # 10 most similar "on the other side": high similarity * alpha, but labelled -1
    print(pos_first)
    print(neg_first)

    if not positive:
        pos_first = neg_first

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(min(cols * rows, len(pos_first))):
        fname = os.path.join(DATA_PATH, DATASET_NAME, 'train', filenames_train[pos_first[i][1]][0], filenames_train[pos_first[i][1]][1])
        img = np.array(Image.open(fname).convert('RGB'))
        figure.add_subplot(rows, cols, i+1)
        plt.xticks([])
        plt.yticks([])
        # print(filenames_train[pos_first[i][1]][0], filenames_train[pos_first[i][1]][1], pos_first[i][1], X_train[pos_first[i][1]])
        print(simi[pos_first[i][1]])
        print(f'({simi_only[pos_first[i][1]]:.4f}, {alpha_y[0,pos_first[i][1]]})')
        if title:
            plt.title(f'{filenames_train[pos_first[i][1]][0]}\n{simi[pos_first[i][1]]:.4f} ({simi_only[pos_first[i][1]]:.4f})')
        plt.imshow(img.squeeze())
    
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.axis('off')
    plt.show()


def visualize2(img_test_path, layer_name, positive=True, title=True, show=True):
    img_test = Image.open(img_test_path)
    if img_test.mode != 'RGB':
        img_test = img_test.convert('RGB')

    model = build_model(MODEL_NAME, NUM_CLASSES, BASE_MODEL_BIAS)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_{BEST_MODEL}_{MODEL_NAME}_SVM.pth'), weights_only=True))
    # model.load_state_dict(torch.load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_best_{BEST_MODEL}_{MODEL_NAME}.pth'), weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    clf_svm = load(os.path.join(MODEL_PATH, DATASET_NAME, f'model_svm_{MODEL_NAME}.pickle'))
    X_train = load(os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'X_{MODEL_NAME}_train.pickle')).cpu().numpy()
    y_train = load(os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'y_{MODEL_NAME}_train.pickle')).cpu().numpy()
    filenames = load(os.path.join(DATA_FEATURES_PATH, DATASET_NAME, f'f_{MODEL_NAME}_train.pickle')) # do we need filenames saved for every model???

    transform = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    img_test = transform(img_test)
    
    trainset = ImageDatasetWithFilenames(os.path.join(DATA_PATH, DATASET_NAME, 'train'),
                                        transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1
    )

    feature_vector_extractor = create_feature_extractor(model, return_nodes=[name2model[MODEL_NAME][2]]) # feature vector (before classification layer) extractor / HARD CODED 'avgpool'!!!

    x_test = feature_vector_extractor(img_test.unsqueeze(0).to(DEVICE))[name2model[MODEL_NAME][2]]
    if MODEL_NAME.startswith('densenet'):
        x_test = F.relu(x_test, inplace=True)
        x_test = F.adaptive_avg_pool2d(x_test, (1, 1))
        x_test = torch.flatten(x_test, 1)
    x_test = x_test.detach().squeeze().cpu().numpy()

    # print(x_test.shape)
    # exit(0)

    decision_scores = []
    for i in clf_svm.classes_:
        score = clf_svm.estimators_[i].decision_function([x_test])
        decision_scores.append(score[0])

    if max(decision_scores) < 0:
        print('EVERY PREDICTION IS NEGATIVE IN OVR!')

    pred_y = np.argmax(decision_scores)
    print(f'Prediction: {pred_y}')
    print(f'Decision score: {decision_scores[pred_y]}')

    pred_y = clf_svm.predict([x_test])[0]
    print(f'Prediction (verif.): {pred_y}')
    alpha_y = csr_matrix(
        (
            clf_svm.estimators_[pred_y].dual_coef_[0],
            (
                [0] * len(clf_svm.estimators_[pred_y].dual_coef_[0]), 
                clf_svm.estimators_[pred_y].support_
             )
        ),
        shape=(1, X_train.shape[0])
    )
    score2 = alpha_y @ (X_train @ x_test.T) + clf_svm.estimators_[pred_y].intercept_
    print(f'Decision score (verif.): {score2[0]}')

    simi = alpha_y.multiply(X_train @ x_test.T)
    simi_only = X_train @ x_test.T
    coefs = list(zip(simi.data, simi.nonzero()[1]))

    simi = np.array(simi.todense()).squeeze()

    positive_coefs = filter(lambda c: c[0] > 0, coefs)
    negative_coefs = filter(lambda c: c[0] < 0, coefs)
    pos_first = sorted(list(positive_coefs), key=lambda c: c[0], reverse=True)[:10] # 10 most similar: (coef, index)
    neg_first = sorted(list(negative_coefs), key=lambda c: c[0], reverse=False)[:10] # 10 most similar "on the other side": high similarity * alpha, but labelled -1
    print(f'Most similar: {pos_first}')
    print(f'Most similar, but negative: {neg_first}')
    if not positive:
        pos_first = neg_first

    A_grad = []
    def hook_b(module, grad_in, grad_out, vals=A_grad):
        vals.append(grad_out[0].detach())
        print(f'Grad shape: {grad_out[0].shape}')

    activation = []
    def hook_f(module, input, output, vals=activation):
        vals.append(output)
        print(f'Activation shape: {output.shape}')

    layer_b = reduce(getattr, layer_name.split('.'), model)
    layer_f = reduce(getattr, name2model[MODEL_NAME][2].split('.'), model)

    f_hook_b = layer_b.register_full_backward_hook(hook_b) # calculating the gradients
    f_hook_f = layer_f.register_forward_hook(hook_f) # a forward hook is needed to obtain the activation vectors, for the dot product calculation not to be detached from the graph, and the gradients to be calculated

    feature_vector_test = feature_vector_extractor(img_test.unsqueeze(0).to(DEVICE))[name2model[MODEL_NAME][2]]
    if MODEL_NAME.startswith('densenet'):
        feature_vector_test = F.relu(feature_vector_test, inplace=True)
        feature_vector_test = F.adaptive_avg_pool2d(feature_vector_test, (1, 1))
        feature_vector_test = torch.flatten(feature_vector_test, 1)
    feature_vector_test = feature_vector_test.squeeze()

    heatmaps = []
    heatmap_filenames = []

    if show:
        figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(min(cols * rows, len(pos_first))):
        (image, label, filename) = trainloader.dataset[pos_first[i][1]]
        print(label, filename, pos_first[i][1], filenames[pos_first[i][1]])

        img_train = Image.open(os.path.join(DATA_PATH, DATASET_NAME, 'train', filenames[pos_first[i][1]][0], filenames[pos_first[i][1]][1]))
        if img_train.mode != 'RGB':
            img_train = img_train.convert('RGB')

        input = image.to(DEVICE).unsqueeze(0)

        with torch.enable_grad():
            g_train = feature_vector_extractor(input)[name2model[MODEL_NAME][2]]
            if MODEL_NAME.startswith('densenet'):
                g_train = F.relu(g_train, inplace=True)
                g_train = F.adaptive_avg_pool2d(g_train, (1, 1))
                g_train = torch.flatten(g_train, 1)
            g_train = g_train.squeeze()
            loss = g_train.dot(feature_vector_test.detach())

        model.zero_grad()
        loss.backward(retain_graph=True)

        alpha_gradcam = F.adaptive_avg_pool2d(A_grad[-1], (1, 1)) # averaging every feature map
        hmap = F.relu(torch.sum(alpha_gradcam * A_grad[-1], dim=1).squeeze(0)) # heatmap
        hmap /= torch.max(hmap) # normalize

        heatmaps.append(cv2.resize(hmap.cpu().numpy(), (img_train.size[0], img_train.size[1])))
        heatmap_filenames.append(os.path.join(*filenames[pos_first[i][1]]))

        # visualize/save heatmap:
        orig = np.array(Image.open(img_test_path).convert('RGB'))
        orig_resized = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
        map_r = cv2.resize(hmap.cpu().numpy(), (IMG_SIZE, IMG_SIZE))

        cmap = cv2.COLORMAP_JET

        heatmap = cv2.applyColorMap(np.uint8(255 * map_r), cmap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(cv2.resize(np.array(img_train), (IMG_SIZE, IMG_SIZE)), 0.5, heatmap, 0.5, 0)

        if show:
            figure.add_subplot(rows, cols, i+1)
            plt.xticks([])
            plt.yticks([])
            if title:
                plt.title(f'{filenames[pos_first[i][1]][0]}\n{simi[pos_first[i][1]]:.4f} ({simi_only[pos_first[i][1]]:.4f})')
            plt.imshow(cv2.resize(overlay, (img_train.size[0], img_train.size[1])))
    
    if show:
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.axis('off')
        plt.show()

    return heatmaps, heatmap_filenames


# def select_test_images_cub(data_path, images_file, parts_file, N=1):
#     f = codecs.open(images_file, 'r')
#     data = f.readlines()
#     f.close()
#     image_map = {}
#     image_map_r = {}
#     for r in data:
#         rr = r.strip().split(' ')
#         image_map[rr[1]] = rr[0]
#         image_map_r[rr[0]] = rr[1]

#     parts = {}
#     f = codecs.open(parts_file, 'r')
#     data = f.readlines()
#     f.close()
#     for r in data:
#         rr = r.strip().split(' ')
#         if rr[4] == '1':
#             if parts.get(image_map_r[rr[0]], -1) == -1:
#                 parts[image_map_r[rr[0]]] = [(rr[1], rr[2], rr[3])]
#             else:
#                 parts[image_map_r[rr[0]]].append((rr[1], rr[2], rr[3]))

#     selected = []
    
#     for d in os.listdir(os.path.join(data_path, 'cub_200_full', 'test')):
#         filenames = [f for f in os.listdir(os.path.join(data_path, 'cub_200_full', 'test', d))]
#         r = np.random.randint(low=0, high=len(filenames), size=N)
#         for rr in r:
#             fname = os.path.join(d, filenames[rr])
#             selected.append((fname, parts[fname]))

#     dump_json(selected, os.path.join(DATA_FEATURES_PATH, 'cub_features.json'))


def select_test_images_cub(data_path, N=1):
    selected = []

    for d in os.listdir(os.path.join(data_path, 'cub_200_full', 'test')):
        filenames = [f for f in os.listdir(os.path.join(data_path, 'cub_200_full', 'test', d))]
        r = np.random.randint(low=0, high=len(filenames), size=N)
        for rr in r:
            fname = os.path.join(d, filenames[rr])
            selected.append(fname)

    dump_json(selected, os.path.join(DATA_FEATURES_PATH, 'cub_features.json'))


def generate_gradcams(data_path, layer_name):
    files = load_json(os.path.join(DATA_FEATURES_PATH, 'cub_features.json'))
    heatmaps = []
    for f in files:
        img_path = os.path.join(data_path, 'cub_200_full', 'test', f)
        h, fnames = visualize2(img_path, layer_name, positive=True, title=False, show=False)
        heatmaps.append((fnames, h))
    dump(heatmaps, os.path.join(DATA_FEATURES_PATH, f'cub_{layer_name}.pickle'))


def calculate_diversity(images_file, parts_file, layer_name, threshold=0.8):
    f = codecs.open(images_file, 'r')
    data = f.readlines()
    f.close()
    image_map = {}
    image_map_r = {}
    for r in data:
        rr = r.strip().split(' ')
        image_map[rr[1]] = rr[0]
        image_map_r[rr[0]] = rr[1]

    parts = {}
    f = codecs.open(parts_file, 'r')
    data = f.readlines()
    f.close()
    for r in data:
        rr = r.strip().split(' ')
        if rr[4] == '1':
            if parts.get(image_map_r[rr[0]], -1) == -1:
                parts[image_map_r[rr[0]]] = [(rr[1], rr[2], rr[3])]
            else:
                parts[image_map_r[rr[0]]].append((rr[1], rr[2], rr[3]))

    heatmaps = load(os.path.join(DATA_FEATURES_PATH, f'cub_{layer_name}.pickle'))
    score = []
    for img in heatmaps:
        p = {}
        for i in range(len(img[0])):
            fname = img[0][i]
            hmap = img[1][i]
            hmap_bin = hmap > threshold
            for part in parts[fname]:
                if hmap_bin[int(float(part[2])), int(float(part[1]))]:
                    p[part[0]] = p.get(part[0], 0) + 1
        # print(p)
        if len(p) > 0:
            score.append(float(len(p)) / np.mean(list(p.values())))
        else:
            score.append(0.)

    # print(f'Diversity: {np.mean(score)} (+/- {np.std(score)})')
    return np.mean(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=False, action='store_true')
    parser.add_argument('--test', required=False, action='store_true')
    # parser.add_argument('--svm', required=False, action='store_true')
    # parser.add_argument('--finetuned', required=False, action='store_true')

    args = parser.parse_args()

    if args.train:
        pass

    exit(0)

    # MODEL_NAME = 'densenet121'
    # generate_features('train')
    # generate_features('test')

    # train_svm(verbose=True, kernel='linear')
    # # train_svm(verbose=True, kernel='poly', kernel_params={'degree': 4})

    # test_svm()

    # set_nn_weights()

    # visualize('/home/miafranc/zzz/xsvm/data/stanford_dogs/test/n02085620-Chihuahua/n02085620_10074.jpg',
    #           positive=True,
    #           title=False)

    # select_test_images_cub('/home/miafranc/zzz/xsvm/data/',
    #                        1)
    # exit(0)

    # img_test_path = '/home/miafranc/zzz/xsvm/data/cub_200_full/test/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg'
    # img_test = Image.open(img_test_path)
    # # plt.imshow(img_test)
    # # plt.plot([312], [182], 'ro', markersize=12, linewidth=3)
    # # plt.plot([183], [101], 'ro', markersize=2, linewidth=1)
    # # plt.show()

    # # a = np.array(img_test)[:, :, 0]
    # a = np.array(img_test)
    # # a[183, 101] = [255, 0, 0]
    # a[101, 183] = [255, 0, 0]
    # img_test = Image.fromarray(a)
    # # print(a)
    # # print(a[183, 101])

    # plt.imshow(img_test)
    # plt.show()

    # exit(0)

    # model = build_model('resnet50', NUM_CLASSES, BASE_MODEL_BIAS)
    # model = build_model('densenet121', NUM_CLASSES, BASE_MODEL_BIAS)
    # print(model)
    # exit(0)

    # generate_gradcams('/home/miafranc/zzz/xsvm/data/',
    #                 #   'layer3.5.conv1')
    #                 #   'layer4.0.conv1')
    #                 #   'layer4.1.conv1')
    #                 #   'features.denseblock4.denselayer16.conv1')
    #                 #   'features.denseblock4.denselayer14.conv2')
    #                   'features.denseblock4.denselayer15.conv1')
    # exit(0)

    # t = 0.85

    # layers = ['layer3.5.conv1', 'layer4.0.conv1', 'layer4.1.conv1', 
    #           'features.denseblock4.denselayer14.conv2', 'features.denseblock4.denselayer15.conv1', 'features.denseblock4.denselayer16.conv1']
    # scores = {l:0 for l in layers}
    # thresholds = []
    # for t in [0.5 + 0.05*i for i in range(10)]:
    #     print(t)
    #     thresholds.append(t)
    #     for l in layers:
    #         scores[l] += calculate_diversity('/home/miafranc/zzz/xsvm/x3/cub/images.txt', 
    #                         '/home/miafranc/zzz/xsvm/x3/cub/part_locs.txt', 
    #                         l,
    #                         t)

    # for l in layers:
    #     print(f'{l}: {scores[l] / len(thresholds)}')

    # exit(0)

    visualize('/home/miafranc/zzz/xsvm/data/cub_200_full/test/001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg',
              positive=False,
              title=True)
    exit(0)

    # heatmaps = visualize2('/home/miafranc/zzz/xsvm/data/stanford_dogs/test/n02085620-Chihuahua/n02085620_10074.jpg',
    # heatmaps = visualize2('/home/miafranc/zzz/xsvm/data/cub_200_full/test/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg',
    heatmaps = visualize2('/home/miafranc/zzz/xsvm/data/cub_200_full/test/001.Black_footed_Albatross/Black_Footed_Albatross_0003_796136.jpg',
               layer_name='layer3.5.conv1', # resnet
            #    layer_name='layer4.0.conv1', # resnet
            #    layer_name='features.7.0.block.0', # convnext
            #    layer_name='features.denseblock4.denselayer15.conv1', # densenet
               positive=False,
               title=False,
               show=True)

    # a1 = load('a_grad')
    # a2 = load('a_grad2')
    # print(len(a1))
    # print(a1[0].shape)
    # print(a1[1].shape)
    # print(len(a2))
    # print(a2[0].shape)
    # print(a2[1].shape)
    # for i in range(a1[0].shape[1]):
    #     for j in range(a1[0].shape[2]):
    #         for k in range(a1[0].shape[3]):
    #             if a1[0][0,i,j,k] != a2[0][0,i,j,k]:
    #                 print('jajj')

    # gradcam('/home/miafranc/zzz/xsvm/data/stanford_dogs/test/n02085620-Chihuahua/n02085620_10074.jpg')
    # gradcam2('/home/miafranc/zzz/xsvm/data/stanford_dogs/test/n02085620-Chihuahua/n02085620_10074.jpg')
