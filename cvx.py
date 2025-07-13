import numpy as np
import torch
import codecs
import pickle
from scipy.stats import spearmanr, kendalltau

DATA_N, DATA_D = 100, 10
NORM_WEIGHT = 0.05

def exact_solution1(trainset):
    '''
    Exact solution of regularized least squares
    '''
    X = trainset.tensors[0].cpu().numpy().astype(np.float64)
    y = trainset.tensors[1].cpu().numpy().astype(np.float64)
    y = np.expand_dims(y, axis=0)

    w = np.linalg.solve(X.T @ X + NORM_WEIGHT * X.shape[0] * np.eye(X.shape[1]), X.T @ y.T)  # better than with np.linalg.inv

    return w


def exact_solution2(trainset):
    '''
    Exact solution of regularized least squares using betas from the representer theorem
    '''
    X = trainset.tensors[0].cpu().numpy().astype(np.float64)
    y = trainset.tensors[1].cpu().numpy().astype(np.float64)
    y = np.expand_dims(y, axis=0)

    A = X @ X.T

    beta = np.linalg.solve(A @ A + NORM_WEIGHT * X.shape[0] * A, A @ y.T)  # better than with np.linalg.inv

    return beta


def solve_exactly_and_evaluate(trainset, validationset):
    '''
    Calculates and returns the optimal w and beta vectors + calculates accuracy.
    '''
    w = exact_solution1(trainset)
    beta = exact_solution2(trainset)

    X_t = trainset.tensors[0].cpu().numpy().astype(np.float64)
    y_t = trainset.tensors[1].cpu().numpy().astype(np.float64)
    y_t = np.expand_dims(y_t, axis=0)
    
    pred_train = 2* (X_t @ w > 0) - 1
    accuracy = (pred_train == y_t.T).sum() / X_t.shape[0]
    print(f'Train acc: {accuracy}')

    X_v = validationset.tensors[0].cpu().numpy().astype(np.float64)
    y_v = validationset.tensors[1].cpu().numpy().astype(np.float64)
    y_v = np.expand_dims(y_v, axis=0)
    
    pred_val = 2* (X_v @ w > 0) - 1
    accuracy = (pred_val == y_v.T).sum() / X_v.shape[0]
    print(f'Val. acc: {accuracy}')

    return w, beta


def load_dataset():
    f = codecs.open(f'data/gaussian/gaussian_train_{DATA_N}_{DATA_D}', 'rb')
    trainset = pickle.load(f)
    f.close()
    f = codecs.open(f'data/gaussian/gaussian_val_{DATA_N}_{DATA_D}', 'rb')
    valset = pickle.load(f)
    f.close()
    
    return trainset, valset


def exact_solution3(trainset, lamb=1e-5):
    X = trainset.tensors[0].cpu().numpy().astype(np.float64)
    y = trainset.tensors[1].cpu().numpy().astype(np.float64)
    y = np.expand_dims(y, axis=0)

    w = np.linalg.solve(X.T @ X + NORM_WEIGHT * X.shape[0] * np.eye(X.shape[1]), X.T @ y.T)  # better than with np.linalg.inv

    beta = np.linalg.solve(X @ X.T + lamb*np.eye(X.shape[0]), X @ w)

    return w, beta


def loss(X, y, beta):
    w = X.T @ beta
    f = X @ w
    return (1./X.shape[0]) * (np.linalg.norm(f - y) ** 2) + NORM_WEIGHT * (np.linalg.norm(w) ** 2)


if __name__ == '__main__':
    trainset, valset = load_dataset()

    w1, beta1 = solve_exactly_and_evaluate(trainset, valset)

    w3, beta3 = exact_solution3(trainset)

    X = trainset.tensors[0].to(torch.float64)
    y = trainset.tensors[1].unsqueeze(0).T.to(torch.float64)
    
    f = X @ w1
    beta2 = (y - f) / (NORM_WEIGHT * X.shape[0])

    print(f'beta1={beta1.T}')
    print(f'beta2={beta2.T}')
    print(f'beta3={beta3.T}')

    print('*'*30)

    f1 = (X @ X.T @ beta1).T
    print(f'f1={f1}')
    f2 = (X @ X.T @ beta2).T
    print(f'f2={f2}')
    f3 = (X @ X.T @ beta3).T
    print(f'f3={f3}')

    print('*'*30)

    a = []
    a.append(np.argsort(beta1[:,0]))
    a.append(np.argsort(beta2[:,0]))
    a.append(np.argsort(beta3[:,0]))

    B = [beta1, beta2.numpy(), beta3]
    F = [f1.numpy(), f2.numpy(), f3.numpy()]
    for i in range(len(B)-1):
        for j in range(i+1, len(B)):
            print(f'beta{i}-{j}: {np.linalg.norm(B[i]-B[j])}')
            print(f'f{i}-{j}: {np.linalg.norm(F[i]-F[j])}')
            print(f'sign beta{i}-{j}: {((B[i]>0) == (B[j]>0)).sum() / X.shape[0]}')
            print(f'sign f{i}-{j}: {((F[i]>0) == (F[j]>0)).sum() / X.shape[0]}')
            print(f'Kendall tau beta{i}-{j}: {kendalltau(a[i], a[j], variant="b")}')

    print('*'*30)

    print(f'||f|| = {np.mean([np.linalg.norm(f1), np.linalg.norm(f2), np.linalg.norm(f3)])}')
    print(f'||beta|| = {np.mean([np.linalg.norm(beta1), np.linalg.norm(beta2), np.linalg.norm(beta3)])}')

    print('*'*30)
    print(f'loss1 = {loss(X, y, beta1)}')
    print(f'loss2 = {loss(X, y, beta2)}')
    print(f'loss3 = {loss(X, y, beta3)}')
