import numpy as np


def one_hot_to_int(labels):
    return np.nonzero(labels)[1].tolist()


def int_to_one_hot(labels, class_dict_len):
    return np.eye(class_dict_len)[labels].tolist()


def confusion_matrix(actual, predicted, class_dict_len):
    cm = np.zeros((class_dict_len, class_dict_len), dtype=int)
    for a, p in zip(one_hot_to_int(actual), one_hot_to_int(predicted)):
        cm[a, p] += 1
    return cm


def pca(data, dims=2):
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = sc.linalg.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs

def rstft(x, n=1024, w=1024, h=512):
    ns = (len(x) - w) // h
    y = np.empty((n // 2 + 1, ns), np.complex)
    for i in range(ns):
        y[:, i] = np.fft.rfft(x[(h * i):(h * i + w)], n)
    return y