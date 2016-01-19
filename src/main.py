#-*- coding: utf-8 -*-

import random
import numpy as np
import math
from sklearn import datasets
from scipy.cluster.vq import kmeans, whiten, vq

def load_iris_data():
    """
    Fonction permettant de charger en mémoire les données IRIS, contenues dans scikit learn.
    """

    iris = datasets.load_iris()
    X = iris.data[:, :4]
    C = iris.target

    return (X, C)

def construct_degree_matrix(epsilon_graph):
    """
    """

    degree_matrix = np.zeros((len(epsilon_graph), len(epsilon_graph)))

    i = 0
    while i < len(epsilon_graph):
        line = epsilon_graph[i]
        count = 0
        for j in range(0, len(line)):
            if line[j] != 0:
                count+=1
        degree_matrix[i,i] = count
        i += 1

    return degree_matrix

def similarity(xi, xj, sigma=1):
    """
    Donne la similarité entre deux données.
    """

    return math.exp(-(np.linalg.norm(np.subtract(xi, xj))) / math.pow(sigma, 2))

def construct_epsilon_graph(X, epsilon = 0.5):
    """
    Construit l'epsilon-graph associé aux données de l'IRIS
    """

    res = np.zeros((len(X), len(X)))

    for i, xi in enumerate(X):
        for j, xj in enumerate(X):
            sim = similarity(xi, xj)

            if sim > epsilon:
                res[i, j] = sim
            else:
                res[i, j] = 0

    return res

def construct_spectral_vector_matrix(R, k):
    """
    """

    U,s,_ = np.linalg.svd(R, full_matrices=True)

    return U[:k]

def spectral_clustering(k):
    X, C = load_iris_data()
    epsilon_graph = construct_epsilon_graph(X)
    D = construct_degree_matrix(epsilon_graph)
    W = construct_epsilon_graph(X, -1)
    L = np.subtract(D, W)
    U_k = construct_spectral_vector_matrix(L, k)

    kmeans_data = U_k.T
    print(kmeans_data.shape)
    print(X.shape)
    whitened = whiten(kmeans_data)

    centroids,_ = kmeans(whitened, k)
    clusters,_ = vq(whitened, centroids)

    for i,data in enumerate(X):
        print("{}, {}".format(data, clusters[i]))

def main():
    spectral_clustering(3)

if __name__ == '__main__':
    main()
