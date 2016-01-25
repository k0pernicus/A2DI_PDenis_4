#-*- coding: utf-8 -*-

import random
import numpy as np
import math
from sklearn import datasets
from scipy.cluster.vq import kmeans, whiten, vq
from numpy import arange

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

def similarity(xi, xj, sigma = 1):
    """
    Donne la similarité entre deux données.
    """

    return math.exp(-(np.linalg.norm(np.subtract(xi, xj))) / math.pow(sigma, 2))

def construct_epsilon_graph(X, epsilon = 0.5, sigma = 1):
    """
    Construit l'epsilon-graph associé aux données de l'IRIS
    """

    res = np.zeros((len(X), len(X)))

    for i, xi in enumerate(X):
        for j, xj in enumerate(X):
            sim = similarity(xi, xj, sigma)

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

def kmeans_clustering(data, k):
    whitened = whiten(data)
    centroids, _ = kmeans(whitened, k)
    clusters, _ = vq(whitened, centroids)

    return clusters

def spectral_clustering(data, k, sigma = 1, epsilon = 0.5):
    epsilon_graph = construct_epsilon_graph(data, epsilon, sigma)
    D = construct_degree_matrix(epsilon_graph)
    W = construct_epsilon_graph(data, -1)
    L = np.subtract(D, W)
    U_k = construct_spectral_vector_matrix(L, k)
    kmeans_data = U_k.T

    return kmeans_clustering(kmeans_data, k)

def count_error(clusters, classes):
    count = 0

    for i in range(len(clusters)):
        if not clusters[i] == classes[i]:
            count += 1

    return count

def mean_kmeans_error(data, classes, k, n):
    errors = 0

    for i in range(n):
        clusters = kmeans_clustering(data, k)
        errors += count_error(clusters, classes)

    return errors / n

def mean_spectral_error(data, classes, k, sigma, epsilon, n):
    errors = 0

    for i in range(n):
        clusters = spectral_clustering(data, k, sigma, epsilon)
        errors += count_error(clusters, classes)

    return errors / n

def find_best_params(data, classes, k, n):
    best_sigma = 0.1
    best_epsilon = 0.1
    best_error = mean_spectral_error(data, classes, k, best_sigma, best_epsilon, n)

    for sigma in arange(0.1, 1.1, 0.1):
        for epsilon in arange(0.1, 1.1, 0.1):
            error = mean_spectral_error(data, classes, k, sigma, epsilon, n)
            print("sigma: {}, epsilon: {}, error: {}".format(sigma, epsilon, error))

            if error < best_error:
                best_sigma = sigma
                best_epsilon = epsilon
                best_error = error

    return best_sigma, best_epsilon, best_error

def print_best_param_and_errors(n):
    X, C = load_iris_data()
    best_spectral_sigma, best_spectral_epsilon, best_spectral_error = find_best_params(X, C, 3, n)

    print("-------")
    print("Kmeans:")
    print("\terrors  = {}".format(mean_kmeans_error(X, C, 3, n)))
    print("-------")
    print("Spectral:")
    print("\tsigma   = {}".format(best_spectral_sigma))
    print("\tepsilon = {}".format(best_spectral_epsilon))
    print("\terrors  = {}".format(best_spectral_error))

def main():
    ####### RECHERCHE DES MEILLEURS PARAMÈTRES #####
    #
    # /!\ C'est un peu long
    print_best_param_and_errors(10)
    #
    # Résultats :


if __name__ == '__main__':
    main()
