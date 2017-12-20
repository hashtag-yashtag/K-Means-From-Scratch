import math
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import pylab

np.random.seed(200)
K = 5

X, _ = make_moons(n_samples=500, random_state=42, noise=0.1)
X[:, 1] = X[:, 1] * 100

centroids = [0 for x in range(K)]

#pylab.scatter(X[:, 0], X[:, 1], c=_)
#pylab.show()

for i in range (0, K):
    pos = np.random.randint(0, 500)
    centroids[i] = X[pos]

print(centroids.__len__())

def EuclideanDistance(feature_one, feature_two):
    distance_sq = 0
    for i in range(len(feature_one)):
        distance_sq += (feature_one[i] - feature_two[i])**2

    return math.sqrt(distance_sq)

def distanceFromCentroids(centroids, X):
    distanceArray = [[0 for x in range(len(X))] for y in range(K)]
    print(distanceArray)
    print(centroids[1])
    for i in range(len(centroids)):
        for j in range(len(X)):
            distanceArray[i][j] = EuclideanDistance(centroids[i], X[j])
    return distanceArray

check = distanceFromCentroids(centroids, X)
print(check)

    # tolerance = 0
    # max_iterations = 0
    # def __init__(self, k=3, tolerance = 0.001, max_iterations = 500):
    #     self.centroids
    #     self.k = k
    #     self.tolerance = tolerance
    #     self.max_iterations = max_iterations
    #
    # def EuclideanDistance(feature_one, feature_two):
    #     distance_sq = 0;
    #     for i in range(len(feature_one)):
    #         distance_sq += (feature_one[i] - feature_two[i])**2
    #
    #     return math.sqrt(distance_sq)
    #
    # def clustering(self, X):
    #     for i in range(self.k):
    #         self.centroids[i] = X[i]
    #
    #     for i in range (self.max_iterations):
    #         self.classes = {}
    #         for i in range(self.k):
    #             self.classes[i] = []
    #
    #         for features in X:
    #             distances = [np.linalg.norm(features - self.centroids[centroid])]
