import math
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import pylab

#200 random numbers
np.random.seed(200)

#Change the value of K depending on the number of clusters needed
K = 3

# make_moons is a dataset in sklearn
X, _ = make_moons(n_samples=500, random_state=42, noise=0.1)

#To increase the gain and not have a ton of zeros in the final calculation
X[:, 1] = X[:, 1] * 100

pylab.scatter(X[:,0], X[:,1])
pylab.show()

#This list stores the centroids at each iteration
centroids = []
for i in range (0, K):
    pos = np.random.randint(0, 500)
    x_coord = X[pos][0]
    y_coord = X[pos][1]
    centroids.append([x_coord, y_coord])

# Finds the distance between two points (Can be multidimensional)
def EuclideanDistance(feature_one, feature_two):
    distance_sq = 0
    for i in range(len(feature_one)):
        distance_sq += (feature_one[i] - feature_two[i])**2

    return math.sqrt(distance_sq)

#Calculates the distance of each point in the dataset from the centroids
def distanceFromCentroids(centroids, X):
    distanceArray = [[0 for x in range(len(X))] for y in range(K)]
    for i in range(len(centroids)):
        for j in range(len(X)):
            distanceArray[i][j] = EuclideanDistance(centroids[i], X[j])
    return distanceArray

def CentroidOfCluster(X):
    centroid = 0
    for i in X:
        centroid = centroid + (i/len(X))
    return centroid

# Sum Squared Error
# Find more information here:
# https://hlab.stanford.edu/brian/error_sum_of_squares.html
def SSE(X, r_k):
    d = 0
    for i in X:
        for j in r_k:
            d = d + EuclideanDistance(i, j)**2

    return d

# Sorts the points into their respective clusters
def formClusters(distanceArray):
    clusterArray = []
    minIndex = 0
    for i in range(len(distanceArray[0])):
        min = float("inf")
        for j in range(len(distanceArray)):
            if distanceArray[j][i] < min:
                min = distanceArray[j][i]
                minIndex = j
        clusterArray.append(minIndex)

    return clusterArray

#Binds the entire program together
def kMeans(centroids, X, K):
    distanceArray = distanceFromCentroids(centroids, X)
    clusterArray = formClusters(distanceArray)
    Clusters = [[] for x in range(K)]
    for j in range(len(clusterArray)):
        Clusters[clusterArray[j]].append(X[j])

    r_k = []
    for x in range(len(Clusters)):
        if (Clusters[x] != []):
            pp = CentroidOfCluster(Clusters[x])
            x_coord = pp[0]
            y_coord = pp[1]
            r_k.append([x_coord, y_coord])
        else:
            x_coord = X[np.random.randint(0, 500)][0]
            y_coord = X[np.random.randint(0, 500)][1]
            r_k.append([x_coord, y_coord])

    return r_k


i = 1
while(1):
    print("Iteration: ", i)
    i+=1
    #print("Centroids: ", centroids)
    error = SSE(X, centroids)

    # Uncomment the following lines if you wish to see the changes in the dataset
    # at each iteration

    #GRAPHING:
    #pylab.scatter(X[:,0], X[:,1], color='blue')
    #for x in range(len(centroids)):
    #    pylab.scatter(centroids[x][0], centroids[x][1], color='red')
    #pylab.show()

    centroids = kMeans(centroids, X, K)
    newError = SSE(X, centroids)
    print("Error: ")
    print(np.abs(newError - error))
    if(np.abs(newError - error) < 1e-5):
        print(newError, ", ",error)
        break


#Final centroids and graph
pylab.scatter(X[:,0], X[:,1], color='blue')
for x in range(len(centroids)):
    pylab.scatter(centroids[x][0], centroids[x][1], color='red')

pylab.show()