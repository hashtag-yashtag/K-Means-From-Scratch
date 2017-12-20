# K-Means-From-Scratch
A program written in Python which performs K-Means Clustering on the 2 Moon dataset available in sklearn
Change K accordingly with respect to the number of clusters you wish to create.
Replace
X, _ = make_moons(n_samples=500, random_state=42, noise=0.1)
with
X, _ = *INSERT YOUR DATASET HERE*
to create clusters on whichever datasets you wish to work on

The KMeans uses Euclidean distance (Not to be confused with Manhattan Distance)
This runs for as long as the Sum Squared Error is greater than 0.005
