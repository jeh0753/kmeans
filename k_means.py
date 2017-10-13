import numpy as np
import random
from scipy.spatial.distance import cdist
#Datasets for testing
from sklearn.datasets import load_iris
from sklearn import preprocessing


class KMeans(object):
    def __init__(self):
        self.n_ks = None 
        self.X = None
        self.iters = None
        self.centroids = None
        self.smallest_dist_idx = None
        pass

    def update_centroids(self):
        '''
        Updates each centroid a single time
        '''
        dist_to_centroids = cdist(self.X, self.centroids)
        self.smallest_dist_idx = dist_to_centroids.argmin(axis=1) 
        for k in range(self.n_ks):
            self.centroids[k] = np.mean(self.X[self.smallest_dist_idx==k], axis=0)
        pass

    def fit(self, X, n_ks=3, n_iters=1000):
        '''
        Initiates n_ks centroids, then updates them n_iters times to fit to X.
        Input: A numpy matrix of the data, and optionally the number of ks, and number of iterations 
        '''
        self.centroids = np.array(random.sample(X, n_ks))
        self.X = X
        self.n_iters = n_iters
        self.n_ks = n_ks
        for _ in range(self.n_iters):
            self.update_centroids() 
        pass

    def score(self):
        '''
        Compute the Total Sum of Squares for the data points in all clusters
        Output: a float, the TSS
        '''
        total = 0
        for k in range(self.n_ks):
            total += np.sum((np.sum(self.X[self.smallest_dist_idx==k], axis=0) - self.centroids[k])**2)
        return total


if __name__ == '__main__':
    data = load_iris()
    X = np.array(data.data)
    X = preprocessing.scale(X)
    kmean = KMeans()
    kmean.fit()
    kmean.score()
    kmean.centroids
