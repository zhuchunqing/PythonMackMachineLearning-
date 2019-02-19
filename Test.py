import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from collections import Counter
import KNNpackaging as knn 
from sklearn import datasets
def train_test_split(X,Y,test_redio=0.2,seed=None):
    iriss=datasets.load_iris()
    X = iriss.data
    Y = iriss.target
    assert X.shape[0]==Y.shape[0],\
    "the size of X must be equal to the size of y"
    assert 0.0<=test_redio<=1.0,\
    "test_ration must be valid"
    if seed:
        np.random.seed(seed)
    shuffke_indexes=np.random.permutation(len(X))
    test_ratio=0.2
    test_size=int(len(X)*test_ratio)
    test_indexes=shuffke_indexes[:test_size]
    train_indexes=shuffke_indexes[test_size:]
    X_train=X[train_indexes]
    Y_train=Y[train_indexes]
    X_test=X[test_indexes]
    Y_test=Y[test_indexes]
    return X_train,Y_train,X_test,Y_test
