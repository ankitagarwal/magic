import numpy as np
import math
from sklearn import datasets
import scipy.sparse as ss
from scipy.spatial.distance import euclidean
from scipy import stats

def knn_predict(k, X_train, y_train, x_test):
    predicted = []
    for idx, x_test_pt in enumerate(x_test):
        distances = []
        sa = ss.lil_matrix(x_test_pt)
        sa = sa.todense()
        print idx
        for idx2, x_train_pt in enumerate(X_train):
            sb = ss.lil_matrix(x_train_pt)
            sb = sb.todense()
            distances.append(euclidean(sa, sb))
        min_idx = np.argsort(distances)[0:k]
        predicted.append(stats.mode(y_train[min_idx]).mode[0])
    return predicted

data = datasets.load_svmlight_file("./../a9a")
X_train = data[0]
y_train = data[1]
data = datasets.load_svmlight_file("./../a9a.t")
X_test = data[0]
y_test = data[1]

y_test = y_test[0:10];X_test = X_test[0:10]
y_pred = knn_predict(20, X_train, y_train, X_test)
print y_pred