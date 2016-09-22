import pandas as pd
from statsmodels.nonparametric.api import KernelReg
from kernel_regression import KernelRegression
from sklearn.cross_validation import train_test_split as sk_split
import numpy as np

df = pd.read_csv("abalone.data", header=None)
print df.shape
X = df.loc[:, 1:7].as_matrix()
y = df.loc[:, 8].as_matrix().reshape(-1, 1)
print X.shape
print y.shape
X_train, X_test, y_train, y_test = sk_split(X, y, test_size=0.20)
kr = KernelRegression(kernel="rbf")
kr.fit(X_train, y_train)
print len(X_test)
# Memory issues split X_test, just did two chunks here
X_test_1 = X_test[0:100, :]
X_test_2 = X_test[101:200, :]
pred_y = kr.predict(X_test_1)
pred_y = kr.predict(X_test_2)

