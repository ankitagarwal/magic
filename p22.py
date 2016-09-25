from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, cross_validation


data = datasets.load_svmlight_file("a9a")
X = data[0]
print X.shape
y = data[1]

loo = cross_validation.LeaveOneOut(X.shape[0])
errors = []
# Train and find best k
# k = [1, 2, 5, 10, 20]
# for idx, k in enumerate(k):
#     score = 0
#     i = 0
#     for train_index, test_index in loo:
#         i += 1
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         clf = KNeighborsClassifier(k)
#         clf.fit(X_train, y_train)
#         score += clf.score(X_test, y_test)
#         if i % 1000 == 0:
#             print "Progress - i = " + str(i) + " k = " + str(k)
#     sc = score/float(X.shape[0])
#     print "Finale score for k = " + str(k) + " " + str(sc)
#     errors.append(1-sc)

clf = KNeighborsClassifier(20)
clf.fit(X, y)

data = datasets.load_svmlight_file("a9a.t")
X = data[0]
y = data[1]

X1 = X[0:6000]
y1 = y[0:6000]
X2 = X[6001:]
y2 = y[6001:]
score = 0
score += clf.score(X1, y1)
score += clf.score(X2, y2)
score = score/2
print "Score on test data is -  " + str(score)
print y[0:10]