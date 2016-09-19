import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# Read train data
train = pd.read_csv("adult.data", header=None)
# print train.head(n=1)
print len(train)

# Clean missing values
train.replace(r"\?", np.nan, inplace=True, regex=True)
train = train.dropna(axis=0)
print len(train)

# Read test data
test = pd.read_csv("adult.test", skiprows=1, header=None)
# print test.head(n=1)
print len(test)

# clean missing values
test.replace(r"\?", np.nan, inplace=True, regex=True)
test = test.dropna(axis=0)
print len(test)

# Some prepossessing convert categorical value to classes -
for i in [1, 3, 5, 6, 7, 8, 9, 13]:
    le = preprocessing.LabelEncoder()
    le.fit(train[i])
    train[i] = le.transform(train[i])
    test[i] = le.transform(test[i])

X_cat = train.ix[:, [1, 3, 5, 6, 7, 8, 9, 13]]
y = train[14]
X_cont = train.ix[:, [0, 2, 4, 10, 11, 12]]
print X_cat.head(n=5)
print X_cont.head(n=1)

model1 = MultinomialNB().fit(X_cat, y)
model2 = GaussianNB().fit(X_cont, y)

X_cat_test = test.ix[:, [1, 3, 5, 6, 7, 8, 9, 13]]
y_true = test[14]
y_true.replace(r"\.", "", inplace=True, regex=True)
X_cont_test = test.ix[:, [0, 2, 4, 10, 11, 12]]

prob = np.multiply(model1.predict_proba(X_cat_test), model2.predict_proba(X_cont_test))

y_pred = []
for idx, row in enumerate(prob):
    if row[0] > row[1]:
        y_pred.append(" <=50K")
    else:
        y_pred.append(" >50K")

print accuracy_score(y_true, model1.predict(X_cat_test))
print accuracy_score(y_true, model2.predict(X_cont_test))
print accuracy_score(y_true, y_pred)