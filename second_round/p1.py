from scipy.ndimage import label
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

# Get lables and priors
def get_prior(X, y):
    groups = y.groupby(y)
    dict = groups.count().to_dict()
    priors = {}
    labels = {}
    for label in dict:
        priors[label] = float(dict[label])/float(len(y))
    return dict, priors

# Get featurewise likelyhood
def get_likelyhood(train, x_idx, lbl_count):
    likelyhoods = {}
    for x in x_idx:
        groups = DataFrame({'count' : train.groupby( [x, 14] ).size()}).reset_index()
        for idx, row in groups.iterrows():
            lbl = row[14]
            groups.ix[idx, "likelyhood"] = row["count"]/float(lbl_count[lbl])
        likelyhoods[x] = groups
    return likelyhoods

def classify(likelyhood, prior, pt, x_idx, lbls):
    probs = {}
    for lbl in lbls:
        p_lbl = 1
        # Loop through all features and multiple likelyhoods as needed.
        for x in x_idx:
            like = likelyhood[x]
            factor = like[(like[x] == pt[x]) & (like[14] == lbl)]["likelyhood"]
            p_lbl *= factor.values[0]
        p_lbl *= p_lbl * prior[lbl]
        probs[lbl] = p_lbl
    # print probs
    return max(probs.iterkeys(), key=lambda k: probs[k])



train = pd.read_csv("./../adult.data", header=None)
test = pd.read_csv("./../adult.test", header=None, skiprows=1)

# Clean missing values
train.replace(r"\?", np.nan, inplace=True, regex=True)
train.replace(r" <=50K", "<=50K", inplace=True, regex=True)
train.replace(r" >50K", ">50K", inplace=True, regex=True)
train = train.dropna(axis=0)

X_cat = train.ix[:, [1, 3, 5, 6, 7, 8, 9, 13]]
y = train[14]
train_whole = train.ix[:, [1, 3, 5, 6, 7, 8, 9, 13, 14]]

# Clean missing values for test data
test.replace(r"\?", np.nan, inplace=True, regex=True)
test.replace(r" <=50K", "<=50K", inplace=True, regex=True)
test.replace(r" >50K", ">50K", inplace=True, regex=True)
test = test.dropna(axis=0)

X_cat_test = test.ix[:, [1, 3, 5, 6, 7, 8, 9, 13]]
y_test = test[14]
test_whole = test.ix[:, [1, 3, 5, 6, 7, 8, 9, 13, 14]]

x_idx = [1, 3, 5, 6, 7, 8, 9, 13]
lbls, prior = get_prior(X_cat, y)
likelyhood = get_likelyhood(train_whole, x_idx, lbls)
# print lbls
# print prior
# print likelyhood
# print X_cat[0:1]
# print X_cat_test[0:50]
for idx, pt in X_cat_test[0:10].iterrows():
    print classify(likelyhood, prior, pt, x_idx, lbls)