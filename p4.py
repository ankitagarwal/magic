from sklearn.mixture import GMM
import random
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from sklearn.grid_search import GridSearchCV

# Build model to draw sample from
data = np.random.rand(3000,2)
gmm = GMM(n_components=3)
gmm.fit(data)
sample = gmm.sample(1000)

# Get best BW
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.linspace(0.001, 1.0, 30)},
                    cv=20) # 20-fold cross-validation
grid.fit(sample)
print grid.best_params_

# Fit KDE
kde = KernelDensity(kernel='gaussian', bandwidth=0.0699).fit(sample)
print kde.score(sample)

# GMM fit https://github.com/scikit-learn/scikit-learn/issues/7295
components = [2, 3, 5, 10]
for idx, value in enumerate(components):
    g = GMM(n_components=value, random_state=43)
    g.fit(sample)
    print "log likely hood for components = " + str(value) + " is " + str(g.score_samples(sample))