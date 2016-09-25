import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.mixture import GMM

# Build model to draw sample from
data1 = np.random.randn(100, 2)
data2 = np.random.randn(100, 2) + np.array([20, 30])
data = np.vstack([data1, data2])
print data

# display predicted scores by the model as a contour plot
x = np.linspace(-20.0, 40.0)
y = np.linspace(-20.0, 40.0)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T

ax = []
fig, ax = plt.subplots(1, 3, figsize=(20, 10))
fig.subplots_adjust(hspace = .4)

# Plot models for various components
for components in range(1, 4):
    gmm = GMM(n_components = components, covariance_type='full')
    gmm.fit(data)
    Z = -gmm.score_samples(XX)[0]
    Z = Z.reshape(X.shape)

    CS = ax[components-1].contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
    ax[components-1].scatter(data[:, 0], data[:, 1], .8)

plt.colorbar(CS, shrink=0.8, extend='both')
plt.axis('tight')
plt.show()