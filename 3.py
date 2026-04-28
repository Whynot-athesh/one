import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# data
np.random.seed(0)
X = np.r_[np.random.randn(500,2), np.random.randn(500,2)+[-6,3]]

# find best model using BIC
best_bic = 1e9
for k in range(1,7):
    gmm = GaussianMixture(n_components=k).fit(X)
    bic = gmm.bic(X)
    if bic < best_bic:
        best_bic, best_gmm = bic, gmm

# predict + plot
labels = best_gmm.predict(X)
plt.scatter(X[:,0], X[:,1], c=labels)
plt.title(f"Best components: {best_gmm.n_components}")
plt.show()