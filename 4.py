import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis

# data
X = StandardScaler().fit_transform(load_iris().data)

# models
models = [
    ("PCA", PCA(2)),
    ("FA", FactorAnalysis(2)),
    ("Varimax", FactorAnalysis(2, rotation="varimax"))
]

# plot
for i, (name, m) in enumerate(models, 1):
    m.fit(X)
    plt.subplot(1, 3, i)
    plt.imshow(m.components_.T)
    plt.title(name)

plt.show()