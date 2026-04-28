import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# fixed data (same every run)
X, y = make_moons(noise=0.3, random_state=0)
X = StandardScaler().fit_transform(X)

# fixed split
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# models
models = [
    ("KNN", KNeighborsClassifier(3)),
    ("SVM", SVC()),
    ("Tree", DecisionTreeClassifier())
]

# train + plot
for i, (name, m) in enumerate(models, 1):
    m.fit(Xtr, ytr)
    acc = m.score(Xte, yte)

    plt.subplot(1, 3, i)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title(f"{name}: {acc:.2f}")

plt.show()