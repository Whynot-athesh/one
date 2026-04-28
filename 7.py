from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics

# data
X, y = load_digits(return_X_y=True)
X = minmax_scale(X)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

# model (RBM + Logistic)
model = Pipeline([
    ("rbm", BernoulliRBM(n_components=100)),
    ("log", LogisticRegression(max_iter=1000))
])

# train + test
model.fit(Xtr, ytr)
pred = model.predict(Xte)

print(metrics.classification_report(yte, pred))