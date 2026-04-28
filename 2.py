import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# data
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# models
m1 = DecisionTreeRegressor(max_depth=2).fit(X, y)
m2 = DecisionTreeRegressor(max_depth=5).fit(X, y)

# predict
Xt = np.arange(0, 5, 0.01)[:, None]

# plot
plt.scatter(X, y)
plt.plot(Xt, m1.predict(Xt))
plt.plot(Xt, m2.predict(Xt))
plt.show()