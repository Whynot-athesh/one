import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

# load data
df = pd.read_csv("diabetes.csv")
X, y = df.drop("Outcome", axis=1), df["Outcome"]

# scale + split
X = StandardScaler().fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=10)

# Decision Tree CV
print("DT:", cross_val_score(DecisionTreeClassifier(), X, y, cv=5).mean())

# Bagging
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100).fit(Xtr, ytr)
print("Bag:", bag.score(Xte, yte))

# Random Forest CV
print("RF:", cross_val_score(RandomForestClassifier(), X, y, cv=5).mean())