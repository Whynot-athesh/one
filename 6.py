from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

# load data
data = fetch_20newsgroups(subset="all")
X = TfidfVectorizer(stop_words="english").fit_transform(data.data)

# model
k = len(set(data.target))
km = KMeans(n_clusters=k, random_state=0).fit(X)

# results
print("Homogeneity:", metrics.homogeneity_score(data.target, km.labels_))
print("V-measure:", metrics.v_measure_score(data.target, km.labels_))
print("Completeness:", metrics.completeness_score(data.target, km.labels_))
print("ARI:", metrics.adjusted_rand_score(data.target, km.labels_))

# top words per cluster
terms = TfidfVectorizer(stop_words="english").fit(data.data).get_feature_names_out()
for i in range(k):
    print(f"\nCluster {i}:")
    for ind in km.cluster_centers_[i].argsort()[-5:]:
        print(terms[ind])