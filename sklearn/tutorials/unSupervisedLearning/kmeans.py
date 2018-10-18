from sklearn import cluster
from sklearn import datasets

iris=datasets.load_iris()
X=iris.data
y=iris.target

k_means=cluster.KMeans(n_clusters=3)
k_means.fit(X)
print(k_means.labels_)
print(y)