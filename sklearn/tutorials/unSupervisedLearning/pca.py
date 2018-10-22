import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data
y = iris.target
pca = PCA(n_components=2)
new_X = pca.fit_transform(X)
print(X)
print(new_X)
print(pca.explained_variance_ratio_)

k_means = cluster.KMeans(n_clusters=3)
k_means.fit(new_X)
plt.scatter(new_X[:, 0], new_X[:, 1], c=y)
plt.show()
