import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
k_means = cluster.KMeans(n_clusters=4)
k_means.fit(X)
print(k_means.labels_)
print(y)
plt.scatter(X[:,0],X[:,1],c=k_means.labels_)
plt.show()

# data, target = datasets.make_blobs(n_samples=100, n_features=2, centers=3)
# plt.scatter(data[:, 0], data[:, 1], c=target)
# plt.show()