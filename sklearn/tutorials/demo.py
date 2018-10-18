import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_digits()
data = iris.images
output = iris.target
print(data.shape)
data1 = data.reshape((data.shape[0], -1))
print(data1.shape)
plt.imshow(data[0], cmap=plt.cm.gray_r)
plt.show()
