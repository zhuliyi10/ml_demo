import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

load_data = datasets.load_boston()
data_x = load_data.data
data_y = load_data.target
print(data_x)
print(data_y)
model = LinearRegression()
model.fit(data_x, data_y)
print(model.predict(data_x[:4, :]))
print(data_y[:4])

x, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
plt.scatter(x, y)
plt.show()
