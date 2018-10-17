import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

load_data = datasets.load_boston()
data_x = load_data.data
data_y = load_data.target

model = LinearRegression()
model.fit(data_x, data_y)

print(model.coef_)
print(model.intercept_)

print(model.get_params())

print(model.score(data_x,data_y))