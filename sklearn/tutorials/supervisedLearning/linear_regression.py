import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)
y_loss = y_predict - y_test
print(np.mean(y_loss ** 2))
print(regr.score(X_test, y_test))

x = range(X_test.shape[0])
plt.plot(x, y_predict)
plt.plot(x, y_test)
plt.show()
