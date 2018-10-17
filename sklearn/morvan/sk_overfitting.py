import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import validation_curve  # 学习曲线模块
from sklearn.svm import SVC

digits = datasets.load_digits()
x = digits.data
y = digits.target
param_range=np.logspace(-6,-2.3,5)
train_loss, test_loss = validation_curve(SVC(), x, y,param_name='gamma',param_range=param_range, cv=10, scoring='mean_squared_error')

train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)
plt.plot(param_range, train_loss_mean, 'o-', color='r', label='training')
plt.plot(param_range, test_loss_mean, 'o-', color='g', label='test')
plt.xlabel('gamma')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
