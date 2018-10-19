import numpy as np
from sklearn import preprocessing

X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])
# 使得均值0，方差1
X_scaled = preprocessing.scale(X_train)
print(X_scaled)

print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

# 使得放缩在某个范围[0,1]
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print(X_train_minmax)

# 使得放缩在某个范围[-1,1]
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
print(X_train_maxabs)

# 二进制归一化
binarizer = preprocessing.Binarizer()
X_train_binarizer = binarizer.fit_transform(X_train)
print(X_train_binarizer)
