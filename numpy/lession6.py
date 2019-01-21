import numpy as np
a=np.array([1, 2, 3])
print(a[:])
# a=a.reshape([3,1])
a = a[:, np.newaxis]#数组的转置，np.newaxis添加一维
b = np.array([2, 3, 4])[:, np.newaxis]
print(np.vstack((a, b)))#垂直拼接
print(np.hstack((a, b)))#水平拼接
