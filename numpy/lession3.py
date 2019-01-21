import numpy as np
"""
数组的运算
"""
a = np.array([10, 20, 30, 40])
b = np.arange(4)
c = np.array([[1, 0], [1, 1]])
d = np.arange(4).reshape(2, 2)
print(a)
print(b)
print(a - b)
print(a + b)
print(a * b)
print(b ** 2)
print(10 * np.sin(b))
print(a < 3)
print(c * d)
print(np.dot(c, d))#点积
r = np.random.random((2, 4))#0到1的随机数
print(r)
r=np.random.uniform(0.6,1,(2,3))#指定范围随机数
print(r)
print(np.max(r,axis=1))#是对axis维平均，比如axis=0是对第一维求平均
