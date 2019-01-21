import numpy as np
"""
新建一个特殊数组
"""

array = np.zeros((2, 3))# 值为0的数组
print(array)

array = np.ones((3, 4))# 值为1的数组
print(array)
array = np.empty((2, 3))# 空值数组,值默认也是0
print(array)
#arange就是产生某个范围的系列，类型默认int
array = np.arange(12).reshape(3, 4)
print(array)
# np.linspace就是生成线性系列，功能与arange相似
array = np.linspace(1, 10, 10).reshape(2, 5)
print(array)
