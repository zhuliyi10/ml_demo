import numpy as np
"""
numpy本身数据处理 其中 axis是代表维度，0是代表每1维的操作，1代表第2维，-1代表最后一维
"""
a = np.arange(14, 2, -1).reshape(2, 6)
print(a)
print(np.mean(a))#平均值
print(np.median(a))#中位数
print(np.cumsum(a))#累加
print(np.diff(a))#差值，默认最后一维
print(np.nonzero(a))#不为0的项
print(np.sort(a))#排序，默认最后一维
print(np.clip(a, 5, 9))#设置最大最小值区间，不在这个区间改成临界值
