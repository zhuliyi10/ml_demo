import numpy as np

a = np.arange(12).reshape(3, 4)
print(a)
b = np.array_split(a, 3, axis=1)#对axis维数据进行分割
print(b)
