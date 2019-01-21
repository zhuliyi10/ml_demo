import numpy as np
"""
赋值，数组之前的赋值只是引用
"""
a = np.arange(4)
b = a
c = a
d = b
a[0] = 11
print(b is a)
