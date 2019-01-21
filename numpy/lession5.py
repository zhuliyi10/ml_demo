import numpy as np

a=np.arange(3,15).reshape(3,4)
print(a)
print(a[2,1:])
print(a.flatten())#转化成一维数组