import numpy as np

data=np.linspace(-1,1,10,dtype=np.float32)[:,np.newaxis]
data1=np.random.normal(0,1,[2,10])
print(data)
print(data1)
print(np.matmul(data1,data))
# data=np.ones([3,1])
# data1=np.ones([1,3])
# print(data)
# print(data1)
# data2=np.matmul(data1,data)
# print(data2)