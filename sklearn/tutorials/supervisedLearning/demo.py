import numpy as np

output=np.random.normal(size=(2,10))#正态分布
print(output)
output=np.random.random_sample(size=(2,10))#0--1随机数
print(output)
output=np.random.uniform(1,10,(2,2))#指定范围
print(output.dtype)
output=np.random.rand()#产生一个0--1随机数
print(output)