import numpy
import pandas

df1 = pandas.DataFrame(numpy.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
df2 = pandas.DataFrame(numpy.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'd'])
df3 = pandas.DataFrame(numpy.ones((3, 4)) * 2, columns=['a', 'b', 'c', 'd'])
s = pandas.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
print(df1)
print(df2)
print(df3)
res = pandas.concat([df1, df2, df3], axis=0, ignore_index=True)
print(res)
df = df1.append([df2, df3], ignore_index=True)
print(df.append(s, ignore_index=True))

# df1 = pandas.DataFrame(numpy.ones((3, 4)) * 0, columns=['a', 'b', 'c', 'd'])
# df2 = pandas.DataFrame(numpy.ones((3, 4)) * 1, columns=['a', 'b', 'c', 'e'])
# print(df1)
# print(df2)
# res = pandas.concat([df1, df2], join='inner', sort=False, ignore_index=True)
# print(res)
