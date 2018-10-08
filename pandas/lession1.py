import numpy
import pandas

a = pandas.Series([1, 2, 3, 4])
print(a)

dates = pandas.date_range('20181008', periods=6)
print(dates)

df = pandas.DataFrame(numpy.arange(12).reshape(3, 4))
print(df)
print(df.describe())

print(df.sort_index(axis=1, ascending=False))

print(df.sort_values(by=1))
