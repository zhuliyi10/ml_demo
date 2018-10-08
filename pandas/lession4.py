import numpy
import pandas

dates = pandas.date_range('20181008', periods=6)
df = pandas.DataFrame(numpy.arange(24).reshape(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])
df.iloc[0,1]=numpy.nan
print(df)
print(df.isnull())