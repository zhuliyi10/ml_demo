import numpy
import pandas

dates = pandas.date_range('20181008', periods=6)
df = pandas.DataFrame(numpy.arange(24).reshape(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])
print(df)
df.iloc[2,2]=100
df.B[df.A>4]=0
df['F']=numpy.nan
df['E']=pandas.Series([1,2,3,4,5,6],index=dates)
print(df)
