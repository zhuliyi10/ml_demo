import numpy
import pandas


data=pandas.read_csv('student.csv')
print(data)
print(data.to_excel('student.xls'))