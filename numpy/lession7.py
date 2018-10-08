import numpy

a = numpy.arange(12).reshape(3, 4)
print(a)
b = numpy.array_split(a, 3, axis=1)
print(b)
