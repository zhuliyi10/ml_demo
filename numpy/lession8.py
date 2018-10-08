import numpy

a = numpy.arange(4)
b = a
c = a
d = b
a[0] = 11
print(b is a)
