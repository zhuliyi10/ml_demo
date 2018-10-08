import numpy

a = numpy.arange(14, 2, -1).reshape(2, 6)
print(a)
print(numpy.mean(a))
print(numpy.median(a))
print(numpy.cumsum(a))
print(numpy.diff(a))
print(numpy.nonzero(a))
print(numpy.sort(a))
print(numpy.clip(a, 5, 9))
