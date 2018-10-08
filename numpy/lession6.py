import numpy

a = numpy.array([1, 2, 3])[:, numpy.newaxis]
b = numpy.array([2, 3, 4])[:, numpy.newaxis]
print(numpy.vstack((a, b)))
print(numpy.hstack((a, b)))
