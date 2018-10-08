import matplotlib.pyplot as plt
import numpy
import pandas

# data=pandas.Series(numpy.random.randn(1000),index=numpy.arange(1000))
# data=data.cumsum()
# data.plot()
# plt.show()

data = pandas.DataFrame(
    numpy.random.randn(1000, 4),
    index=numpy.arange(1000),
    columns=list("ABCD")
)
data = data.cumsum()
data.plot()
plt.show()
