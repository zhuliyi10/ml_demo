import matplotlib.pyplot as plt
import numpy as np

n = 12
X = np.arange(n)
Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
Y2 = -(1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
plt.bar(X, +Y1, facecolor='red')
plt.bar(X, Y2, facecolor='blue')
for x, y in zip(X, Y1):
    plt.text(x, y, '%.2f' % y, ha='center', va='bottom')
for x, y in zip(X, Y2):
    plt.text(x, y, '%.2f' % y, ha='center', va='top')
plt.show()
