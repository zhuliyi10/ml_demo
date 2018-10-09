import matplotlib.pyplot as plt
import numpy as np

a = np.random.uniform(0.3, 0.8, 9).reshape(3, 3)
print(a)
plt.imshow(a, interpolation='nearest', cmap='bone', origin='lower')
plt.colorbar(shrink=.92)
plt.show()
