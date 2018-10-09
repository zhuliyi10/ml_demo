import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-3, 3, 100)
y1 = 2 * x + 1
y2 = x ** 2
x0 = 1
y0 = 3
plt.figure(num=3, figsize=(8, 5))
plt.plot(x, y2, label='linear line')
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='square line')
plt.plot([x0, x0, ], [0, y0], '--', linewidth=2, color='black')
plt.scatter([x0, ], [y0, ], s=60, color='blue')
plt.annotate(r'$2x+1=%s$' % y0, xy=(x0, y0), xycoords='data',xytext=(+30,-30),textcoords='offset points',arrowprops=dict(arrowstyle='->',connectionstyle="arc3,rad=.2"),fontsize=16)

plt.xlabel("horizontal")
plt.ylabel("vertical")

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
plt.legend(loc='best')
# ax.yaxis.set_ticks_position('left')
plt.show()
