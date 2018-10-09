import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt



def animate(i):
    line.set_ydata(np.sin(x + i / 10.0))
    return line,


def init():
    line.set_ydata(np.sin(x))
    return line,

fig, ax = plt.subplots()
x = np.arange(0, 2 * np.pi, 0.01)
line, = ax.plot(x, np.sin(x))
ani = animation.FuncAnimation(fig=fig, func=animate, frames=1000, init_func=init, interval=20, blit=True)
plt.show()
