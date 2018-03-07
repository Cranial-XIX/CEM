
import matplotlib.pyplot as plt
import numpy as np
def f(x):
	x1 = 1/(np.sqrt(2 * np.pi / 4.0)) * np.exp(-x**2 * 2.0)
	x2 = 1/(np.sqrt(2 * np.pi )) * np.exp(-(x-2)**2/2.0)
	return x1 + x2

def f2(x):
	x1 = 1/(np.sqrt(2 * np.pi / 4.0)) * np.exp(-x**2 * 2.0)
	x2 = 1/(np.sqrt(2 * np.pi )) * np.exp(-(x-2)**2/2.0)
	return 10 * (x1 + x2)

x = np.arange(-10, 10, 0.1)
y = map(f, x)
y2 = np.exp(map(f2, x))
y2 = y2 / sum(y2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('x')
ax.set_ylabel('value')

a, = plt.plot(x, y, label='original')
b, = plt.plot(x, y2, label='scaled')
plt.legend(handles=[a, b])
plt.xticks(np.arange(-10, 10, 1))
plt.show()