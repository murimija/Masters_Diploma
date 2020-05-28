import matplotlib.pyplot as plt
plt.plot([1, 2, 4, 4, 5], [1, 2, 8, 4, 5])
plt.show()
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()
x = np.arange(-4, 5)
y1 = x ** 2
y2 = 10 / (x ** 2 + 1)
fig, ax = plt.subplots()

ax.plot(x, y1, 'rx', x, y2, 'b+', linestyle='solid')
ax.fill_between(x, y1, y2, where=y2 > y1, interpolate=True,
                color='green', alpha=0.3)

lgnd = ax.legend(['y1', 'y2'], loc='upper center', shadow=True)
lgnd.get_frame().set_facecolor('#ffb19a')

plt.show()