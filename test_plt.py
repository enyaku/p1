
import numpy as np
import matplotlib.pyplot as plt


x = np.arange(0, 10, 0.1)
y = np.exp(x)
plt.plot(x, y)
plt.title("exponential function: $ y = e^x $")
plt.ylim(0, 5000)  # yを0-5000の範囲に限定


plt.show()
