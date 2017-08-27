import numpy as np
import matplotlib.pyplot as plt


x = linspace(0,10);
amp = 2;
y = amp*cos(x);
plt.plot(x,y)
plt.xlabel(['Sine wave: ' num2str(amp) ' units in amplitude.'])

plt.show()
