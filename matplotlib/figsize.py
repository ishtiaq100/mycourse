import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = np.linspace(0,5,11)
y = x**2

fig = plt.figure(figsize=(12,8))
axis1 = fig.add_axes([0.1,0.1,.4,.8])
axis1.plot(x,y)
axis2 = fig.add_axes([0.3,0.4,.3,.3])
axis2.plot(x,y)
plt.savefig("myplot.png")
plt.show()