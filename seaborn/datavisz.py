import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("RealEstate-USA.csv",delimiter=',')
df = pd.DataFrame(data)
df= df.head(10)
print(df.shape)
print(df.info())
print(df.describe())
#print(df)
sns.set_theme(style='ticks')

sns.lineplot(x='city',y='price',data=d)
plt.show()

input("Write for stop")