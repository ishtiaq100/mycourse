import numpy as np
price = np.array([200,400,300])
discount=10
newprice = price - (price*discount/100)
print(newprice)
