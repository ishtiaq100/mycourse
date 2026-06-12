import numpy as np
broked_by,status,price = np.genfromtxt("RealEstate-USA.csv",dtype=None,usecols=[0,1,2],skip_header=1,delimiter=",",unpack=True)
#price = np.genfromtxt("RealEstate-USA.csv",dtype=int,usecols=[2],skip_header=1,delimiter=",",unpack=True)
print(price)
print(status)
maxprice = np.max(price)
minprice = np.min(price)
print("Maximum Price is ",maxprice)
print("Minumum Price is ",minprice)