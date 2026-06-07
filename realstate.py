import numpy as np
id,status,price,bed = np.genfromtxt('RealEstate-USA.csv',delimiter=',',usecols=(0,1,2,3),unpack=True,dtype=None,skip_header=1)
print(id)
print(status)
print(price)
print(bed)


print("the mean of the prices is ",np.mean(price))

print("the median of the prices is ",np.median(price))

print("the Avg of the prices is ",np.average(price))