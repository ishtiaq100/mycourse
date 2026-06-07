import numpy as np
address,city,country = np.genfromtxt('FastFoodRestaurants.csv',delimiter=',',invalid_raise=False,usecols=(0,1,2),unpack=True,dtype=None,skip_header=1)
print(address)
print(city)
print(country)



#print("the mean of the prices is ",np.mean(city))
