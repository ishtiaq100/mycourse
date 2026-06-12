import numpy as np
broked_by,status,price = np.genfromtxt("FastFoodRestaurants.csv",dtype=None,usecols=[0,1,2],skip_header=1,delimiter=",",unpack=True)
