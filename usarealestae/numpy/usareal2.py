import numpy as np
address, latitude , longitude , name, = np.genfromtxt("FastFoodRestaurants.csv",
                                                                delimiter=',',
                                                                usecols=(0, 4, 5, 6),
                                                                unpack=True,
                                                                dtype=None,
                                                                #dtype=('U100', 'f8', 'f8', 'U100'),   # force correct types
                                                                encoding='utf-8',
                                                                skip_header=1,
                                                                invalid_raise=False)

print(address)
print(latitude)
print(longitude)
print(name)