import numpy as np
arr = [[[2,5],[4,8]]]
#print(arr)
newarr = np.insert(arr,1,[35,45],axis=1)
print(newarr)