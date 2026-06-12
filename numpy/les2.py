import numpy as np
#array attributes
# arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
# print(arr)
# print(arr.shape)

# print(arr.size)

# print(arr.dtype)

# #array method
# print(arr.min())
# print(arr.max())
# print(arr.sum())
# print(np.sum(arr,axis=1))

# dataarang = np.arange(1,51)
# print(dataarang)
# #print(dataarang.reshape(10,5))

# print(dataarang[11])
# arr = np.arange(1,31).reshape(6,5)
# print(arr)
# #print(arr[0:2,1:3])
# #print(arr[3:5,3:5])
# #print(arr[:,2])
# print(arr[:,0])


#boolean index
arr = np.arange(11,21)
bool_index = arr %2 == 0
print(bool_index)
arr = arr[bool_index]
print(arr)