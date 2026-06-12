import numpy as np
arr = np.array([
    [18,85,78],
    [19,92,88],
    [17,96,95],
    [18,65,70],
    [20,90,85]
    
    ])
print(arr.shape)
print(np.mean(arr[:,0]))

#extract all the column
print(arr[:,1])
#print the max number
print(np.max(arr[:,2]))

print(arr[arr[:,1] > 90])
arr[:,1] += 5
print(arr)
print(arr[:,1:])
print(np.mean(arr[:,1:],axis=1))

print(arr[(arr[:,1] >= 80 ) & (arr[:,2] >= 80 ) ])

arr[:,2][arr[:,2] < 70]=0
print(arr)