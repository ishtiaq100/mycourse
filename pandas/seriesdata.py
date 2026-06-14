import numpy as np
import pandas as pd
labels = ['a','b','c']
mylist = [10,20,30]
arr = np.array([10,20,30])
d= {1:10,2:20,3:30}
pmylist =pd.Series(mylist)
print(pmylist)
print(pd.Series(mylist,index=labels))
print(pd.Series(arr))
print(pd.Series(d))