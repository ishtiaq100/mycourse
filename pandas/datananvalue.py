import numpy as np
import pandas as pd
data={
    'A': [1,2,np.nan,4,5],
    'B': [1,2,3,np.nan,5],
    'C': [1,2,3,4,np.nan],
    'D': [np.nan,2,3,4,5]
}
df = pd.DataFrame(data)
print(df)
#check is null
print(df.isna())
print(df.isna().sum())

values={'A':0,'B':1,'C':2,'D':50}
df.fillna(value=values,inplace=True)
print(df)